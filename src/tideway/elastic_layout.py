"""Pure geometry for stable-address, physically elastic KV/expert storage."""

from dataclasses import dataclass


def ceil_div(n: int, d: int) -> int:
    # Allocator indices can arrive as NumPy int32 scalars. Promote operands
    # before arithmetic: byte offsets routinely exceed signed 32-bit range.
    n, d = int(n), int(d)
    return (n + d - 1) // d


@dataclass(frozen=True)
class StripeLayout:
    stripes: int
    units: int
    unit_bytes: int
    dummy_tail: int = 0

    @property
    def usable(self) -> int:
        return self.units - self.dummy_tail

    @property
    def nbytes(self) -> int:
        return self.stripes * self.units * self.unit_bytes

    def chunks(self, mapped_units: int, chunk: int) -> set[int]:
        mapped_units, chunk = int(mapped_units), int(chunk)
        if not 0 <= mapped_units <= self.usable:
            raise ValueError(f"mapped units {mapped_units} outside [0, {self.usable}]")
        result = set()
        for stripe in range(self.stripes):
            base = stripe * self.units * self.unit_bytes
            for lo, hi in (
                (base, base + mapped_units * self.unit_bytes),
                (base + self.usable * self.unit_bytes, base + self.units * self.unit_bytes),
            ):
                if hi > lo:
                    result.update(range(lo // chunk, ceil_div(hi, chunk)))
        return result


def bank_bytes(slots: int, row_bytes: list[int], chunk: int) -> int:
    return sum(ceil_div(slots * row, chunk) * chunk for row in row_bytes)


def fit_slots(budget: int, row_bytes: list[int], chunk: int, maximum: int) -> int:
    lo, hi = 0, maximum
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bank_bytes(mid, row_bytes, chunk) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def high_water_from_free(free: list[int], capacity: int, reserved_prefix: int = 0) -> int:
    """One past the last live index; requires the complete free set, not merely evictable pages."""
    # Runtime supplies an already sorted NumPy view of the GPU free list. Keep
    # the validation vectorized, then find the contiguous free suffix in O(log N),
    # avoiding a Python set/139K-element scan at every agent turn boundary.
    if hasattr(free, "dtype"):
        if free.size and (
            free[0] < reserved_prefix or free[-1] >= capacity or bool((free[1:] <= free[:-1]).any())
        ):
            raise ValueError("invalid, unsorted or duplicate free indices")
        lo, hi = 0, len(free)
        offset = capacity - len(free)
        while lo < hi:
            mid = (lo + hi) // 2
            if int(free[mid]) == offset + mid:
                hi = mid
            else:
                lo = mid + 1
        return offset + lo
    if len(set(free)) != len(free) or any(i < reserved_prefix or i >= capacity for i in free):
        raise ValueError("invalid or duplicate free indices")
    available = set(free)
    high = capacity
    while high > reserved_prefix and high - 1 in available:
        high -= 1
    return high


@dataclass(frozen=True)
class SWACompactionPlan:
    """A disjoint move plan; full indices and the slot-zero sentinel never move."""

    full_indices: object
    source: object
    destination: object
    live_count: int
    high_water: int


def plan_swa_compaction(mapping, free, capacity: int, mapped_units: int) -> SWACompactionPlan:
    """Validate ownership and fill low holes with high live slots, with no copy cycles.

    ``mapping`` is the canonical full->SWA array (including its final -1 sentinel).
    There are N live slots. The destinations are free slots in [1, N], and sources
    are live slots in [N+1, capacity). These sets are disjoint by construction, so
    all stripes can copy in parallel without any KV-sized scratch allocation.
    """
    import numpy as np

    mapping, free = np.asarray(mapping), np.asarray(free)
    if mapping.ndim != 1 or free.ndim != 1 or not 1 <= mapped_units <= capacity:
        raise ValueError("invalid SWA compaction geometry")
    if (
        not mapping.size
        or mapping[-1] != -1
        or np.any(mapping[:-1] < 0)
        or np.any(mapping >= capacity)
    ):
        raise ValueError("invalid full-to-SWA mapping")
    full = np.flatnonzero(mapping > 0)
    live = mapping[full].astype(np.int64, copy=False)
    # Every physical slot has exactly one owner or one free-list entry. In
    # particular, never trust an apparently free destination without this check.
    ownership = np.sort(np.concatenate((live, free)))
    if not np.array_equal(ownership, np.arange(1, capacity)):
        raise ValueError("SWA ownership/free-list mismatch or duplicate slot")
    if live.size and int(live.max()) >= mapped_units:
        raise ValueError("live SWA slot is outside physically mapped prefix")
    count = len(live)
    tail = live > count
    holes = np.sort(free[free <= count]).astype(np.int64, copy=False)
    source = live[tail]
    if len(source) != len(holes):
        raise AssertionError("SWA tail/hole conservation failed")
    return SWACompactionPlan(full[tail], source, holes, count, int(live.max()) + 1 if count else 1)
