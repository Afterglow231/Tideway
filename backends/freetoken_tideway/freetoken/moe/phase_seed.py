"""Opt-in final-prefill tail selection and strictly current-layer refresh.

Only the request that has actually arrived is consulted. Intermediate chunks
retain the existing all-token, empty-only policy. Refresh never removes an
expert observed in the final tail and never changes another layer's residency.
"""

from __future__ import annotations

import triton
import triton.language as tl


def is_final_single_prefill(batch, chunked_req_type):
    return bool(
        batch is not None
        and batch.is_prefill
        and len(batch.reqs) == 1
        and not isinstance(batch.reqs[0], chunked_req_type)
        and getattr(batch.reqs[0], "mm_embeds", None) is None
    )


def current_phase_is_final():
    from freetoken.core import get_global_ctx
    from freetoken.scheduler.prefill import ChunkedReq

    return is_final_single_prefill(get_global_ctx().batch, ChunkedReq)


def tail_routing_view(ids, tokens, is_final):
    return ids[-tokens:] if is_final and tokens > 0 and ids.ndim == 2 else ids


@triton.jit
def _advance_refresh_clock_if_full(quotas, clock, L: tl.constexpr, BLOCK_L: tl.constexpr):
    layer = tl.arange(0, BLOCK_L)
    count = tl.sum(tl.load(quotas + layer, layer < L, 0))
    # _prepare_quotas already advanced when any empty capacity was available.
    if count == 0:
        tl.store(clock, tl.load(clock) + 1)


@triton.jit(do_not_specialize=["layer", "N", "refresh_rows"])
def _route_phase_histogram(
    ids,
    hist,
    quotas,
    layer,
    N,
    refresh_rows,
    WIDTH: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(quotas + layer) > 0 or refresh_rows > 0:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offset = (i // WIDTH) * ROW_STRIDE + (i % WIDTH) * COL_STRIDE
        expert = tl.load(ids + offset, i < N, 0).to(tl.int32)
        counts = tl.histogram(expert, BLOCK_E, mask=(i < N) & (expert >= 0) & (expert < E))
        bins = tl.arange(0, BLOCK_E)
        tl.atomic_add(hist + bins, counts, bins < E)


@triton.jit(do_not_specialize=["layer", "exclude_lo", "exclude_hi", "refresh_rows"])
def _admit_empty_and_refresh(
    hist,
    slots,
    owners,
    usage,
    clock,
    pinned,
    quotas,
    src,
    dst,
    count_out,
    empty_counters,
    refresh_counters,
    layer,
    exclude_lo,
    exclude_hi,
    refresh_rows,
    E: tl.constexpr,
    C: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    quota = tl.load(quotas + layer)
    empty_count = tl.full((), 0, tl.int32)
    replace_count = tl.full((), 0, tl.int32)
    if quota > 0 or refresh_rows > 0:
        e = tl.arange(0, BLOCK_E)
        h = tl.load(hist + e, e < E, 0).to(tl.int64)
        previous = tl.load(slots + layer * E + e, e < E, 0)
        candidate = (e < E) & (h > 0) & (previous < 0)
        key = tl.where(candidate, h * E + E - 1 - e, -1)
        ranked = tl.sort(key, descending=True)
        chosen = tl.where(ranked >= 0, E - 1 - ranked % E, 0).to(tl.int32)
        candidate_count = tl.sum(candidate.to(tl.int32))

        s = tl.arange(0, BLOCK_C)
        old = tl.load(owners + s, s < C, -1)
        pin = tl.load(pinned + s, s < C, 1)
        allowed = (s >= 2 * E) & (s < C) & ((s < exclude_lo) | (s >= exclude_hi))
        free = allowed & (old < 0) & (pin == 0)
        free_count = tl.sum(free.to(tl.int32))
        empty_count = tl.minimum(quota, tl.minimum(free_count, candidate_count))
        free_rank = tl.cumsum(free.to(tl.int32)) - 1
        tl.store(dst + free_rank, s, free & (free_rank < empty_count))

        # Consider only existing slots that the CURRENT layer actually owns.
        # A zero recent count is deliberately stricter than "less frequent";
        # all positive-tail residents remain untouched even if no victim exists.
        valid_slot = (e < E) & (previous >= 2 * E) & (previous < C)
        slot_allowed = (previous < exclude_lo) | (previous >= exclude_hi)
        owner = tl.load(owners + previous, valid_slot, -1)
        pin_e = tl.load(pinned + previous, valid_slot, 1)
        recency = tl.load(usage + previous, valid_slot, 0)
        now = tl.load(clock)
        victim = (
            valid_slot
            & slot_allowed
            & (owner == layer * E + e)
            & (pin_e == 0)
            & (h == 0)
            & (recency < now)
        )
        # Bounded age preserves LRU order over practical runs and avoids overflow
        # even if external bookkeeping contains a huge timestamp. Low ID breaks
        # an exact age tie; no other layer is a fallback victim.
        age = tl.minimum(tl.maximum(now - recency, 0), 1073741823).to(tl.int64)
        victim_key = tl.where(victim, age * E + E - 1 - e, -1)
        ranked_victim = tl.sort(victim_key, descending=True)
        victim_e = tl.where(ranked_victim >= 0, E - 1 - ranked_victim % E, 0).to(tl.int32)
        replace_count = tl.minimum(
            refresh_rows, tl.minimum(tl.sum(victim.to(tl.int32)), candidate_count - empty_count)
        )
        victim_slot = tl.load(slots + layer * E + victim_e, e < replace_count, 0)
        tl.store(dst + empty_count + e, victim_slot, e < replace_count)
        # This scratch list is complete before map updates or indexed rereads.
        tl.debug_barrier()
        tl.store(slots + layer * E + victim_e, -1, e < replace_count)
        tl.debug_barrier()
        total = empty_count + replace_count
        target = tl.load(dst + e, e < total, 0)
        gid = layer * E + chosen
        tl.store(slots + gid, target, e < total)
        tl.store(owners + target, gid, e < total)
        tl.store(usage + target, now, e < total)
        tl.store(src + e, (layer % 2) * E + chosen, e < total)
    tl.store(count_out, (empty_count + replace_count).to(tl.int64))
    tl.store(empty_counters + 2, tl.load(empty_counters + 2) + empty_count.to(tl.int64))
    tl.store(empty_counters + 3, tl.load(empty_counters + 3) + (empty_count > 0).to(tl.int64))
    tl.store(refresh_counters + 0, tl.load(refresh_counters + 0) + replace_count.to(tl.int64))
    tl.store(refresh_counters + 1, tl.load(refresh_counters + 1) + (replace_count > 0).to(tl.int64))
