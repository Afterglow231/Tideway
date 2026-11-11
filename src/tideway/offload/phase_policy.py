"""Explicit, bounded, causal phase policy for Tideway's existing expert pager.

This module allocates CPU metadata only. It neither allocates weight payloads nor
reads future routes. ``off``/``exclusive`` leave the legacy implementation intact.
GPU transfer primitives, ordering, RAM leases and full-layer prefill are unchanged.
"""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import islice
import math
import numpy as np


@dataclass(frozen=True)
class PhasePolicyConfig:
    mode: str = "off"
    mirror_rows: int = 64
    mirror_window: int = 32
    decay: float = 0.97
    victim_window: int = 64
    writeback_min_heat: float = 0.0
    mirror_value: float = 0.35
    prefill_hot_guard: bool = True
    prefill_sample_tokens: int = 0
    prefill_weight: float = 0.15
    seed_empty_tail: bool = False
    phase_snapshot: bool = True
    restore_rows: int = 0
    duplicate_policy: str = "auto"
    gpu_victim_window: int = 0  # phase-mirror only; 0 retains original GPU LRU

    def __post_init__(self):
        for name in (
            "mirror_rows",
            "mirror_window",
            "victim_window",
            "prefill_sample_tokens",
            "restore_rows",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("prefill_hot_guard", "seed_empty_tail", "phase_snapshot"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("decay", "writeback_min_heat", "mirror_value", "prefill_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if type(self.gpu_victim_window) is not int or self.gpu_victim_window < 0:
            raise ValueError("gpu_victim_window must be a nonnegative integer")
        if self.gpu_victim_window and self.mode != "phase-mirror":
            raise ValueError("gpu_victim_window requires phase-mirror mode")
        if self.duplicate_policy not in ("auto", "legacy", "gpu-stay"):
            raise ValueError("unknown duplicate eviction policy")
        if self.mode not in ("off", "exclusive", "mirror-only", "heat-aware", "phase-mirror"):
            raise ValueError("unknown phase policy mode")
        if self.mirror_rows < 0 or self.mirror_window < 1 or self.victim_window < 1:
            raise ValueError("mirror rows >= 0 and positive windows required")
        if not 0 < self.decay < 1:
            raise ValueError("decay must lie strictly between zero and one")
        if self.writeback_min_heat < 0 or self.mirror_value < 0:
            raise ValueError("nonnegative admission values required")
        if self.restore_rows < 0:
            raise ValueError("restore_rows must be nonnegative")
        if self.prefill_sample_tokens < 0 or not 0 <= self.prefill_weight <= 1:
            raise ValueError("invalid prefill observation settings")
        if not all(
            math.isfinite(float(x))
            for x in (self.decay, self.writeback_min_heat, self.mirror_value, self.prefill_weight)
        ):
            raise ValueError("finite policy parameters required")


class PhasePolicy:
    """Layer-normalized online reuse state, retaining no future information.

    Mirrors are nominated by actual GPU misses satisfied from RAM, after an
    expert has been seen at least twice in Decode. A short, bounded nomination
    protects experts oscillating between tiers, rather than permanently
    duplicating the hottest GPU residents. It does not reserve extra RAM rows.
    """

    def __init__(self, manager, config):
        self.manager, self.config = manager, config
        self.L, self.E = int(manager.L), int(manager.E)
        self.heat = np.zeros((self.L, self.E), dtype=np.float64)
        self.prefill_heat = np.zeros_like(self.heat)
        self.decode_calls = np.zeros(self.L, dtype=np.int64)
        self.heat_mass = np.zeros(self.L, dtype=np.float64)
        self.visits = np.zeros((self.L, self.E), dtype=np.int64)
        self.clock = 0
        self.phase = "decode"
        self.small_prefill_depth = 0
        self.mirrors = OrderedDict()
        # phase-mirror maintains nonprotected duplicate candidates by actual
        # GPU/RAM ownership events, avoiding a scan of all nominated mirrors
        # for every layer's optional writeback.
        self.phase_unprotected_duplicates = OrderedDict()
        self.pending_mirror_nominations = ()
        self.decode_snapshot = None
        self.retained_duplicates = OrderedDict()
        self.restore_pending = False
        self.reset_stats()
        if self.config.mode == "phase-mirror":
            self.phase_unprotected_duplicates = OrderedDict(
                (e, None) for e in self.manager.ram.duplicates if self._valid_duplicate(e)
            )

    def reset_stats(self):
        # Per-request counters reset; learned state belongs to the service.
        self.stats = dict(
            decode_observations=0,
            prefill_observations=0,
            mirror_nominations=0,
            mirror_candidate_skips=0,
            writeback_rejected=0,
            writeback_existing=0,
            writeback_admitted=0,
            demand_unique_evictions=0,
            gpu_heat_evictions=0,
            prefill_guarded_rows=0,
            phase_snapshots=0,
            snapshot_requested_rows=0,
            snapshot_restored_rows=0,
            snapshot_skipped_rows=0,
            snapshot_restored_bytes=0,
            snapshot_already_gpu_rows=0,
            snapshot_restore_fences=0,
            duplicate_victim_choices=0,
            duplicate_victim_changed=0,
            retained_duplicate_gpu_evictions=0,
            ram_window_fallbacks=0,
            phase_mirror_gpu_retirements=0,
            phase_mirror_ram_retirements=0,
            phase_mirror_forced_evictions=0,
            phase_mirror_boundary_rebuilds=0,
            gpu_heat_selection_calls=0,
            gpu_heat_candidate_rows=0,
            gpu_heat_full_window_calls=0,
            gpu_heat_cold_fallbacks=0,
        )

    def reset_cold(self):
        self.heat.fill(0)
        self.prefill_heat.fill(0)
        self.decode_calls.fill(0)
        self.heat_mass.fill(0)
        self.visits.fill(0)
        self.clock = 0
        self.phase = "decode"
        self.small_prefill_depth = 0
        self.mirrors.clear()
        self.phase_unprotected_duplicates.clear()
        self.pending_mirror_nominations = ()
        self.retained_duplicates.clear()
        self.decode_snapshot = None
        self.restore_pending = False
        self.reset_stats()

    def observe_decode(self, layer, ids, missing):
        layer = int(layer)
        ids = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
        local = ids - layer * self.E
        if layer < 0 or layer >= self.L or np.any(local < 0) or np.any(local >= self.E):
            raise ValueError("decode observation contains an expert from another layer")
        if self.config.mode == "phase-mirror":
            self._enter_decode_phase()
        self.phase = "decode"
        self.clock += 1
        self.heat[layer] *= self.config.decay
        self.heat[layer, local] += 1.0 - self.config.decay
        self.heat_mass[layer] = self.config.decay * self.heat_mass[layer] + (
            1.0 - self.config.decay
        ) * len(local)
        self.visits[layer, local] += 1
        self.decode_calls[layer] += 1
        self.stats["decode_observations"] += 1
        if self.config.mode == "phase-mirror":
            # Defer quota admission until GPU access has retired real victims.
            # Otherwise the incoming miss needlessly pushes out a still-useful
            # FIFO mirror immediately before the outgoing victim frees a slot.
            self.pending_mirror_nominations = tuple(
                int(e)
                for e in np.asarray(missing).reshape(-1)
                if e in self.manager.ram.where and self.visits[layer, int(e) % self.E] >= 2
            )
            return
        if not self.config.mirror_rows:
            return
        for expert in np.asarray(missing).reshape(-1):
            e = int(expert)
            if e in self.manager.ram.where and self.visits[layer, e % self.E] >= 2:
                self.mirrors[e] = self.clock + self.config.mirror_window * self.L
                self.mirrors.move_to_end(e)
                self.stats["mirror_nominations"] += 1
        while len(self.mirrors) > min(self.config.mirror_rows, self.manager.ram.capacity):
            self.mirrors.popitem(last=False)

    def observe_prefill(self, layer, counts):
        """Receive true routed-token counts; whole-layer loads never count as hits."""
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != (self.E,) or np.any(counts < 0) or not np.all(np.isfinite(counts)):
            raise ValueError("one finite, nonnegative routed count per expert required")
        total = float(counts.sum())
        if total:
            self.prefill_heat[int(layer)] = counts / total
        self.stats["prefill_observations"] += 1

    def scores(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        layers = ids // self.E
        calls = self.decode_calls[layers]
        d = self.heat.reshape(-1)[ids] / np.maximum(self.heat_mass[layers], 1e-30)
        # Separate normalization means long P cannot swamp D merely by length.
        p = self.prefill_heat.reshape(-1)[ids]
        weight = self.config.prefill_weight / (1.0 + calls / 8.0)
        return (1.0 - weight) * d + weight * p

    def score(self, e):
        layer, local = divmod(int(e), self.E)
        calls = int(self.decode_calls[layer])
        d = float(self.heat[layer, local]) / max(float(self.heat_mass[layer]), 1e-30)
        w = self.config.prefill_weight / (1.0 + calls / 8.0)
        return (1.0 - w) * d + w * float(self.prefill_heat[layer, local])

    def _valid_duplicate(self, e):
        gpu = self.manager.gpu
        slot = int(gpu.where[int(e)])
        return (
            e in self.manager.ram.where
            and 0 <= slot < len(gpu.slots)
            and int(gpu.slots[slot]) == int(e)
        )

    def _phase_refresh_duplicate(self, e, touch=False):
        e = int(e)
        if not self._valid_duplicate(e):
            self.phase_unprotected_duplicates.pop(e, None)
            if e in self.mirrors:
                self.mirrors.pop(e, None)
                self.stats["phase_mirror_gpu_retirements"] += 1
        elif e in self.mirrors:
            self.phase_unprotected_duplicates.pop(e, None)
        else:
            self.phase_unprotected_duplicates[e] = None
            if touch:
                self.phase_unprotected_duplicates.move_to_end(e)

    def _enter_decode_phase(self):
        if self.phase == "decode":
            return
        self.phase = "decode"
        # One bounded O(GPU slots) reconciliation at a P->D boundary. There is
        # no per-layer scan of the (potentially 1152-entry) nomination table.
        for e in tuple(self.mirrors):
            if not self._valid_duplicate(e):
                self.mirrors.pop(e, None)
                self.stats["phase_mirror_gpu_retirements"] += 1
        self.phase_unprotected_duplicates = OrderedDict(
            (e, None)
            for e in self.manager.ram.duplicates
            if e not in self.mirrors and self._valid_duplicate(e)
        )
        self.stats["phase_mirror_boundary_rebuilds"] += 1

    def _phase_after_gpu_access(self, ids):
        gpu = self.manager.gpu
        # Real cache victim identities are available before runtime executes
        # save_victims; removing nominations here keeps RAM-only rows eligible.
        for e in getattr(gpu, "last_evicted", ()):
            e = int(e)
            self.phase_unprotected_duplicates.pop(e, None)
            if e in self.mirrors:
                self.mirrors.pop(e, None)
                self.stats["phase_mirror_gpu_retirements"] += 1
        limit = min(self.config.mirror_rows, self.manager.ram.capacity, len(gpu.slots))
        for e in self.pending_mirror_nominations:
            if not limit or not self._valid_duplicate(e):
                continue
            # Lifetime follows actual GPU residency, not a fixed token TTL.
            self.mirrors[e] = self.clock
            self.mirrors.move_to_end(e)
            self.phase_unprotected_duplicates.pop(e, None)
            self.stats["mirror_nominations"] += 1
        self.pending_mirror_nominations = ()
        while len(self.mirrors) > limit:
            e, _ = self.mirrors.popitem(last=False)
            self._phase_refresh_duplicate(e)
        for e in ids:
            self._phase_refresh_duplicate(int(e))

    def _phase_ram_victim(self, protect, writeback):
        ram = self.manager.ram
        if self.phase != "decode":
            # P temporarily uses its original union-capacity policy. Preserve
            # exactly the same exclusions and LRU order, but inline membership
            # checks: long P often skips thousands of anchors for each miss.
            # Anchors short-circuit before leases, with locally bound tables.
            anchors, leases = ram.anchors, ram.leases
            victim = next(
                (
                    e
                    for e in ram.duplicates
                    if e not in protect and e not in anchors and e not in leases
                ),
                None,
            )
            if victim is not None or writeback:
                return victim
            return next(
                (e for e in ram.lru if e not in protect and e not in anchors and e not in leases),
                None,
            )
        # Event-maintained candidates contain no mirrors, so a full mirror
        # working set makes this test O(1), rather than scanning 1152 rows for
        # every GPU victim that would otherwise be written back.
        victim = next(
            (e for e in self.phase_unprotected_duplicates if self._eligible(ram, e, protect)), None
        )
        if victim is not None or writeback:
            return victim  # Optional writeback NEVER consumes a live mirror.
        victim = next(
            (e for e in ram.lru if self._eligible(ram, e, protect) and not self.is_mirror(e)), None
        )
        if victim is not None:
            return victim
        # Mandatory demand cannot deadlock when every legal row is a mirror.
        victim = next((e for e in ram.lru if self._eligible(ram, e, protect)), None)
        if victim is not None:
            self.stats["phase_mirror_forced_evictions"] += 1
        return victim

    def is_mirror(self, e):
        if self.config.mode == "phase-mirror":
            if self.phase != "decode" or e not in self.mirrors:
                return False
            if not self._valid_duplicate(e):
                self._phase_refresh_duplicate(e)
                return False
            return True
        expires = self.mirrors.get(int(e))
        if expires is None:
            return False
        if expires < self.clock or e not in self.manager.ram.where:
            self.mirrors.pop(int(e), None)
            return False
        return True

    @staticmethod
    def _eligible(ram, e, protect):
        return e not in protect and e not in ram.leases and e not in ram.anchors

    def note_existing_victim(self, e):
        if self.config.mode == "phase-mirror":
            self._phase_refresh_duplicate(e)
        self.stats["writeback_existing"] += 1
        if e in self.retained_duplicates:
            self.retained_duplicates.pop(e, None)
            self.stats["retained_duplicate_gpu_evictions"] += 1

    def _ram_victim(self, protect, writeback=False, incoming=None):
        if self.config.mode == "phase-mirror":
            return self._phase_ram_victim(protect, writeback)
        ram = self.manager.ram
        if self.config.mode == "mirror-only" and self.config.duplicate_policy != "gpu-stay":
            # This option changes mirror protection only: no heat-ranked
            # unique RAM victims and no bounded-window admission side effects.
            victim = next(
                (
                    e
                    for e in ram.duplicates
                    if self._eligible(ram, e, protect) and not self.is_mirror(e)
                ),
                None,
            )
            if victim is not None or writeback:
                return victim
            victim = next(
                (e for e in ram.lru if self._eligible(ram, e, protect) and not self.is_mirror(e)),
                None,
            )
            if victim is not None:
                return victim
            # Required demand can finally reclaim a protected mirror when no
            # other legal row remains; optional writeback never forces this.
            return next((e for e in ram.lru if self._eligible(ram, e, protect)), None)
        duplicates, mirror_candidates = [], []
        # Rank at most one bounded window, not the whole expert table.
        for e in islice(ram.duplicates, self.config.victim_window):
            if not self._eligible(ram, e, protect):
                continue
            if self.is_mirror(e):
                mirror_candidates.append(e)
                self.stats["mirror_candidate_skips"] += 1
            else:
                duplicates.append(e)
        if duplicates:
            self.stats["duplicate_victim_choices"] += 1
            legacy = duplicates[0]
            hot_policy = self.config.duplicate_policy == "gpu-stay" or (
                self.config.duplicate_policy == "auto" and self.config.mode == "heat-aware"
            )
            if hot_policy:
                # Delete a copy likely to stay on GPU; preserve a soon-evicted
                # GPU tail's RAM copy. Vector scoring avoids per-expert Python.
                gpu = self.manager.gpu
                ids = np.asarray(duplicates, dtype=np.int64)
                ranks = np.lexsort(
                    (np.arange(len(ids)), -gpu.usage[gpu.where[ids]], -self.scores(ids))
                )
                selected = int(ids[ranks[0]])
                if selected != legacy:
                    self.stats["duplicate_victim_changed"] += 1
                    self.retained_duplicates[legacy] = None
                    self.retained_duplicates.move_to_end(legacy)
                    while len(self.retained_duplicates) > len(gpu.slots):
                        self.retained_duplicates.popitem(last=False)
                return selected
            return legacy
        if writeback:
            # Optional admission does not need a full-table fallback or to
            # shrink union capacity by evicting an exclusive RAM expert.
            if mirror_candidates and incoming is not None:
                values = self.scores(np.asarray(mirror_candidates, dtype=np.int64))
                index = int(np.argmin(values))
                if self.score(incoming) > self.config.mirror_value * float(values[index]):
                    return mirror_candidates[index]
            return None
        if self.config.mode == "mirror-only":
            # Explicit gpu-stay affects duplicate selection only. Unique RAM
            # replacement remains LRU in this independently measurable arm.
            victim = next(
                (e for e in ram.lru if self._eligible(ram, e, protect) and not self.is_mirror(e)),
                None,
            )
            if victim is not None:
                return victim
            return next((e for e in ram.lru if self._eligible(ram, e, protect)), None)
        candidates = list(mirror_candidates)
        for e in islice(ram.lru, self.config.victim_window):
            if e not in ram.duplicates and self._eligible(ram, e, protect):
                candidates.append(e)
        if not candidates:
            # A window can contain only current-route/leased/anchored rows.
            # Demand must still make progress: find the first legal row,
            # without globally ranking it. This exceptional path is counted.
            self.stats["ram_window_fallbacks"] += 1
            return next((e for e in ram.lru if self._eligible(ram, e, protect)), None)
        ids = np.asarray(candidates, dtype=np.int64)
        values = self.scores(ids)
        values *= np.asarray(
            [self.config.mirror_value if e in ram.duplicates else 1.0 for e in candidates]
        )
        return candidates[int(np.argmin(values))]

    def place(self, e, protect, on_gpu, anchor, writeback=False):
        ram = self.manager.ram
        if e in ram.where:
            if writeback:
                self.stats["writeback_existing"] += 1
            result = ram.place(e, protect, on_gpu=on_gpu, anchor=anchor)
            if self.config.mode == "phase-mirror":
                self._phase_refresh_duplicate(e, touch=on_gpu)
            return result
        if writeback and not self.manager.writeback:
            self.stats["writeback_rejected"] += 1
            return None
        victim = None
        if not ram.free:
            if (
                writeback
                and self.config.mode == "heat-aware"
                and self.score(e) < self.config.writeback_min_heat
            ):
                self.stats["writeback_rejected"] += 1
                return None
            victim = self._ram_victim(protect, writeback=writeback, incoming=e)
            if victim is None:
                if writeback:
                    self.stats["writeback_rejected"] += 1
                    return None
                raise RuntimeError("no unprotected RAM slot for phase policy")
            if not writeback and victim not in ram.duplicates:
                self.stats["demand_unique_evictions"] += 1
            if self.config.mode == "phase-mirror":
                self.phase_unprotected_duplicates.pop(victim, None)
                if victim in self.mirrors:
                    self.stats["phase_mirror_ram_retirements"] += 1
            self.mirrors.pop(victim, None)
            self.retained_duplicates.pop(victim, None)
        if (
            self.config.mode == "heat-aware"
            and self.phase == "prefill"
            and self.config.prefill_hot_guard
        ):
            # D priority controls which existing rows enter the guard first;
            # it must not remove P's remaining cross-chunk retention capacity.
            # Cold streamed rows may fill unused guard positions without ever
            # counting as Decode observations. Keep one whole layer recyclable.
            anchor = not on_gpu and len(ram.anchors) < max(0, ram.capacity - self.E)
        result = ram.place(e, protect, on_gpu=on_gpu, anchor=anchor, victim_override=victim)
        if self.config.mode == "phase-mirror":
            self._phase_refresh_duplicate(e, touch=on_gpu)
        if writeback:
            self.stats["writeback_admitted"] += 1
        return result

    def _decode_gpu_heat_access(self, ids):
        """GPU-only opt-in; no RAM policy, payload allocation, or future data.

        Candidates are noncurrent, nonreserved occupied slots. The full-GPU
        case avoids an initial LRU sort; one deterministic lexicographic sort
        chooses (least D probability, oldest use, smallest physical slot).
        P observations do not enter these scores. Cold heat exactly delegates
        to the existing Cache.access, rather than synthesizing a new mapping.
        """
        gpu = self.manager.gpu
        if not self.heat_mass.any():
            self.stats["gpu_heat_cold_fallbacks"] += 1
            return gpu.access(ids)
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not len(ids):
            return gpu.access(ids)  # Preserve native invalid-empty behavior.
        if len(ids) > len(gpu.slots) or ids.min() < 0 or ids.max() >= len(gpu.where):
            raise ValueError("invalid atomic layer request")
        positions = gpu.where[ids]
        missing = ids[positions < 0]
        hits = positions[positions >= 0]
        gpu.step += 1
        gpu.usage[hits] = gpu.step
        if not len(missing):
            gpu.last_evicted = np.empty(0, dtype=np.int64)
            return 0
        available = np.flatnonzero((gpu.usage != gpu.step) & (gpu.usage < (1 << 61)))
        if len(available) < len(missing):
            raise RuntimeError("insufficient nonreserved GPU slots for atomic layer")
        free = available[gpu.slots[available] < 0]
        if len(free):
            free = free[np.argsort(gpu.usage[free], kind="stable")]
        victims = free[: len(missing)].tolist()
        remaining = len(missing) - len(victims)
        if remaining:
            occupied = available[gpu.slots[available] >= 0]
            window = max(remaining, int(self.config.gpu_victim_window))
            if window >= len(occupied):
                pool = occupied
                self.stats["gpu_heat_full_window_calls"] += 1
            else:
                order = np.argsort(gpu.usage[occupied], kind="stable")
                pool = occupied[order[:window]]
            owners = gpu.slots[pool]
            layers = owners // self.E
            # EMA normalized per layer; only causally observed Decode routes.
            keys = self.heat.reshape(-1)[owners] / np.maximum(self.heat_mass[layers], 1e-30)
            rank = np.lexsort((pool, gpu.usage[pool], keys))
            victims.extend(pool[rank[:remaining]].tolist())
            self.stats["gpu_heat_selection_calls"] += 1
            self.stats["gpu_heat_candidate_rows"] += len(pool)
            self.stats["gpu_heat_evictions"] += remaining
        victims = np.asarray(victims, dtype=np.int64)
        old = gpu.slots[victims]
        gpu.last_evicted = old[old >= 0].copy()
        gpu.where[old[old >= 0]] = -1
        gpu.slots[victims] = missing
        gpu.where[missing] = victims
        gpu.usage[victims] = gpu.step
        return len(missing)

    def gpu_access(self, ids):
        gpu = self.manager.gpu
        if self.config.mode != "heat-aware":
            heat_enabled = (
                self.config.mode == "phase-mirror"
                and self.config.gpu_victim_window > 0
                and self.phase == "decode"
                and self.small_prefill_depth == 0
            )
            result = self._decode_gpu_heat_access(ids) if heat_enabled else gpu.access(ids)
            if self.config.mode == "phase-mirror":
                self._phase_after_gpu_access(ids)
            return result
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if len(ids) == 0:
            return 0
        if len(ids) > len(gpu.slots) or ids.min() < 0 or ids.max() >= len(gpu.where):
            raise ValueError("invalid atomic layer request")
        gpu.step += 1
        positions = gpu.where[ids]
        hits = positions[positions >= 0]
        gpu.usage[hits] = gpu.step
        missing = ids[positions < 0]
        if not len(missing):
            gpu.last_evicted = np.empty(0, dtype=np.int64)
            return 0
        available = np.flatnonzero((gpu.usage != gpu.step) & (gpu.usage < (1 << 61)))
        if len(available) < len(missing):
            raise RuntimeError("insufficient nonreserved GPU slots for atomic layer")
        # Preserve LRU generations, with heat only breaking choices inside the
        # oldest bounded candidate window; no full-universe CPU scan.
        order = available[np.argsort(gpu.usage[available], kind="stable")]
        free = order[gpu.slots[order] < 0]
        count = len(missing)
        victims = list(free[:count])
        remain = count - len(victims)
        if remain:
            occupied = order[gpu.slots[order] >= 0]
            pool = occupied[: max(remain, self.config.victim_window)]
            keys = self.scores(gpu.slots[pool])
            rank = np.lexsort((pool, gpu.usage[pool], keys))
            victims.extend(pool[rank[:remain]].tolist())
            self.stats["gpu_heat_evictions"] += remain
        victims = np.asarray(victims, dtype=np.int64)
        old = gpu.slots[victims]
        gpu.last_evicted = old[old >= 0].copy()
        gpu.where[old[old >= 0]] = -1
        gpu.slots[victims] = missing
        gpu.where[missing] = victims
        gpu.usage[victims] = gpu.step
        return len(missing)

    @contextmanager
    def small_prefill(self):
        """Explicit native prefill context, including a one-token append.

        The DS native tiny-prefill path calls ensure just like Decode does.
        Phase comes from the caller, never guessed from tensor dimensions.
        Its already-read CPU routes update P only and cannot trigger D restore.
        """
        self.capture_decode_snapshot()
        self.small_prefill_depth += 1
        try:
            yield
        finally:
            self.small_prefill_depth -= 1

    def capture_decode_snapshot(self):
        """Capture once per D->P transition, before any borrowed slots change.

        Snapshot entries are desired expert identities and priorities, never
        reusable physical mappings. The single metadata object is replaced,
        so repeated conversations cannot grow snapshot storage.
        """
        if self.phase == "prefill":
            return
        self.phase = "prefill"
        if self.config.mode == "phase-mirror":
            self.pending_mirror_nominations = ()
            self.phase_unprotected_duplicates.clear()
        if not self.config.phase_snapshot:
            return
        gpu = self.manager.gpu
        slots = np.flatnonzero(gpu.slots >= 0)
        ids = gpu.slots[slots].copy()
        masses = self.heat_mass[ids // self.E]
        priority = self.heat.reshape(-1)[ids] / np.maximum(masses, 1e-30)
        recency = gpu.usage[slots].copy()
        rank = np.lexsort((-recency, -priority))
        self.decode_snapshot = dict(
            heat=self.heat.copy(),
            heat_mass=self.heat_mass.copy(),
            ids=ids[rank],
            priority=priority[rank],
            recency=recency[rank],
            clock=int(self.clock),
        )
        self.restore_pending = True
        self.stats["phase_snapshots"] += 1

    def restore_snapshot(self, protect):
        """Optionally restore RAM-resident D targets with real, budgeted H2D.

        No SSD reads are speculated, and no payload is assumed valid merely
        because a prior physical slot had its expert ID. Source RAM ownership
        is checked again after normal victim handling. This runs on the same
        existing stream as demand loading, inside the measured ensure call.
        """
        if self.config.mode == "phase-mirror":
            self._enter_decode_phase()
        if not self.restore_pending:
            return
        self.restore_pending = False
        snapshot = self.decode_snapshot
        budget = int(self.config.restore_rows)
        if snapshot is None or not budget:
            return
        manager, ram, gpu = self.manager, self.manager.ram, self.manager.gpu
        protected = set(int(e) for e in protect)
        targets = []
        for e, priority in zip(snapshot["ids"], snapshot["priority"]):
            e = int(e)
            if priority <= 0:
                continue
            if gpu.where[e] >= 0:
                self.stats["snapshot_already_gpu_rows"] += 1
                continue
            if e in protected:
                # Current demand is handled by normal ensure, with no duplicate
                # accounting or speculative admission of its missing payload.
                continue
            if len(targets) >= budget:
                break
            self.stats["snapshot_requested_rows"] += 1
            if e not in ram.where or e in ram.leases:
                self.stats["snapshot_skipped_rows"] += 1
                continue
            targets.append(e)
        if not targets:
            return
        active = [e for e in protected if gpu.where[e] >= 0]
        available = int(np.count_nonzero(gpu.usage < (1 << 61))) - len(protected)
        if len(targets) > available:
            self.stats["snapshot_skipped_rows"] += len(targets) - max(0, available)
            targets = targets[: max(0, available)]
        if not targets:
            return
        old = gpu.slots.copy()
        # The atomic cache operation protects all existing current-route hits.
        # No missing current-route IDs enter mappings until regular demand copy.
        self.gpu_access(np.asarray(active + targets, dtype=np.int64))
        destinations = [int(gpu.where[e]) for e in targets]
        pairs = [(int(old[slot]), slot) for slot in destinations if old[slot] >= 0]
        for e, _ in pairs:
            ram.duplicates.pop(e, None)
        protected.update(targets)
        for e in targets:
            ram.duplicates[e] = None
        manager.save_victims(pairs, protected, "decode")
        rows = [ram.where[e] for e in targets]  # protected rows must still exist
        if manager.dma_decode:
            manager.mover.ram_to_gpu_dma(destinations, rows)
        else:
            manager.mover.ram_to_gpu(destinations, rows)
        # Restore runs in the middle of ensure, before its normal RAM allocator
        # resumes. The local protect set cannot outlive this function; both
        # restored H2D source rows and preceding writeback DtoH destinations
        # must be finished before ordinary demand can recycle them. Unlike
        # end-of-ensure demand uploads, no next-layer raw.cpu fence intervenes.
        # One explicit, timed phase-boundary fence is deliberately conservative
        # and introduces no asynchronous prefetch/writeback mechanism.
        import torch

        torch.cuda.current_stream().synchronize()
        self.stats["snapshot_restore_fences"] += 1
        manager.stats["h2d_rows"] += len(targets)
        manager.stats["decode_h2d_rows"] += len(targets)
        self.stats["snapshot_restored_rows"] += len(targets)
        self.stats["snapshot_restored_bytes"] += len(targets) * int(manager.geo.row_bytes)

    def begin_prefill(self):
        self.capture_decode_snapshot()
        ram = self.manager.ram
        if self.config.mode != "heat-aware" or not self.config.prefill_hot_guard:
            return
        eligible = [e for e in ram.lru if self.manager.gpu.where[e] < 0]
        limit = max(0, ram.capacity - self.E)
        hot = [e for e in eligible if self.heat.reshape(-1)[e] > 0]
        hot.sort(key=self.score, reverse=True)
        chosen = set(hot[:limit])
        # Fill every remaining legacy guard position, even before any Decode
        # has happened. Otherwise a multi-chunk P scans through the RAM pool
        # and rereads weights that the original capacity-E guard retained.
        for e in eligible:
            if len(chosen) >= limit:
                break
            chosen.add(e)
        ram.anchors = chosen
        self.stats["prefill_guarded_rows"] += len(ram.anchors)

    def seed_order(self, layer, quota):
        ids = np.arange(layer * self.E, (layer + 1) * self.E, dtype=np.int64)
        rank = np.argsort(-self.scores(ids), kind="stable")
        return rank[: max(0, int(quota))].tolist()

    def snapshot(self):
        return dict(
            config=asdict(self.config),
            phase=self.phase,
            clock=self.clock,
            counters=dict(self.stats),
            nominated_mirrors=len(self.mirrors),
            metadata_array_bytes=int(
                self.heat.nbytes
                + self.prefill_heat.nbytes
                + self.decode_calls.nbytes
                + self.visits.nbytes
                + self.heat_mass.nbytes
            ),
            retained_duplicate_entries=len(self.retained_duplicates),
            retained_duplicate_entry_limit=len(self.manager.gpu.slots),
            mirror_entry_limit=(
                min(self.config.mirror_rows, self.manager.ram.capacity, len(self.manager.gpu.slots))
                if self.config.mode == "phase-mirror"
                else min(self.config.mirror_rows, self.manager.ram.capacity)
            ),
            phase_unprotected_duplicate_entries=len(self.phase_unprotected_duplicates),
            pending_mirror_nomination_entries=len(self.pending_mirror_nominations),
            extra_payload_bytes=0,
            observes_future=False,
            snapshot_bytes=(
                sum(v.nbytes for v in self.decode_snapshot.values() if isinstance(v, np.ndarray))
                if self.decode_snapshot
                else 0
            ),
            snapshot_entries=(len(self.decode_snapshot["ids"]) if self.decode_snapshot else 0),
            decode_calls_per_layer=self.decode_calls.tolist(),
        )

    def export_heat(self):
        """Explicit, untimed diagnostic export; copies cannot mutate policy state."""
        return dict(
            decode_ema=self.heat.copy(),
            prefill_frequency=self.prefill_heat.copy(),
            decode_calls=self.decode_calls.copy(),
            decode_visits=self.visits.copy(),
            decode_heat_mass=self.heat_mass.copy(),
            clock=int(self.clock),
        )


def normalize_phase_policy_config(config=None):
    """Validate public input without touching a manager, cache, or CUDA state.

    Omission and an empty mapping disable the feature. A nonempty mapping must
    name its mode explicitly so a misspelled or incomplete enablement request
    cannot silently select a different policy.
    """
    if config is None:
        return PhasePolicyConfig()
    if isinstance(config, PhasePolicyConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("phase_policy must be None, a mapping, or PhasePolicyConfig")
    values = dict(config)
    if values and "mode" not in values:
        raise ValueError("a nonempty phase_policy mapping requires an explicit mode")
    return PhasePolicyConfig(**values)


def configure_phase_policy(manager, config=None):
    """Configure a bounded Tideway pager before warmup; default is disabled.

    Active policies never attach to FreeToken or Hybrid managers. Public
    ``offload.install`` additionally validates compatibility before manager
    construction and makes this call within its rollback boundary.
    """
    config = normalize_phase_policy_config(config)
    if config.mode in ("off", "exclusive"):
        if getattr(manager, "phase_policy", None) is not None:
            manager.phase_policy = None
        return None
    from .runtime import TidewayOffload

    if not isinstance(manager, TidewayOffload):
        raise TypeError(
            "phase policy requires the bounded TidewayOffload pager; no fallback is performed"
        )
    if getattr(manager, "writeback_ahead", False):
        raise ValueError("phase policy requires writeback_ahead=False")
    policy = PhasePolicy(manager, config)
    manager.phase_policy = policy
    return policy
