"""Hybrid expert residency with a fixed GPU core and fixed RAM rows.

The combined RAM and GPU capacity must cover the expert pool while leaving a
GPU LRU working set. A routing prior selects the GPU core; every other expert
has a fixed RAM row. Decode uses device-side lookup and gather so it can remain
inside CUDA graphs. Prefill streams through two layer buffers and reloads any
core rows that share those buffers.
"""

import time

import numpy as np
import torch

from .runtime import Mover, RamTier

USAGE_MAX = torch.iinfo(torch.int64).max
SCRATCH_ROWS = 32


def staging_mirror_layout(volatile_core, heat_order, requested):
    """Return expert→(existing allocation, row), keeping transient rows in each.

    This pure capacity calculation is also the audit boundary: the sum of mirror
    and temporary rows stays exactly ``3 * SCRATCH_ROWS`` for every setting.
    """
    if not 0 <= int(requested) <= 3 * (SCRATCH_ROWS - 1):
        raise ValueError(f"volatile_mirror_rows must be in [0, {3 * (SCRATCH_ROWS - 1)}]")
    volatile = set(int(e) for e in volatile_core)
    chosen = [int(e) for e in heat_order if int(e) in volatile][: int(requested)]
    counts = [0, 0, 0]
    layout = {}
    for i, e in enumerate(chosen):
        b = i % 3
        counts[b] += 1
        layout[e] = (b, SCRATCH_ROWS - counts[b])
    return layout, [SCRATCH_ROWS - n for n in counts]


class HybridOffload:
    def __init__(
        self,
        cache,
        host_rows,
        reader,
        geo=None,
        prior_hist=None,
        core_rows=None,
        batch_prefill=True,
        volatile_mirror_rows=0,
    ):
        self.geo = geo if geo is not None else reader.geo
        g = self.geo
        self.E, self.L, self.DOUBLE = g.experts, g.layers, g.double_buffer
        self.total, self.S = g.total_experts, cache.cache_size
        self.cache, self.reader, self.capacity = cache, reader, host_rows
        self.bank_names = g.bank_names
        self.batch_prefill = batch_prefill
        if prior_hist is None:
            raise ValueError(
                "hybrid layout needs a routing prior (prefill histogram of an independent workload)"
            )
        hist = np.asarray(prior_hist, dtype=np.int64).reshape(-1)
        if hist.shape[0] != self.total:
            raise ValueError(f"prior has {hist.shape[0]} experts, model has {self.total}")
        C = int(core_rows) if core_rows is not None else self.total - host_rows
        if C <= 0:
            raise ValueError(
                f"RAM ({host_rows} rows) holds the whole pool ({self.total}); use the native full-pool path instead"
            )
        if C > self.S - 16:
            raise ValueError(
                f"RAM+GPU cannot hold the pool with >=16 LRU slots: core {C} > slots {self.S} - 16"
            )
        if self.total - C > host_rows:
            raise ValueError("not enough RAM rows for the non-core experts")
        order = np.argsort(-hist, kind="stable")
        self.core = np.sort(order[:C])
        self.noncore = np.sort(order[C:])
        self.is_core = np.zeros(self.total, dtype=bool)
        self.is_core[self.core] = True
        # Prefer fixed slots beyond the double buffer; any remainder is volatile.
        fixed = np.arange(self.DOUBLE, self.S)
        n_fixed = min(C, len(fixed))
        # Keep the hottest core experts outside the volatile buffers.
        core_by_heat = order[:C]
        self.core_slot = np.full(self.total, -1, dtype=np.int64)
        self.core_slot[core_by_heat[:n_fixed]] = fixed[:n_fixed]
        self.volatile_core = np.sort(core_by_heat[n_fixed:])
        self.core_slot[self.volatile_core] = np.arange(len(self.volatile_core))
        assert len(self.volatile_core) <= self.DOUBLE
        used = set(self.core_slot[self.core].tolist())
        self.lru_slots = np.array(sorted(set(range(self.S)) - used), dtype=np.int64)

        self.ram_row = np.full(self.total, -1, dtype=np.int64)
        self.ram_row[self.noncore] = np.arange(len(self.noncore))
        self.ram = RamTier(g, host_rows)
        self.mover = Mover(cache, self.ram, self.bank_names, g)
        dev = cache.device
        self.core_slots_dev = torch.as_tensor(
            self.core_slot[self.core], dtype=torch.int64, device=dev
        )
        self.lru_slots_dev = torch.as_tensor(self.lru_slots, dtype=torch.int64, device=dev)
        rr = self.ram_row.copy()
        rr[rr < 0] = 0  # Core experts never miss; keep unused table entries in bounds.
        self.ram_row_dev = torch.as_tensor(
            rr.reshape(self.L, self.E), dtype=torch.int32, device=dev
        )
        self._rows = torch.zeros(self.E, dtype=torch.int32, device=dev)
        self.miss_acc = torch.zeros((), dtype=torch.int64, device=dev)
        self.call_acc = torch.zeros((), dtype=torch.int64, device=dev)

        from freetoken.moe.host_banks import HostBank

        self.scratch = HostBank((SCRATCH_ROWS, g.row_bytes), torch.uint8)
        self.scratch.tensor.zero_()
        self.scratch.pin()
        # Each prefill buffer owns staging storage until its copy completes.

        self.vscratch = []
        for _ in range(2):
            hb = HostBank((SCRATCH_ROWS, g.row_bytes), torch.uint8)
            hb.tensor.zero_()
            hb.pin()
            self.vscratch.append(hb)
        self.vscratch_views = [
            [self.geo.host_views(hb.tensor[i]) for i in range(SCRATCH_ROWS)] for hb in self.vscratch
        ]
        # Borrow tails of the THREE EXISTING staging allocations. This changes
        # their use, never their size: each keeps at least one transient row.
        # Mirrored weights are immutable and stay valid across prefill chunks,
        # decode graph replay/recapture, and LRU resets.
        self.volatile_mirror_requested = int(volatile_mirror_rows)
        self.mirror_views = {}
        self.mirror_ready = set()
        layout, transient_rows = staging_mirror_layout(
            self.volatile_core, order, self.volatile_mirror_requested
        )
        staging = [self.scratch] + self.vscratch
        for e, (b, row) in layout.items():
            self.mirror_views[e] = self.geo.host_views(staging[b].tensor[row])
        self.scratch_rows = transient_rows[0]
        self.vscratch_views = [
            views[: transient_rows[i + 1]] for i, views in enumerate(self.vscratch_views)
        ]
        self.core_loaded = np.zeros(self.total, dtype=bool)
        self.ram_loaded = np.zeros(self.total, dtype=bool)
        # Cache static transfer plans once core slots and RAM rows are populated.
        self.static_plan = [None] * self.L
        self.static_plan_used = 0

        import os as _os

        self.profile = bool(_os.environ.get("TIDEWAY_PROFILE_PREFILL"))
        self.prof = dict(chunks=0, chunk_wall_s=0.0, gemm_s=0.0, copy_s=0.0, layers=0)
        self._prof_ev = []
        self.volatile_dirty = len(self.volatile_core) > 0
        self.copy_stream = torch.cuda.Stream()
        self.release_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.ready_events = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]  # Record layer readiness without a host-side copy-stream fence.
        self.released = [False, False]
        self.ready = [None, None]
        self.in_prefill = False
        self.crossover_guard = False
        self.trace = None
        self._pending = 0

        cache.bank_sources = {}
        cache.banks = []
        cache.bank_views = lambda n=None: tuple(
            cache.bank_caches[k] if n is None else cache.bank_caches[k][:n] for k in self.bank_names
        )
        cache.ensure_experts = self.ensure_gpu
        cache.copy_missing = self.copy_missing_gpu
        cache.reset = self.reset_lru
        self._init_tables()
        self.reset_stats()

    def _init_tables(self):
        c = self.cache
        c.slot_for_id.fill_(-1)
        c.id_of_slot.fill_(-1)
        c.usage.zero_()
        c.step.zero_()
        c.src_indices.zero_()
        c.evict_slots.zero_()
        c.num_indices.zero_()
        # Register fixed core slots now; volatile rows are registered after reload.
        fixed_core = self.core[self.core_slot[self.core] >= self.DOUBLE]
        if len(fixed_core):
            ids = torch.as_tensor(fixed_core, dtype=torch.int64, device=c.device)
            slots = torch.as_tensor(self.core_slot[fixed_core], dtype=torch.int64, device=c.device)
            c.slot_for_id.view(-1)[ids] = slots.to(torch.int32)
            c.id_of_slot[slots] = ids.to(torch.int32)
            c.usage[slots] = USAGE_MAX
        torch.cuda.synchronize()

    def reset_lru(self):
        """Clear the LRU region without invalidating the fixed GPU core."""
        c = self.cache
        ids = c.id_of_slot[self.lru_slots_dev]
        valid = ids[ids >= 0].to(torch.int64)
        if valid.numel():
            c.slot_for_id.view(-1)[valid] = -1
        c.id_of_slot[self.lru_slots_dev] = -1
        c.usage[self.lru_slots_dev] = 0
        c.usage[self.core_slots_dev] = USAGE_MAX
        c.num_indices.zero_()

    def reset_stats(self):
        self.stats = dict(
            prefill_rows=0,
            prefill_gpu_hit_rows=0,
            prefill_h2d_rows=0,
            prefill_ssd_rows=0,
            prefill_load_wall_s=0.0,
            restore_rows=0,
            restore_wall_s=0.0,
            before_decode_calls=0,
            volatile_mirror_fill_rows=0,
            volatile_mirror_prefill_rows=0,
            volatile_mirror_restore_rows=0,
            prefill_core_h2d_rows=0,
            restore_ssd_rows=0,
        )
        self.cache.lru_stats.zero_()
        self.miss_acc.zero_()
        self.call_acc.zero_()
        self.reader.bytes = 0
        self.reader.seconds = 0
        if hasattr(self.reader, "thread_seconds"):
            self.reader.thread_seconds = 0

    def ensure_gpu(self, layer_id, expert_ids):
        from flashlib.kernels.slot_cache import lru_ensure

        c = self.cache
        lru_ensure(
            expert_ids,
            c.slot_for_id.view(-1),
            c.id_of_slot,
            c.usage,
            c.step,
            expert_ids,
            c.src_indices,
            c.evict_slots,
            c.num_indices,
            stats=c.lru_stats[layer_id] if c.collect_stats else None,
            id_base=layer_id * self.E,
        )
        c.usage.index_fill_(
            0, self.core_slots_dev, USAGE_MAX
        )  # Restore pinned usage after hit accounting updates the core timestamps.
        self._pending = layer_id

    def copy_missing_gpu(self):
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        c = self.cache
        torch.index_select(self.ram_row_dev[self._pending], 0, c.src_indices, out=self._rows)
        fast_index_copy_multi_jit(
            self.mover.gpu_ptrs,
            self.mover.ram_ptrs,
            self.mover.feat,
            c.evict_slots,
            self._rows,
            c.num_indices,
        )
        self.miss_acc += c.num_indices[0]
        self.call_acc += 1

    def before_decode(self):
        """Restore volatile core rows at the prefill-to-decode boundary."""
        if not self.in_prefill:
            return
        self.in_prefill = False
        self.stats["before_decode_calls"] += 1
        torch.cuda.synchronize()
        if self.volatile_dirty:
            started = time.monotonic()
            before_ssd = self.stats["prefill_ssd_rows"]
            self._load_core_from_ssd(self.volatile_core, register=True)
            self.stats["restore_ssd_rows"] += self.stats["prefill_ssd_rows"] - before_ssd
            self.volatile_dirty = False
            self.stats["restore_rows"] += len(self.volatile_core)
            self.stats["restore_wall_s"] += time.monotonic() - started
        torch.cuda.synchronize()

    def prepare_small_prefill(self):
        """Make the native DSV4 on-demand branch safe at a chunk boundary.

        Streaming may have invalidated sole-owner core slots in its borrowed
        buffers. Restore them before the native branch uses GPU slot lookup.
        A cold tiny request has never streamed the full model: this layout's
        captured miss path requires all RAM/core backing valid, so its one-time
        fill is explicit, timed, and counted rather than reading uninitialized
        payload. Normal large first prefills already establish this residency.
        """
        self.before_decode()
        cold_ram = [int(e) for e in self.noncore if not self.ram_loaded[e]]
        if cold_ram:
            self.reader.read_many([(e, self.ram.views(int(self.ram_row[e]))) for e in cold_ram])
            self.ram_loaded[cold_ram] = True
            self.stats["prefill_ssd_rows"] += len(cold_ram)
        cold_core = self.core[~self.core_loaded[self.core]]
        if len(cold_core):
            self._load_core_from_ssd(cold_core, register=True)
        self.stats["small_prefill_cold_fill_rows"] = (
            self.stats.get("small_prefill_cold_fill_rows", 0) + len(cold_ram) + len(cold_core)
        )
        self.volatile_dirty = False

    def _fill_mirrors(self, experts):
        """Only publish ready after every byte has arrived; no transient reuse."""
        cold = [
            int(e)
            for e in experts
            if int(e) in self.mirror_views and int(e) not in self.mirror_ready
        ]
        if cold:
            self.reader.read_many([(e, self.mirror_views[e]) for e in cold])
            self.mirror_ready.update(cold)
            self.stats["prefill_ssd_rows"] += len(cold)
            self.stats["volatile_mirror_fill_rows"] += len(cold)

    def _load_core_from_ssd(self, experts, register):
        """Stage core experts from SSD into their assigned GPU slots."""
        experts = [int(e) for e in experts]
        c = self.cache
        self._fill_mirrors(experts)
        mirrored = [e for e in experts if e in self.mirror_views]
        for e in mirrored:
            for name, hv in self.mirror_views[e].items():
                c.bank_caches[name][int(self.core_slot[e])].copy_(hv, non_blocking=True)
            self.core_loaded[e] = True  # validity is ordered on the caller's CUDA stream
        self.stats["volatile_mirror_restore_rows"] += len(mirrored)
        disk_experts = [e for e in experts if e not in self.mirror_views]
        views = [self.geo.host_views(self.scratch.tensor[i]) for i in range(self.scratch_rows)]
        for at in range(0, len(disk_experts), self.scratch_rows):
            batch = disk_experts[at : at + self.scratch_rows]
            self.reader.read_many([(e, views[i]) for i, e in enumerate(batch)])
            for i, e in enumerate(batch):
                slot = int(self.core_slot[e])
                for name, hv in views[i].items():
                    c.bank_caches[name][slot].copy_(hv, non_blocking=True)
            torch.cuda.current_stream().synchronize()  # Wait before reusing the staging storage.
            for e in batch:
                self.core_loaded[e] = True
        self.stats["prefill_ssd_rows"] += len(disk_experts)
        self.stats["prefill_core_h2d_rows"] += len(experts)
        if register and experts:
            ids = torch.as_tensor(experts, dtype=torch.int64, device=c.device)
            slots = torch.as_tensor(self.core_slot[experts], dtype=torch.int64, device=c.device)
            c.slot_for_id.view(-1)[ids] = slots.to(torch.int32)
            c.id_of_slot[slots] = ids.to(torch.int32)
            c.usage[slots] = USAGE_MAX

    def begin_prefill(self):
        torch.cuda.synchronize()
        c = self.cache
        # Prefill overwrites both LRU copies and volatile cores in the double buffer.
        ids = c.id_of_slot[: self.DOUBLE]
        valid = ids[ids >= 0].to(torch.int64)
        if valid.numel():
            c.slot_for_id.view(-1)[valid] = -1
        c.id_of_slot[: self.DOUBLE] = -1
        c.usage[: self.DOUBLE] = 0
        torch.cuda.synchronize()
        self.slot_cpu = c.slot_for_id.view(-1).cpu().numpy().astype(np.int64)
        if len(self.volatile_core):
            self.core_loaded[self.volatile_core] = False
            self.volatile_dirty = True
        self.ready = [None, None]
        self.released = [False, False]
        self.in_prefill = True

    def load_layer(self, layer):
        if layer >= self.L or self.ready[layer % 2] == layer:
            return
        bank = layer % 2
        base = bank * self.E
        started = time.monotonic()
        plan = self.static_plan[layer] if self.batch_prefill else None
        if self.profile:
            c0 = torch.cuda.Event(enable_timing=True)
            c0.record(self.copy_stream)
        if plan is not None:
            # Reuse fixed transfer plans without rebuilding host-side copy descriptors.
            with torch.cuda.stream(self.copy_stream):
                if self.released[bank]:
                    self.copy_stream.wait_event(self.release_events[bank])
                t0 = time.monotonic()
                self.mover.gpu_to_gpu(plan["d2d_dst"][bank], plan["d2d_src"])
                t1 = time.monotonic()
                if not self.mover.run_prepared(plan["h2d"][bank]):
                    self.mover.ram_to_gpu(plan["h2d_dst"][bank], plan["h2d_src"])
                t2 = time.monotonic()
            self.stats["plan_d2d_issue_s"] = self.stats.get("plan_d2d_issue_s", 0.0) + (t1 - t0)
            self.stats["plan_h2d_issue_s"] = self.stats.get("plan_h2d_issue_s", 0.0) + (t2 - t1)
            if self.profile:
                c1 = torch.cuda.Event(enable_timing=True)
                c1.record(self.copy_stream)
                self._prof_ev.append(("copy", c0, c1))
            self.stats["prefill_gpu_hit_rows"] += plan["n_core"]
            self.stats["prefill_h2d_rows"] += plan["n_ram"]
            self.stats["prefill_rows"] += self.E
            self.static_plan_used += 1
            self.ready_events[bank].record(self.copy_stream)
            self.ready[bank] = layer
            self.stats["prefill_load_wall_s"] += time.monotonic() - started
            return
        es = np.arange(layer * self.E, (layer + 1) * self.E)
        slot = self.slot_cpu[es]
        core = self.is_core[es]
        hit = (slot >= self.DOUBLE) & (~core | self.core_loaded[es])
        ram_hit = ~core & ~hit & self.ram_loaded[es]
        ram_cold = ~core & ~hit & ~self.ram_loaded[es]
        core_cold = core & ~hit
        c = self.cache
        # Populate fixed RAM rows only during the first prefill.
        cold_list = [int(e) for e in es[ram_cold]]
        if cold_list:
            self.reader.read_many([(e, self.ram.views(int(self.ram_row[e]))) for e in cold_list])
            self.ram_loaded[cold_list] = True
            self.stats["prefill_ssd_rows"] += len(cold_list)
        with torch.cuda.stream(self.copy_stream):
            if self.released[bank]:
                self.copy_stream.wait_event(self.release_events[bank])
            hit_local = np.flatnonzero(hit)
            self.mover.gpu_to_gpu(
                [base + int(local_expert) for local_expert in hit_local],
                [int(slot[local_expert]) for local_expert in hit_local],
            )
            ram_local = np.flatnonzero(ram_hit | ram_cold)
            if self.batch_prefill:
                self.mover.ram_to_gpu_batch(
                    [base + int(local_expert) for local_expert in ram_local],
                    [int(self.ram_row[es[local_expert]]) for local_expert in ram_local],
                )
            else:
                self.mover.ram_to_gpu(
                    [base + int(local_expert) for local_expert in ram_local],
                    [int(self.ram_row[es[local_expert]]) for local_expert in ram_local],
                )
            # Stage cold core rows through SSD; restore volatile rows before decode.
            cold_core = [int(e) for e in es[core_cold]]
            if cold_core:
                fixed = [e for e in cold_core if self.core_slot[e] >= self.DOUBLE]
                volatile = [e for e in cold_core if self.core_slot[e] < self.DOUBLE]
                if fixed:
                    self._load_core_from_ssd(fixed, register=False)
                    self.mover.gpu_to_gpu(
                        [base + int(e - layer * self.E) for e in fixed],
                        [int(self.core_slot[e]) for e in fixed],
                    )
                if volatile:
                    self._fill_mirrors(volatile)
                    mirrored = [e for e in volatile if e in self.mirror_views]
                    for e in mirrored:
                        for name, hv in self.mirror_views[e].items():
                            c.bank_caches[name][base + e - layer * self.E].copy_(
                                hv, non_blocking=True
                            )
                    self.stats["volatile_mirror_prefill_rows"] += len(mirrored)
                    self.stats["prefill_core_h2d_rows"] += len(volatile)
                    volatile = [e for e in volatile if e not in self.mirror_views]
                    views = self.vscratch_views[bank]
                    if self.ready[bank] is not None:
                        self.ready_events[
                            bank
                        ].synchronize()  # The previous transfer must finish before the CPU overwrites staging rows.
                    for at in range(0, len(volatile), len(views)):
                        if at > 0:
                            self.copy_stream.synchronize()  # Fence when a layer exceeds one staging batch.
                        batch = volatile[at : at + len(views)]
                        self.reader.read_many([(e, views[i]) for i, e in enumerate(batch)])
                        for i, e in enumerate(batch):
                            for name, hv in views[i].items():
                                c.bank_caches[name][base + e - layer * self.E].copy_(
                                    hv, non_blocking=True
                                )
                    self.stats["prefill_ssd_rows"] += len(volatile)
        self.stats["prefill_gpu_hit_rows"] += int(hit.sum())
        self.stats["prefill_h2d_rows"] += int((ram_hit | ram_cold).sum())
        self.stats["prefill_rows"] += self.E
        # A fully loaded, nonvolatile layer can reuse its static plan.
        if (
            self.batch_prefill
            and self.static_plan[layer] is None
            and self.core_loaded[es[core]].all()
            and self.ram_loaded[es[~core]].all()
            and not any(self.core_slot[e] < self.DOUBLE for e in es[core])
        ):
            core_l = np.flatnonzero(core)
            ram_l = np.flatnonzero(~core)
            self.static_plan[layer] = dict(
                d2d_dst=[
                    [b * self.E + int(local_expert) for local_expert in core_l] for b in range(2)
                ],
                d2d_src=[int(self.core_slot[es[local_expert]]) for local_expert in core_l],
                h2d=[
                    self.mover.prepare_batch(
                        self.mover.gpu_base,
                        self.mover.ram_base,
                        [b * self.E + int(local_expert) for local_expert in ram_l],
                        [int(self.ram_row[es[local_expert]]) for local_expert in ram_l],
                    )
                    for b in range(2)
                ],
                h2d_dst=[
                    [b * self.E + int(local_expert) for local_expert in ram_l] for b in range(2)
                ],
                h2d_src=[int(self.ram_row[es[local_expert]]) for local_expert in ram_l],
                n_core=int(len(core_l)),
                n_ram=int(len(ram_l)),
            )
        # Fixed RAM rows remain valid; the compute stream waits on the readiness event.

        self.ready_events[bank].record(self.copy_stream)
        self.ready[bank] = layer
        self.stats["prefill_load_wall_s"] += (
            time.monotonic() - started
        )  # Host submission time, excluding asynchronous transfer completion.

    def prefill(self, layer, hidden, weights, ids, *, allow_small_streaming=False):
        top_k = getattr(layer, "top_k", None)
        if (
            self.crossover_guard
            and not allow_small_streaming
            and top_k is not None
            and hidden.shape[0] * top_k < self.E
        ):
            raise RuntimeError(
                f"chunk of {hidden.shape[0]} tokens is below the native on-demand crossover "
                f"({self.E}/{top_k} = {self.E // top_k} tokens); this arm only reproduces the whole-layer streaming path"
            )
        number = layer.layer_id
        if number == 0:
            if self.profile and self._prof_ev:
                self._prof_flush()
            self.begin_prefill()
            if self.profile:
                self._prof_t0 = time.monotonic()
        self.load_layer(number)
        bank = number % 2
        torch.cuda.current_stream().wait_event(self.ready_events[bank])
        views = tuple(
            self.cache.bank_caches[k][bank * self.E : (bank + 1) * self.E] for k in self.bank_names
        )
        if self.profile:
            g0 = torch.cuda.Event(enable_timing=True)
            g0.record()
        result = layer._expert_gemm(
            self.cache,
            hidden,
            weights,
            ids,
            views=views,
            n=self.E,
            alphas=self.cache.alphas_for_layer(number),
            is_prefill=True,
        )
        if self.profile:
            g1 = torch.cuda.Event(enable_timing=True)
            g1.record()
            self._prof_ev.append(("gemm", g0, g1))
        self.release_events[bank].record(torch.cuda.current_stream())
        self.released[bank] = True
        self.load_layer(number + 1)
        if self.profile and number == self.L - 1:
            self._prof_ev.append(("chunk_end", time.monotonic(), None))
        return result

    def _prof_flush(self):
        torch.cuda.synchronize()
        for kind, a, b in self._prof_ev:
            if kind == "chunk_end":
                self.prof["chunks"] += 1
                self.prof["chunk_wall_s"] += a - self._prof_t0
            else:
                self.prof[kind + "_s"] += a.elapsed_time(b) / 1000.0
                if kind == "gemm":
                    self.prof["layers"] += 1
        self._prof_ev = []

    # ------------------------------------------------------------------ reporting
    def snapshot(self):
        torch.cuda.synchronize()
        if self.profile and self._prof_ev:
            self._prof_flush()
        st = self.cache.lru_stats.sum(0).tolist()  # [ACTIVE, MISS, CALLS]
        accesses, misses = int(st[0]), int(st[1])
        s = dict(self.stats)
        s.update(
            accesses=accesses,
            gpu_hits=accesses - misses,
            decode_h2d_rows=int(self.miss_acc.item()),
            h2d_rows=int(self.miss_acc.item())
            + self.stats["prefill_h2d_rows"]
            + self.stats["prefill_core_h2d_rows"],
            d2h_rows=0,
            decode_d2h_rows=0,
            prefill_d2h_rows=0,
            ssd_rows=self.stats["prefill_ssd_rows"],
            decode_ssd_rows=0,
            ensure_calls=int(self.call_acc.item()),
            ssd_direct_bytes=self.reader.bytes,
            disk_read_wall_s=self.reader.seconds,
            disk_read_thread_s=getattr(self.reader, "thread_seconds", None),
            ssd_workers=getattr(self.reader, "workers", 1),
            gather_launches=self.mover.launches,
            batch_prefill=self.batch_prefill,
            batch_calls=self.mover.batch_calls,
            batch_entries=self.mover.batch_entries,
            static_plan_layers=int(sum(p is not None for p in self.static_plan)),
            static_plan_used=self.static_plan_used,
            prefill_profile=(dict(self.prof) if self.profile else None),
            host_rows=int(self.ram_loaded.sum()),
            host_capacity_rows=self.capacity,
            host_allocated_bytes=self.ram.allocated_bytes(),
            gpu_rows=int((self.cache.id_of_slot >= 0).sum().item()),
            staging_allocated_bytes=3 * SCRATCH_ROWS * self.geo.row_bytes,
            host_total_allocated_bytes=self.ram.allocated_bytes()
            + 3 * SCRATCH_ROWS * self.geo.row_bytes,
            volatile_mirror_rows=len(self.mirror_views),
            volatile_mirror_requested=self.volatile_mirror_requested,
            volatile_mirror_bytes=len(self.mirror_views) * self.geo.row_bytes,
            transient_staging_rows=self.scratch_rows + sum(len(v) for v in self.vscratch_views),
            core_rows=int(len(self.core)),
            volatile_core_rows=int(len(self.volatile_core)),
            lru_slots=int(len(self.lru_slots)),
            policy="Tideway hybrid (v4): static core pinned on GPU + small GPU LRU over RAM copies + fixed-row RAM; "
            "decode fully device-side (lru_ensure + RAM row table gather), CUDA-graph captured",
            movement="decode: fused SM gather from pinned RAM inside the graph; prefill: D2D + RAM H2D; "
            "unmirrored volatile cores reload from SSD after buffer reuse",
            h2d_wall_s=None,
            d2h_wall_s=None,
            ensure_wall_s=None,
            note="decode counters come from the kernel stats (device accumulators); no per-layer host timing exists in the captured path",
        )
        return s

    def verify_payload(self):
        """Compare resident core and RAM bytes against the checkpoint."""
        from freetoken.moe.host_banks import HostBank

        start = time.monotonic()
        scratch = HostBank((2, self.geo.row_bytes), torch.uint8)
        scratch.tensor.zero_()
        scratch.pin()
        expected, observed = (
            self.geo.host_views(scratch.tensor[0]),
            self.geo.host_views(scratch.tensor[1]),
        )
        checked = 0
        torch.cuda.synchronize()
        for e in self.noncore:
            if not self.ram_loaded[e]:
                continue
            self.reader.read(int(e), expected)
            for name, value in self.ram.views(int(self.ram_row[e])).items():
                assert torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), (
                    "RAM",
                    int(e),
                    name,
                )
            checked += 1
        for e in self.core:
            if not self.core_loaded[e]:
                continue
            self.reader.read(int(e), expected)
            slot = int(self.core_slot[e])
            for name, value in observed.items():
                value.copy_(self.cache.bank_caches[name][slot], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            for name, value in observed.items():
                assert torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), (
                    "GPU core",
                    int(e),
                    name,
                )
            checked += 1
        for e in sorted(self.mirror_ready):
            self.reader.read(e, expected)
            for name, value in self.mirror_views[e].items():
                assert torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), (
                    "RAM mirror",
                    e,
                    name,
                )
            checked += 1
        return dict(
            all_resident_payloads_equal=True,
            checked_copies=checked,
            wall_s=time.monotonic() - start,
        )


def recapture_decode_graphs(engine, bs=(1,)):
    """Recapture native decode graphs after installing the hybrid manager."""
    import gc
    from freetoken.engine.engine import _page_table_width
    from freetoken.engine.graph import GraphRunner

    engine.attn_backend.reset_capture()
    engine.graph_runner.destroy_cuda_graphs()
    gc.collect()
    torch.cuda.synchronize()
    free_min = engine._sync_get_memory()[0]
    engine.graph_runner = GraphRunner(
        stream=engine.stream,
        device=engine.device,
        model=engine.model,
        attn_backend=engine.attn_backend,
        cuda_graph_bs=list(bs),
        cuda_graph_max_bs=engine.config.cuda_graph_max_bs,
        free_memory=free_min,
        max_seq_len=_page_table_width(engine.max_seq_len, engine.config.page_size),
        vocab_size=engine.config.model_config.vocab_size,
        dummy_req=engine.dummy_req,
        moe_offload_cache=engine.moe_offload_cache,
    )
    return dict(graph_bs=list(engine.graph_runner.graph_bs_list), free_before=free_min)


def hook_before_decode(engine, manager):
    """Run the decode phase transition before each scheduled decode forward."""
    original = engine.forward_batch

    def forward_batch(batch, args, _orig=original, _m=manager):
        if batch.is_decode:
            _m.before_decode()
        return _orig(batch, args)

    engine.forward_batch = forward_batch
    return original
