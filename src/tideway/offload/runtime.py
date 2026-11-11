"""Tideway bounded expert runtime.

RamTier tracks pinned rows and ownership; Mover transfers expert banks;
TidewayOffload coordinates decode misses and double-buffered prefill streaming."""

from collections import Counter, OrderedDict
import contextlib
import os
import time
import numpy as np
import torch

from .cache import Cache


class RamTier:
    """Fixed-capacity pinned expert rows with LRU and replica ownership tracking."""

    def __init__(self, geo, host_rows):
        from freetoken.moe.host_banks import HostBank

        self.geo = geo
        self.capacity = host_rows
        self.banks = {b.name: HostBank((host_rows, b.nbytes), torch.uint8) for b in geo.banks}
        for hb in self.banks.values():
            hb.tensor.zero_()
            hb.pin()
        self.where = {}  # expert -> row
        self.lru = OrderedDict()  # expert -> None, LRU order
        self.free = list(range(host_rows - 1, -1, -1))
        self.duplicates = OrderedDict()  # experts that are also GPU-resident (evict these first)
        self.anchors = set()  # RAM-only residents protected during a long prefill
        self.leases = Counter()  # SSD destinations / in-flight H2D sources cannot be recycled
        self.evictions = 0
        self.duplicate_evictions = 0

    def views(self, row):
        return {
            b.name: self.banks[b.name].tensor[row].view(getattr(torch, b.dtype)).view(*b.shape)
            for b in self.geo.banks
        }

    def allocated_bytes(self):
        return sum(hb.tensor.numel() for hb in self.banks.values())

    def touch(self, e, on_gpu):
        self.lru[e] = None
        self.lru.move_to_end(e)
        if on_gpu:
            self.duplicates[e] = None
            self.duplicates.move_to_end(e)

    def forget(self, e):
        del self.lru[e]
        self.duplicates.pop(e, None)
        self.anchors.discard(e)

    def duplicate_victim(self, protect):
        return next((e for e in self.duplicates if e not in protect and e not in self.leases), None)

    def acquire(self, experts):
        for e in experts:
            if e not in self.where:
                raise RuntimeError(f"cannot lease absent RAM expert {e}")
            self.leases[e] += 1

    def release(self, experts):
        for e in experts:
            if self.leases[e] <= 0:
                raise RuntimeError(f"RAM lease underflow for expert {e}")
            self.leases[e] -= 1
            if not self.leases[e]:
                del self.leases[e]

    def can_place(self, protect):
        return (
            bool(self.free)
            or self.duplicate_victim(protect) is not None
            or any(
                e not in protect and e not in self.anchors and e not in self.leases
                for e in self.lru
            )
        )

    def place(self, e, protect, on_gpu, anchor, victim_override=None):
        """Return an expert row and its residency state, preferring replica eviction."""
        if e in self.where:
            self.touch(e, on_gpu)
            return self.where[e], True
        if self.free:
            row = self.free.pop()
        else:
            if victim_override is not None:
                victim = int(victim_override)
                if (
                    victim not in self.where
                    or victim in protect
                    or victim in self.leases
                    or victim in self.anchors
                ):
                    raise RuntimeError("unsafe phase-policy RAM victim")
                if victim in self.duplicates:
                    self.duplicate_evictions += 1
            else:
                victim = self.duplicate_victim(protect)
                if victim is not None:
                    self.duplicate_evictions += 1
                else:
                    victim = next(
                        (
                            x
                            for x in self.lru
                            if x not in protect and x not in self.anchors and x not in self.leases
                        ),
                        None,
                    )
            if victim is None:
                raise RuntimeError("no unprotected RAM slot")
            row = self.where.pop(victim)
            self.forget(victim)
            self.evictions += 1
        self.where[e] = row
        self.touch(e, on_gpu)
        if anchor:
            self.anchors.add(e)
        return row, False


class Mover:
    """Move expert rows between RAM and GPU banks with fused gather or DMA."""

    # Routing readback drains each decode layer's <= 3 moves before ring reuse.
    RING = 32

    def __init__(self, cache, ram: RamTier, bank_names, geo):
        from freetoken.kernel.pinned import device_ptr

        dev = cache.device
        self.cache = cache
        self.ram = ram
        feats = [b.nbytes for b in geo.banks]
        assert all(f % 16 == 0 for f in feats), feats
        self.gpu_ptrs = torch.tensor(
            [cache.bank_caches[n].data_ptr() for n in bank_names], dtype=torch.int64, device=dev
        )
        self.ram_ptrs = torch.tensor(
            [device_ptr(ram.banks[n].tensor) for n in bank_names], dtype=torch.int64, device=dev
        )
        self.feat = torch.tensor(feats, dtype=torch.int64, device=dev)
        # Use batched copy-engine transfers when available; host pointers have UVA aliases.

        self.gpu_base = [cache.bank_caches[n].data_ptr() for n in bank_names]
        self.ram_base = [ram.banks[n].tensor.data_ptr() for n in bank_names]
        self.feats = feats
        self._batch = None  # None: unprobed; False: unavailable; otherwise the batch-copy callable.
        self.batch_calls = 0
        self.batch_entries = 0
        self.launches = 0
        self.dma_copies = 0
        # Pinned ring buffers avoid synchronous index uploads and protect in-flight copies.

        rows = 2 * geo.experts  # The prefill double buffer can hold at most two layers of victims.
        self._host_idx = torch.empty((self.RING, 2, rows), dtype=torch.int32, pin_memory=True)
        self._host_num = torch.empty((self.RING, 1), dtype=torch.int64, pin_memory=True)
        self._dev_idx = torch.empty((self.RING, 2, rows), dtype=torch.int32, device=dev)
        self._dev_num = torch.empty((self.RING, 1), dtype=torch.int64, device=dev)
        self._ring = 0

    def move(self, dst_ptrs, src_ptrs, dst_rows, src_rows, blocks_per_bank=8, stream=None):
        n = len(dst_rows)
        if n == 0:
            return
        assert n <= self._host_idx.shape[2], (n, self._host_idx.shape)
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        k = self._ring
        self._ring = (k + 1) % self.RING
        hi, hn = self._host_idx[k], self._host_num[k]
        hi[0, :n] = torch.as_tensor(dst_rows, dtype=torch.int32)
        hi[1, :n] = torch.as_tensor(src_rows, dtype=torch.int32)
        hn[0] = n
        di, dn = self._dev_idx[k], self._dev_num[k]
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            di[:, :n].copy_(hi[:, :n], non_blocking=True)
            dn.copy_(hn, non_blocking=True)
            fast_index_copy_multi_jit(
                dst_ptrs, src_ptrs, self.feat, di[0], di[1], dn, blocks_per_bank=blocks_per_bank
            )
        self.launches += 1

    def ram_to_gpu(self, gpu_rows, ram_rows):
        self.move(self.gpu_ptrs, self.ram_ptrs, gpu_rows, ram_rows)

    def gpu_to_ram(self, ram_rows, gpu_rows, stream=None):
        self.move(self.ram_ptrs, self.gpu_ptrs, ram_rows, gpu_rows, stream=stream)

    def gpu_to_gpu(self, dst_rows, src_rows):
        self.move(self.gpu_ptrs, self.gpu_ptrs, dst_rows, src_rows, blocks_per_bank=64)

    def _batch_fn(self):
        if self._batch is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch = load_batch_memcpy()
            except Exception:  # Fall back to fused gather if the batch-copy API is unavailable.
                self._batch = False
        return self._batch

    @staticmethod
    def _runs(dst_rows, src_rows):
        """Coalesce row pairs into runs contiguous at both source and destination."""
        out = []
        for d, r in zip(dst_rows, src_rows):
            if out and out[-1][0] + out[-1][2] == d and out[-1][1] + out[-1][2] == r:
                out[-1][2] += 1
            else:
                out.append([d, r, 1])
        return out

    def _batch_copy(self, dst_base, src_base, dst_rows, src_rows):
        fn = self._batch_fn()
        if not fn or not len(dst_rows):
            return False
        dst, src, size = [], [], []
        runs = self._runs([int(x) for x in dst_rows], [int(x) for x in src_rows])
        for b, feat in enumerate(self.feats):
            for d0, s0, n in runs:
                dst.append(dst_base[b] + d0 * feat)
                src.append(src_base[b] + s0 * feat)
                size.append(n * feat)
        fn(
            torch.tensor(dst, dtype=torch.int64),
            torch.tensor(src, dtype=torch.int64),
            torch.tensor(size, dtype=torch.int64),
            torch.cuda.current_stream().cuda_stream,
        )
        self.batch_calls += 1
        self.batch_entries += len(dst)
        return True

    def prepare_batch(self, dst_base, src_base, dst_rows, src_rows):
        """Precompute CPU copy descriptors for a static layer transfer plan."""
        dst, src, size = [], [], []
        runs = self._runs([int(x) for x in dst_rows], [int(x) for x in src_rows])
        for b, feat in enumerate(self.feats):
            for d0, s0, n in runs:
                dst.append(dst_base[b] + d0 * feat)
                src.append(src_base[b] + s0 * feat)
                size.append(n * feat)
        return (
            torch.tensor(dst, dtype=torch.int64),
            torch.tensor(src, dtype=torch.int64),
            torch.tensor(size, dtype=torch.int64),
        )

    def run_prepared(self, plan):
        fn = self._batch_fn()
        if not fn or plan is None or plan[0].numel() == 0:
            return False
        fn(plan[0], plan[1], plan[2], torch.cuda.current_stream().cuda_stream)
        self.batch_calls += 1
        self.batch_entries += int(plan[0].numel())
        return True

    def ram_to_gpu_batch(self, gpu_rows, ram_rows):
        """Copy a complete layer with batched DMA, falling back to fused gather."""
        if not self._batch_copy(self.gpu_base, self.ram_base, gpu_rows, ram_rows):
            self.ram_to_gpu(gpu_rows, ram_rows)

    def gpu_to_ram_batch(self, ram_rows, gpu_rows):
        if not self._batch_copy(self.ram_base, self.gpu_base, ram_rows, gpu_rows):
            self.gpu_to_ram(ram_rows, gpu_rows)

    # Decode uses independent copy engines for small transfers in both directions.

    def ram_to_gpu_dma(self, gpu_rows, ram_rows):
        for slot, row in zip(gpu_rows, ram_rows):
            for name, hv in self.ram.views(row).items():
                self.cache.bank_caches[name][slot].copy_(hv, non_blocking=True)
        self.dma_copies += len(gpu_rows)

    def gpu_to_ram_dma(self, ram_rows, gpu_rows, stream=None):
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            for row, slot in zip(ram_rows, gpu_rows):
                for name, hv in self.ram.views(row).items():
                    hv.copy_(self.cache.bank_caches[name][slot], non_blocking=True)
        self.dma_copies += len(gpu_rows)


class TidewayOffload:
    """Coordinate bounded RAM residency, expert transfers and parallel SSD reads."""

    def __init__(
        self,
        cache,
        host_rows,
        reader,
        geo=None,
        writeback=True,
        seed_tail=False,
        prefill_guard=False,
        writeback_ahead="auto",
        dma_decode=True,
        batch_prefill=True,
        ssd_prefetch_rows=0,
    ):
        self.geo = geo if geo is not None else reader.geo
        g = self.geo
        self.E, self.L, self.DOUBLE = g.experts, g.layers, g.double_buffer
        self.cache, self.reader, self.capacity = cache, reader, host_rows
        self.writeback, self.seed_tail, self.prefill_guard = writeback, seed_tail, prefill_guard
        # Reserve writeback rows on a side stream to overlap them with the current GEMM.

        # Enable automatic lookahead only when joint capacity avoids extra SSD reloads.

        if writeback_ahead == "auto":
            writeback_ahead = host_rows + cache.cache_size >= g.total_experts
        self.writeback_ahead = bool(writeback_ahead) and writeback
        self.dma_decode = dma_decode
        self.batch_prefill = batch_prefill
        self.side_stream = torch.cuda.Stream()
        self.side_event = torch.cuda.Event()
        self.ahead_ema = np.full(
            self.L, 2.0
        )  # Per-layer miss EMA estimates the next free-slot reservation.
        # Prefill borrows the cache tail as an arena; decode reuses it as expert slots.

        st = getattr(cache, "shift", None)
        self.arena_lo = self.arena_head = self.arena_inval_hi = None
        if st is not None and getattr(st, "arena", 0) > 0:
            self.arena_lo, self.arena_head = int(st.elastic_lo), int(st.arena_head)
            self.arena_inval_hi = self.arena_lo
        self.bank_names = g.bank_names
        self.ram = RamTier(g, host_rows)
        self.gpu = Cache(cache.cache_size, g.total_experts)
        if self.arena_lo is not None:
            self.gpu.usage[self.arena_lo : self.arena_lo + self.arena_head] = (
                1 << 62
            )  # Exclude the permanently reserved arena head from LRU replacement.
        self.mover = Mover(cache, self.ram, self.bank_names, g)
        self.copy_stream = torch.cuda.Stream()
        self.release_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.ready_events = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]  # The compute stream waits on the selected prefill buffer event.
        self.released = [False, False]
        self.ready = [None, None]
        if int(ssd_prefetch_rows) < 0:
            raise ValueError("ssd_prefetch_rows must be nonnegative")
        self.ssd_prefetch_rows = min(int(ssd_prefetch_rows), self.E)
        self._prefetch = None
        self._prefetch_layer = None
        self._prefetch_experts = ()
        self._upload_leases = [(), ()]
        if self.ssd_prefetch_rows:
            from .prefetch import AsyncReadBatch

            self._prefetch = AsyncReadBatch(reader)
        self.in_prefill = False
        self.phase_policy = None  # explicitly enabled by configure_phase_policy; legacy default
        self.route_map_uploader = None  # explicitly enabled before untimed warmup
        self.empty_tail = []
        self.crossover_guard = (
            False  # Only model classes with a small-prefill branch need this guard.
        )
        # Route tracing is opt-in and configured before the first request.

        self.trace, self.trace_marks, self.prefill_hist = None, [], None
        if os.environ.get("TIDEWAY_TRACE_ROUTING"):
            self.enable_trace()
        # Take ownership of all three cache entry points.
        cache.bank_sources = {}
        cache.banks = []
        cache.bank_views = lambda n=None: tuple(
            cache.bank_caches[k] if n is None else cache.bank_caches[k][:n] for k in self.bank_names
        )
        cache.ensure_experts = self.ensure
        cache.copy_missing = lambda: None
        self.reset_stats()

    @property
    def hwhere(self):
        return self.ram.where

    @property
    def host(self):
        return self.ram.lru

    @property
    def anchors(self):
        return self.ram.anchors

    @property
    def duplicates(self):
        return self.ram.duplicates

    def host_views(self, row):
        return self.ram.views(row)

    def reset_stats(self):
        for name in ("ffn_front_graph", "attention_decode_graph"):
            graph = getattr(self, name, None)
            if graph is not None:
                graph.reset_stats()
        self._finish_prefetch()
        self._release_upload_leases()
        if self.trace is not None:
            self.trace_marks.append(len(self.trace))
        self.stats = dict(
            ssd_rows=0,
            h2d_rows=0,
            d2h_rows=0,
            gpu_hits=0,
            accesses=0,
            host_evictions=0,
            duplicate_evictions=0,
            gpu_only_dropped=0,
            ensure_wall_s=0.0,
            prefill_rows=0,
            prefill_gpu_hit_rows=0,
            prefill_h2d_rows=0,
            decode_h2d_rows=0,
            prefill_d2h_rows=0,
            decode_d2h_rows=0,
            prefill_fill_rows=0,
            prefill_load_wall_s=0.0,
            prefill_ssd_rows=0,
            decode_ssd_rows=0,
            gather_launches=0,
            ahead_d2h_rows=0,
            ahead_reserved_slots=0,
            prefetch_batches=0,
            prefetch_rows=0,
            prefetch_useful_rows=0,
            prefetch_backpressure=0,
            prefetch_wait_s=0.0,
            prefetch_max_leased_rows=0,
        )
        if getattr(self, "phase_policy", None) is not None:
            self.phase_policy.reset_stats()
        if self.route_map_uploader is not None:
            self.route_map_uploader.reset_stats()
        self._ev0, self._dup0, self._launch0 = (
            self.ram.evictions,
            self.ram.duplicate_evictions,
            self.mover.launches,
        )
        self.reader.bytes = 0
        self.reader.seconds = 0
        if hasattr(self.reader, "thread_seconds"):
            self.reader.thread_seconds = 0

    # ------------------------------------------------------------------ RAM placement + SSD
    def _place(self, e, protect):
        anchor = (
            self.in_prefill
            and self.prefill_guard
            and self.gpu.where[e] < 0
            and len(self.ram.anchors) < self.capacity - self.E
        )
        if getattr(self, "phase_policy", None) is not None:
            return self.phase_policy.place(e, protect, on_gpu=self.gpu.where[e] >= 0, anchor=anchor)
        return self.ram.place(e, protect, on_gpu=self.gpu.where[e] >= 0, anchor=anchor)

    def fetch(self, experts, protect):
        """Reserve protected RAM rows, fill SSD misses and return the expert-to-row map."""
        rows, reads = {}, []
        for e in experts:
            row, hit = self._place(e, protect)
            rows[e] = row
            if not hit:
                reads.append((e, self.ram.views(row)))
        if reads:
            self.reader.read_many(reads)
            self.stats["ssd_rows"] += len(reads)
            self.stats[("prefill" if self.in_prefill else "decode") + "_ssd_rows"] += len(reads)
        return rows

    def _release_upload_leases(self):
        """Release sources only after their H2D copy event, never just on enqueue."""
        for bank, experts in enumerate(self._upload_leases):
            if experts:
                self.ready_events[bank].synchronize()
                self.ram.release(experts)
                self._upload_leases[bank] = ()

    def _finish_prefetch(self, layer=None):
        if self._prefetch_layer is None:
            return
        started = time.monotonic()
        # An I/O error leaves leases held and aborts inference. Partially written
        # rows must never become eligible for demand hits or recycling.
        self._prefetch.finish()
        experts = self._prefetch_experts
        self.stats["prefetch_wait_s"] += time.monotonic() - started
        self.stats["ssd_rows"] += len(experts)
        self.stats["prefill_ssd_rows"] += len(experts)
        if layer == self._prefetch_layer:
            self.stats["prefetch_useful_rows"] += len(experts)
        self.ram.release(experts)
        self._prefetch_layer = None
        self._prefetch_experts = ()

    def _prefetch_layer_ahead(self, layer):
        """Admit known next-layer misses using existing, currently unleased RAM.

        We leave the prefill anchor policy intact. When no row is available,
        demand loading will wait for H2D and reuse its sources in the old path.
        No speculative expert predictions or additional payload allocations.
        """
        if self._prefetch is None or layer >= self.L:
            return
        if self._prefetch_layer is not None:
            raise RuntimeError("more than one SSD prefetch layer in flight")
        protect = set(range(layer * self.E, (layer + 1) * self.E))
        experts, items = [], []
        for e in sorted(protect):
            if self.gpu.where[e] >= self.DOUBLE or e in self.ram.where:
                continue
            if len(experts) >= self.ssd_prefetch_rows:
                break
            if not self.ram.can_place(protect):
                self.stats["prefetch_backpressure"] += 1
                break
            row, hit = self._place(e, protect)
            assert not hit
            self.ram.acquire((e,))
            experts.append(e)
            items.append((e, self.ram.views(row)))
        if not experts:
            return
        self._prefetch_layer = layer
        self._prefetch_experts = tuple(experts)
        self._prefetch.submit(items)
        self.stats["prefetch_batches"] += 1
        self.stats["prefetch_rows"] += len(experts)
        self.stats["prefetch_max_leased_rows"] = max(
            self.stats["prefetch_max_leased_rows"], len(self.ram.leases)
        )

    def save_victims(self, pairs, protect, phase):
        """Write GPU-only victims into free or duplicate RAM rows without evicting unique data."""
        dst, src = [], []
        for e, gpu_slot in pairs:
            if e in self.ram.where:
                if getattr(self, "phase_policy", None) is not None:
                    self.phase_policy.note_existing_victim(e)
                continue
            policy = getattr(self, "phase_policy", None)
            if policy is not None:
                result = policy.place(
                    e, protect, on_gpu=self.gpu.where[e] >= 0, anchor=False, writeback=True
                )
                if result is None:
                    self.stats["gpu_only_dropped"] += 1
                    continue
                row, _ = result
            else:
                if not self.writeback or (
                    not self.ram.free and self.ram.duplicate_victim(protect) is None
                ):
                    self.stats["gpu_only_dropped"] += 1
                    continue
                row, _ = self._place(e, protect)
            protect.add(e)
            dst.append(row)
            src.append(gpu_slot)
        if self.dma_decode and (
            phase == "decode"
            or getattr(getattr(self, "phase_policy", None), "small_prefill_depth", 0)
        ):
            self.mover.gpu_to_ram_dma(dst, src)
        elif self.batch_prefill:
            self.mover.gpu_to_ram_batch(dst, src)
        else:
            self.mover.gpu_to_ram(dst, src)
        self.stats["d2h_rows"] += len(dst)
        self.stats[phase + "_d2h_rows"] += len(dst)

    # ------------------------------------------------------------------ decode
    def ensure(self, layer, expert_ids):
        started = time.monotonic()
        policy = getattr(self, "phase_policy", None)
        small_prefill = policy is not None and policy.small_prefill_depth > 0
        request_phase = "prefill" if small_prefill else "decode"
        if self.in_prefill:
            self._finish_prefetch()
            self._release_upload_leases()
            self.in_prefill = False
            if not small_prefill:
                self.ram.anchors.clear()
        if small_prefill:
            self.in_prefill = True
        # Track route synchronization, LRU, SSD reads, transfer submission and slot mapping.

        raw = expert_ids.detach().cpu().numpy().astype(np.int64)
        if self.trace is not None and not small_prefill:
            self.trace.append((layer, raw.reshape(-1).astype(np.int16)))
        elif small_prefill and self.prefill_hist is not None:
            np.add.at(self.prefill_hist, raw.reshape(-1) + layer * self.E, 1)
        if self.writeback_ahead:
            self.side_event.synchronize()  # Finish writeback before either tier reuses the affected rows.
        t_sync = time.monotonic()
        ids = np.unique(raw) + layer * self.E
        needed = set(ids.tolist())
        policy = getattr(self, "phase_policy", None)
        if policy is not None and not small_prefill:
            policy.restore_snapshot(needed)
        missing = ids[self.gpu.where[ids] < 0]
        self.stats["accesses"] += len(ids)
        self.stats["gpu_hits"] += len(ids) - len(missing)
        old = self.gpu.slots.copy()
        policy = getattr(self, "phase_policy", None)
        if policy is not None:
            if small_prefill:
                policy.observe_prefill(layer, np.bincount(raw.reshape(-1), minlength=self.E))
                self.stats["small_prefill_route_rows"] = self.stats.get(
                    "small_prefill_route_rows", 0
                ) + len(ids)
            else:
                policy.observe_decode(layer, ids, missing)
            policy.gpu_access(ids)
        else:
            self.gpu.access(ids)
        destinations = self.gpu.where[missing]
        pairs = [(int(old[s]), int(s)) for s in destinations if old[s] >= 0]
        for e, _ in pairs:
            self.ram.duplicates.pop(e, None)
        for e in needed:
            if e in self.ram.where:
                self.ram.duplicates[e] = None
        t_lru = time.monotonic()
        self.save_victims(pairs, needed, request_phase)
        t_wb = time.monotonic()
        miss = [int(e) for e in missing]
        rows = self.fetch(miss, needed)
        t_fetch = time.monotonic()
        if self.dma_decode:
            self.mover.ram_to_gpu_dma(
                [int(self.gpu.where[e]) for e in miss], [rows[e] for e in miss]
            )
        else:
            self.mover.ram_to_gpu([int(self.gpu.where[e]) for e in miss], [rows[e] for e in miss])
        self.stats["h2d_rows"] += len(miss)
        self.stats[request_phase + "_h2d_rows"] += len(miss)
        mapped = self.gpu.where[raw + layer * self.E]
        if self.route_map_uploader is None:
            expert_ids.copy_(
                torch.as_tensor(mapped, dtype=expert_ids.dtype, device=expert_ids.device)
            )
        else:
            self.route_map_uploader.copy_to(expert_ids, mapped)
        if self.writeback_ahead:
            # Reserve only the estimated deficit; do not accumulate unused slots.
            self.ahead_ema[layer] = 0.5 * self.ahead_ema[layer] + 0.5 * len(miss)
            want = int(np.ceil(self.ahead_ema[(layer + 1) % self.L]))
            free_now = int((self.gpu.slots < 0).sum())
            self._reserve_ahead(want - free_now, needed)
        end = time.monotonic()
        st = self.stats
        st["ensure_sync_s"] = st.get("ensure_sync_s", 0.0) + (t_sync - started)
        st["ensure_lru_s"] = st.get("ensure_lru_s", 0.0) + (t_lru - t_sync)
        st["ensure_writeback_s"] = st.get("ensure_writeback_s", 0.0) + (t_wb - t_lru)
        st["ensure_fetch_s"] = st.get("ensure_fetch_s", 0.0) + (t_fetch - t_wb)
        st["ensure_move_s"] = st.get("ensure_move_s", 0.0) + (end - t_fetch)
        st["ensure_calls"] = st.get("ensure_calls", 0) + 1
        st["ensure_wall_s"] += end - started

    def _reserve_ahead(self, k, protect):
        """Free unprotected GPU victims and write unique copies back on the side stream.

        Mark released slots unused so the next cache access can fill them
        without synchronous writeback.
        """
        gpu = self.gpu
        if k <= 0:
            return
        occupied = np.flatnonzero(gpu.slots >= 0)
        if len(occupied) == 0:
            return
        order = occupied[np.argsort(gpu.usage[occupied], kind="stable")]
        dst, src, taken = [], [], 0
        for slot in order:
            if taken >= k:
                break
            e = int(gpu.slots[slot])
            if e in protect:
                continue
            taken += 1
            gpu.slots[slot] = -1
            gpu.usage[slot] = 0
            gpu.where[e] = -1
            if e in self.ram.where:
                self.ram.duplicates.pop(e, None)
                continue
            if not self.ram.free and self.ram.duplicate_victim(protect) is None:
                self.stats["gpu_only_dropped"] += (
                    1  # Never evict a unique RAM expert just to save a GPU victim.
                )
                continue
            row, _ = self._place(e, protect)
            protect.add(e)
            dst.append(row)
            src.append(int(slot))
        if dst:
            if self.dma_decode:
                self.mover.gpu_to_ram_dma(dst, src, stream=self.side_stream)
            else:
                self.mover.gpu_to_ram(dst, src, stream=self.side_stream)
            self.side_event.record(self.side_stream)
            self.stats["d2h_rows"] += len(dst)
            self.stats["decode_d2h_rows"] += len(dst)
            self.stats["ahead_d2h_rows"] += len(dst)
        self.stats["ahead_reserved_slots"] += taken

    def reset_cold(self):
        """Clear GPU and RAM residency metadata after warmup."""
        self._finish_prefetch()
        self._release_upload_leases()
        torch.cuda.synchronize()
        g = self.gpu
        g.slots[:] = -1
        g.usage[:] = 0
        g.where[:] = -1
        g.step = 0
        if self.arena_lo is not None:
            g.usage[self.arena_lo : self.arena_lo + self.arena_head] = 1 << 62
        self.ram.where.clear()
        self.ram.lru.clear()
        self.ram.duplicates.clear()
        self.ram.anchors.clear()
        self.ram.free = list(range(self.ram.capacity - 1, -1, -1))
        self.in_prefill = False
        self.ready = [None, None]
        self.released = [False, False]
        self.ahead_ema[:] = 2.0
        if getattr(self, "phase_policy", None) is not None:
            self.phase_policy.reset_cold()
        self.reset_stats()

    def before_prefill_arena(self, n_tok):
        """Save unique experts and release rows before the activation arena overwrites them."""
        if getattr(self, "phase_policy", None) is not None:
            self.phase_policy.capture_decode_snapshot()
        if self.arena_lo is None:
            return
        from freetoken.moe import shift as _shift

        st = self.cache.shift
        rows = _shift._arena_rows_estimate(st, int(n_tok)) if st.arena_sized else st.arena_main_rows
        lo, hi = (
            self.arena_lo,
            min(self.arena_lo + self.arena_head + int(rows), self.gpu.slots.shape[0]),
        )
        torch.cuda.synchronize()
        self.in_prefill = False
        self.ram.anchors.clear()
        pairs = [(int(e), int(i)) for i, e in zip(range(lo, hi), self.gpu.slots[lo:hi]) if e >= 0]
        for e, _ in pairs:
            self.gpu.where[e] = -1
            self.ram.duplicates.pop(e, None)
        self.gpu.slots[lo:hi] = -1
        self.gpu.usage[lo + self.arena_head : hi] = 0
        self.save_victims(pairs, set(), "prefill")
        self.stats["arena_evicted_rows"] = self.stats.get("arena_evicted_rows", 0) + len(pairs)
        self.stats["arena_prefills"] = self.stats.get("arena_prefills", 0) + 1
        self.arena_inval_hi = hi
        torch.cuda.synchronize()  # Finish reading expert rows before the activation arena overwrites them.

    # ------------------------------------------------------------------ prefill
    def begin_prefill(self):
        if getattr(self, "phase_policy", None) is not None:
            self.phase_policy.capture_decode_snapshot()
        self._finish_prefetch()
        self._release_upload_leases()
        torch.cuda.synchronize()
        self.in_prefill = False
        self.ram.anchors.clear()
        pairs = [(int(e), i) for i, e in enumerate(self.gpu.slots[: self.DOUBLE]) if e >= 0]
        for e, _ in pairs:  # borrowed slots stop being persistent copies
            self.gpu.where[e] = -1
            self.ram.duplicates.pop(e, None)
        self.gpu.slots[: self.DOUBLE] = -1
        self.gpu.usage[: self.DOUBLE] = 0
        self.save_victims(pairs, set(), "prefill")
        torch.cuda.current_stream().synchronize()  # the copy stream must not overwrite a slot still being read back
        self.ready = [None, None]
        self.released = [False, False]
        empties = np.flatnonzero(self.gpu.slots[self.DOUBLE :] < 0) + self.DOUBLE
        if (
            self.arena_lo is not None
        ):  # Arena rows borrowed by this prefill cannot seed expert residency.
            empties = empties[(empties < self.arena_lo) | (empties >= self.arena_inval_hi)]
        self.empty_tail = empties.tolist()[::-1]
        self.in_prefill = True
        if self.prefill_guard:
            self.ram.anchors = set(
                [e for e in self.ram.lru if self.gpu.where[e] < 0][: self.capacity - self.E]
            )
        if getattr(self, "phase_policy", None) is not None:
            self.phase_policy.begin_prefill()

    def load_layer(self, layer):
        if layer >= self.L or self.ready[layer % 2] == layer:
            return
        bank = layer % 2
        started = time.monotonic()
        protect = set(range(layer * self.E, (layer + 1) * self.E))
        where = self.gpu.where[layer * self.E : (layer + 1) * self.E]
        hit_local = [
            local_expert
            for local_expert in range(self.E)
            if int(where[local_expert]) >= self.DOUBLE
        ]
        ram_local = [
            local_expert for local_expert in range(self.E) if int(where[local_expert]) < self.DOUBLE
        ]
        self._finish_prefetch(layer)
        self._release_upload_leases()
        if self._prefetch is None and self.ready[1 - bank] is not None:
            self.ready_events[
                1 - bank
            ].synchronize()  # Do not evict RAM rows still consumed by the other prefill buffer.
        rows = self.fetch(
            [layer * self.E + local_expert for local_expert in ram_local], protect
        )  # SSD misses read concurrently
        if self._prefetch is not None:
            self._upload_leases[bank] = tuple(rows)
            self.ram.acquire(self._upload_leases[bank])
        base = bank * self.E
        fill_dst, fill_src = [], []
        with torch.cuda.stream(self.copy_stream):
            if self.released[bank]:
                self.copy_stream.wait_event(self.release_events[bank])
            self.mover.gpu_to_gpu(
                [base + local_expert for local_expert in hit_local],
                [int(where[local_expert]) for local_expert in hit_local],
            )
            if self.batch_prefill:
                self.mover.ram_to_gpu_batch(
                    [base + local_expert for local_expert in ram_local],
                    [rows[layer * self.E + local_expert] for local_expert in ram_local],
                )
            else:
                self.mover.ram_to_gpu(
                    [base + local_expert for local_expert in ram_local],
                    [rows[layer * self.E + local_expert] for local_expert in ram_local],
                )
            policy = getattr(self, "phase_policy", None)
            phase_seed = policy is not None and policy.config.seed_empty_tail
            if (
                self.seed_tail and self.empty_tail and not phase_seed
            ):  # cold start: retain streamed experts in EMPTY slots only
                for local_expert in range(self.E):
                    e = layer * self.E + local_expert
                    if self.gpu.where[e] >= 0 or not self.empty_tail:
                        continue
                    slot = self.empty_tail.pop()
                    fill_dst.append(slot)
                    fill_src.append(base + local_expert)
                    self.gpu.slots[slot] = e
                    self.gpu.where[e] = slot
                    self.gpu.usage[slot] = 0
                    if e in self.ram.where:
                        self.ram.duplicates[e] = None
                    self.ram.anchors.discard(e)
                self.mover.gpu_to_gpu(fill_dst, fill_src)
        self.stats["prefill_gpu_hit_rows"] += len(hit_local)
        self.stats["h2d_rows"] += len(ram_local)
        self.stats["prefill_h2d_rows"] += len(ram_local)
        self.stats["prefill_rows"] += self.E
        self.stats["prefill_fill_rows"] += len(fill_dst)
        self.ready_events[bank].record(
            self.copy_stream
        )  # Hold RAM row leases until the next load_layer synchronization.
        self.ready[bank] = layer
        self._prefetch_layer_ahead(layer + 1)
        self.stats["prefill_load_wall_s"] += time.monotonic() - started

    def prefill(self, layer, hidden, weights, ids):
        top_k = getattr(layer, "top_k", None)
        if self.crossover_guard and top_k is not None and hidden.shape[0] * top_k < self.E:
            raise RuntimeError(
                f"chunk of {hidden.shape[0]} tokens is below the native on-demand crossover "
                f"({self.E}/{top_k} = {self.E // top_k} tokens); this arm only reproduces the whole-layer streaming path"
            )
        number = layer.layer_id
        if number == 0:
            self.begin_prefill()
        if self.prefill_hist is not None:
            np.add.at(
                self.prefill_hist,
                ids.detach().reshape(-1).cpu().numpy().astype(np.int64) + number * self.E,
                1,
            )
        self.load_layer(number)
        bank = number % 2
        torch.cuda.current_stream().wait_event(self.ready_events[bank])
        policy = getattr(self, "phase_policy", None)
        if policy is not None:
            sample = policy.config.prefill_sample_tokens
            if sample:
                # Explicit optional observation: true routes from an already
                # executed router. Its GPU->CPU synchronization stays timed.
                route = ids[-sample:].detach().reshape(-1).to(dtype=torch.long)
                counts = torch.bincount(route, minlength=self.E).cpu().numpy()
                policy.observe_prefill(number, counts)
            if policy.config.seed_empty_tail and self.empty_tail:
                quota = (len(self.empty_tail) + self.L - number - 1) // (self.L - number)
                dst, src = [], []
                for local in policy.seed_order(number, self.E):
                    e = number * self.E + local
                    if self.gpu.where[e] >= 0:
                        continue
                    if len(dst) >= quota or not self.empty_tail:
                        break
                    slot = self.empty_tail.pop()
                    self.gpu.slots[slot] = e
                    self.gpu.where[e] = slot
                    self.gpu.usage[slot] = self.gpu.step
                    if e in self.ram.where:
                        self.ram.duplicates[e] = None
                    self.ram.anchors.discard(e)
                    dst.append(slot)
                    src.append(bank * self.E + local)
                self.mover.gpu_to_gpu(dst, src)
                self.stats["prefill_fill_rows"] += len(dst)
        views = tuple(
            self.cache.bank_caches[k][bank * self.E : (bank + 1) * self.E] for k in self.bank_names
        )
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
        self.release_events[bank].record(torch.cuda.current_stream())
        self.released[bank] = True
        self.load_layer(number + 1)  # next layer's reads/uploads overlap this layer's GEMM
        return result

    # ------------------------------------------------------------------ reporting / audit
    def snapshot(self):
        self._finish_prefetch()
        g = set(self.gpu.slots[self.gpu.slots >= 0].tolist())
        h = set(self.ram.where)
        assert len(h) <= self.capacity
        assert set(self.ram.duplicates) == h & g
        s = dict(self.stats)
        s.update(
            host_evictions=self.ram.evictions - self._ev0,
            duplicate_evictions=self.ram.duplicate_evictions - self._dup0,
            gather_launches=self.mover.launches - self._launch0,
            ssd_direct_bytes=self.reader.bytes,
            disk_read_wall_s=self.reader.seconds,
            disk_read_thread_s=getattr(self.reader, "thread_seconds", None),
            ssd_workers=getattr(self.reader, "workers", 1),
            ssd_batches=getattr(self.reader, "batches", None),
            ssd_batch_rows=getattr(self.reader, "batch_rows", None),
            host_rows=len(h),
            gpu_rows=len(g),
            duplicate_rows=len(g & h),
            unique_rows=len(g | h),
            host_capacity_rows=self.capacity,
            host_allocated_bytes=self.ram.allocated_bytes(),
            policy="Tideway: exclusive write-back"
            + (" (v3: one layer ahead on a side stream)" if self.writeback_ahead else "")
            + ", duplicate-first RAM eviction, prefill RAM guard, cold-start tail seeding",
            movement="fused SM gather per direction per layer over per-bank pinned RAM (v2: pinned-ring async index upload); concurrent O_DIRECT reads",
            writeback_ahead=self.writeback_ahead,
            dma_decode=self.dma_decode,
            dma_copies=self.mover.dma_copies,
            batch_prefill=self.batch_prefill,
            batch_calls=self.mover.batch_calls,
            batch_entries=self.mover.batch_entries,
            ssd_prefetch_rows=self.ssd_prefetch_rows,
            prefetch_payload_allocation_bytes=0,
            arena_rows=(self.gpu.slots.shape[0] - self.arena_lo)
            if self.arena_lo is not None
            else 0,
            arena=(
                getattr(__import__("freetoken.moe.shift", fromlist=["arena_stats"]), "arena_stats")(
                    self.cache.shift
                )
                if self.arena_lo is not None
                else None
            ),
            selective_writeback=self.writeback,
            seed_empty_gpu_tail=self.seed_tail,
            prefill_RAM_guard=self.prefill_guard,
            h2d_wall_s=None,
            d2h_wall_s=None,
            note="Direction byte counts are exact; streams overlap, so wall times are not additive.",
        )
        if getattr(self, "phase_policy", None) is not None:
            s["phase_policy"] = self.phase_policy.snapshot()
        if self.route_map_uploader is not None:
            s["route_map_upload"] = self.route_map_uploader.snapshot()
        for name in ("attention_decode_graph", "ffn_front_graph"):
            graph = getattr(self, name, None)
            if graph is not None:
                s[name] = graph.snapshot()
        return s

    def close(self):
        try:
            self._finish_prefetch()
            self._release_upload_leases()
        finally:
            try:
                if self._prefetch is not None:
                    self._prefetch.close()
            finally:
                combined = getattr(self, "segmented_decode", None)
                if combined is not None:
                    combined.close()
                else:
                    try:
                        front = getattr(self, "ffn_front_graph", None)
                        if front is not None:
                            front.close()
                    finally:
                        try:
                            attention = getattr(self, "attention_decode_graph", None)
                            if attention is not None:
                                attention.close()
                        finally:
                            if self.route_map_uploader is not None:
                                self.route_map_uploader.close()

    def enable_trace(self):
        """Enable decode route and prefill expert-count recording before the first request."""
        self.trace = []
        self.trace_marks = []
        self.prefill_hist = np.zeros(self.geo.total_experts, dtype=np.int64)

    def dump_trace(self, path):
        """Write route IDs, request boundaries and prefill expert counts to an NPZ file."""
        if self.trace is None:
            return None
        layers = np.array([layer_index for layer_index, _ in self.trace], dtype=np.int16)
        ids = (
            np.stack([r for _, r in self.trace]) if self.trace else np.zeros((0, 0), dtype=np.int16)
        )
        np.savez_compressed(
            path,
            decode_layers=layers,
            decode_ids=ids,
            case_marks=np.array(self.trace_marks, dtype=np.int64),
            prefill_hist=self.prefill_hist,
            experts_per_layer=self.E,
            layers_total=self.L,
            gpu_slots=self.gpu.slots.shape[0],
            host_rows=self.capacity,
        )
        return dict(
            path=str(path),
            decode_records=int(len(layers)),
            prefill_accesses=int(self.prefill_hist.sum()),
        )

    def verify_payload(self):
        """Compare every resident RAM and GPU copy against the checkpoint."""
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
        for e, row in self.ram.where.items():
            self.reader.read(e, expected)
            for name, value in self.ram.views(row).items():
                assert torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), (
                    "RAM",
                    e,
                    name,
                )
            checked += 1
        for slot, e in enumerate(self.gpu.slots):
            if e < 0:
                continue
            self.reader.read(int(e), expected)
            for name, value in observed.items():
                value.copy_(self.cache.bank_caches[name][slot], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            for name, value in observed.items():
                assert torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)), (
                    "GPU",
                    int(e),
                    name,
                )
            checked += 1
        return dict(
            all_resident_payloads_equal=True,
            checked_copies=checked,
            wall_s=time.monotonic() - start,
        )


def hook_forward_batch(engine, before_prefill=None, before_decode=None):
    """Run prefill callbacks before the engine prepares its activation arena."""
    original = engine.forward_batch

    def forward_batch(batch, args, _orig=original):
        if before_prefill is not None and batch.is_prefill:
            before_prefill(int(batch.input_ids.numel()))
        if before_decode is not None and batch.is_decode:
            before_decode()
        return _orig(batch, args)

    engine.forward_batch = forward_batch
    return original
