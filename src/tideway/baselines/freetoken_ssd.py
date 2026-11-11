"""Independent GPU/RAM LRU stores with demand SSD reads and no writeback."""

from collections import OrderedDict
import time
import numpy as np
import torch
from tideway.offload.cache import Cache


class FreeTokenSSD:
    def __init__(self, cache, host_rows, reader, geo=None):
        self.geo = geo if geo is not None else reader.geo
        g = self.geo
        self.E = g.experts
        self.L = g.layers
        self.DOUBLE = g.double_buffer
        self.cache = cache
        self.reader = reader
        self.capacity = host_rows
        self._allocate_ram(host_rows, g)
        self.host_policy = "lru"
        self.fixed_limit = 0
        self.host_limit = host_rows
        self.crossover_guard = False  # Set by the installer for subclass-owned prefill paths.
        self.hwhere = {}
        self.host = OrderedDict()
        self.evictable = OrderedDict()
        self.free = list(range(host_rows - 1, -1, -1))
        self.gpu = Cache(cache.cache_size, g.total_experts)
        self.gset = set()
        self.copy_stream = torch.cuda.Stream()
        self.release_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.released = [False, False]
        self.ready = [None, None]
        cache.bank_sources = {}
        cache.banks = []
        # Register banks in checkpoint geometry order, including packed weights and scales.

        self.bank_names = g.bank_names
        cache.bank_views = lambda n=None: tuple(
            cache.bank_caches[k] if n is None else cache.bank_caches[k][:n] for k in self.bank_names
        )
        cache.ensure_experts = self.ensure
        cache.copy_missing = lambda: None
        self.reset_stats()

    def _allocate_ram(self, host_rows, g):
        """Allocate an exact-size pinned HostBank for the RAM expert rows."""
        from freetoken.moe.host_banks import HostBank

        self.host_bank = HostBank((host_rows, g.row_bytes), torch.uint8)
        self.host_bank.tensor.zero_()
        self.host_bank.pin()
        self.storage = self.host_bank.tensor

    def touch(self, e):
        self.host[e] = None
        self.host.move_to_end(e)
        if e >= self.fixed_limit:
            self.evictable[e] = None
            self.evictable.move_to_end(e)
        else:
            self.evictable.pop(e, None)

    def forget(self, e):
        del self.host[e]
        self.evictable.pop(e, None)

    def host_slot(self, expert, protect):
        if expert in self.hwhere:
            self.touch(expert)
            return self.hwhere[expert]
        if self.free:
            slot = self.free.pop()
        else:
            victim = next((x for x in self.evictable if x not in protect), None)
            if victim is None:
                raise RuntimeError("no unprotected host victim")
            slot = self.hwhere.pop(victim)
            self.forget(victim)
            self.stats["host_evictions"] += 1
        self.hwhere[expert] = slot
        self.touch(expert)
        return slot

    def host_views(self, slot):
        return self.geo.host_views(self.storage[slot])

    def host_allocated_bytes(self):
        return self.storage.numel()

    def reset(self):
        """Invalidate expert residency while retaining allocated storage."""
        torch.cuda.synchronize()
        for mapping in (self.hwhere, self.host, self.evictable):
            mapping.clear()
        self.free = list(range(self.capacity - 1, -1, -1))
        self.gpu = Cache(self.cache.cache_size, self.geo.total_experts)
        self.gset.clear()
        self.ready, self.released = [None, None], [False, False]
        self.reset_stats()

    def close(self):
        """Drain transfers; the supplied checkpoint reader remains caller-owned."""
        self.copy_stream.synchronize()
        torch.cuda.synchronize()

    def reset_stats(self):
        self.stats = dict(
            ssd_rows=0,
            h2d_rows=0,
            d2h_rows=0,
            gpu_hits=0,
            accesses=0,
            host_evictions=0,
            ensure_wall_s=0,
            prefill_rows=0,
            prefill_gpu_hit_rows=0,
            prefill_h2d_rows=0,
            decode_h2d_rows=0,
            prefill_load_wall_s=0,
        )
        self.reader.bytes = 0
        self.reader.seconds = 0
        if hasattr(self.reader, "thread_seconds"):
            self.reader.thread_seconds = 0

    def source(self, e, protect):
        hit = e in self.hwhere
        slot = self.host_slot(e, protect)
        views = self.host_views(slot)
        if not hit:
            self.reader.read(e, views)
            self.stats["ssd_rows"] += 1
        return views

    def ensure(self, layer, expert_ids):
        started = time.monotonic()
        raw = expert_ids.detach().cpu().numpy().astype(np.int64)
        ids = np.unique(raw) + layer * self.E
        needed = set(ids.tolist())
        missing = ids[self.gpu.where[ids] < 0]
        self.stats["accesses"] += len(ids)
        self.stats["gpu_hits"] += len(ids) - len(missing)
        self.gpu.access(ids)
        for e in missing:
            e = int(e)
            sources = self.source(e, needed)
            dst = int(self.gpu.where[e])
            for name, src in sources.items():
                self.cache.bank_caches[name][dst].copy_(src, non_blocking=True)
            self.stats["h2d_rows"] += 1
            self.stats["decode_h2d_rows"] += 1
        # Sources remain protected for this whole call. The next layer's route
        # readback synchronizes the engine stream before RAM slots can be reused.
        mapped = self.gpu.where[raw + layer * self.E]
        expert_ids.copy_(torch.as_tensor(mapped, dtype=expert_ids.dtype, device=expert_ids.device))
        self.stats["ensure_wall_s"] += time.monotonic() - started

    def begin_prefill(self):
        # Same ownership rule as native begin_prefill: no previous decode may
        # still read slots that become whole-layer double buffers.
        torch.cuda.synchronize()
        old = self.gpu.slots[: self.DOUBLE]
        self.gpu.where[old[old >= 0]] = -1
        self.gpu.slots[: self.DOUBLE] = -1
        self.gpu.usage[: self.DOUBLE] = 0
        self.ready = [None, None]
        self.released = [False, False]

    def load_layer(self, layer):
        if layer >= self.L:
            return
        bank = layer % 2
        if self.ready[bank] == layer:
            return
        started = time.monotonic()
        protect = set(range(layer * self.E, (layer + 1) * self.E))
        with torch.cuda.stream(self.copy_stream):
            if self.released[bank]:
                self.copy_stream.wait_event(self.release_events[bank])
            for local in range(self.E):
                e = layer * self.E + local
                slot = int(self.gpu.where[e])
                if slot >= self.DOUBLE:
                    for name, t in self.cache.bank_caches.items():
                        t[bank * self.E + local].copy_(t[slot], non_blocking=True)
                    self.stats["prefill_gpu_hit_rows"] += 1
                else:
                    sources = self.source(e, protect)
                    for name, src in sources.items():
                        self.cache.bank_caches[name][bank * self.E + local].copy_(
                            src, non_blocking=True
                        )
                    self.stats["h2d_rows"] += 1
                    self.stats["prefill_h2d_rows"] += 1
                self.stats["prefill_rows"] += 1
        # This also ends the leases of RAM source rows before the following
        # layer can evict them. No hidden pageable copy or full host master.
        self.copy_stream.synchronize()
        self.ready[bank] = layer
        self.stats["prefill_load_wall_s"] += time.monotonic() - started

    def prefill(self, layer, hidden, weights, ids):
        # Only subclasses with a native small-prefill branch need this guard.
        top_k = getattr(layer, "top_k", None)
        if self.crossover_guard and top_k is not None and hidden.shape[0] * top_k < self.E:
            raise RuntimeError(
                f"chunk of {hidden.shape[0]} tokens is below the native on-demand "
                f"crossover ({self.E}/{top_k} = {self.E // top_k} tokens); this arm only "
                f"reproduces the whole-layer streaming path"
            )
        number = layer.layer_id
        if number == 0:
            self.begin_prefill()
        self.load_layer(number)
        bank = number % 2
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
        # Read/upload the next layer while this layer's already-launched GEMM
        # runs. Synchronous SSD reads may still leave GPU bubbles.
        self.load_layer(number + 1)
        return result

    def snapshot(self):
        g = set(self.gpu.slots[self.gpu.slots >= 0].tolist())
        h = set(self.hwhere)
        assert len(h) <= self.capacity
        return dict(
            self.stats,
            ssd_direct_bytes=self.reader.bytes,
            disk_read_wall_s=self.reader.seconds,
            ssd_workers=getattr(self.reader, "workers", 1),
            host_rows=len(h),
            gpu_rows=len(g),
            duplicate_rows=len(g & h),
            unique_rows=len(g | h),
            host_capacity_rows=self.capacity,
            host_allocated_bytes=self.host_allocated_bytes(),
            policy="FreeToken GPU LRU and borrowed prefill buffers + independent RAM LRU + demand SSD",
            movement="per-expert copy_() per bank; demand O_DIRECT reads",
            h2d_wall_s=None,
            d2h_wall_s=0,
            note="RAM is not forced to retain copies of GPU hits. No GPU victim writeback.",
        )

    def verify_payload(self):
        """Untimed byte-for-byte audit of every resident copy (RAM and GPU)."""
        from freetoken.moe.host_banks import HostBank

        start = time.monotonic()
        scratch = HostBank((2, self.geo.row_bytes), torch.uint8)
        scratch.tensor.zero_()
        scratch.pin()
        g = self.geo

        def views(index):
            return g.host_views(scratch.tensor[index])

        expected, observed = views(0), views(1)
        checked = 0
        torch.cuda.synchronize()
        for e, slot in self.hwhere.items():
            self.reader.read(e, expected)
            for name, value in self.host_views(slot).items():
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


def install(cache, *, geo, host_rows, reader, ssd_workers=1, kernel_guard=True):
    """Install the FreeToken+SSD adapter on a pristine FreeToken engine."""
    from ._backend import require_pristine_backend
    from ._session import BaselineSession

    require_pristine_backend()
    if reader.geo is not geo and reader.geo.name != geo.name:
        raise ValueError("Checkpoint reader and expert geometry differ")
    if type(ssd_workers) is not int or ssd_workers < 1:
        raise ValueError("ssd_workers must be a positive integer")
    if getattr(reader, "workers", 1) != ssd_workers:
        raise ValueError("Reader concurrency must match the declared ssd_workers")
    if host_rows < geo.experts or cache.cache_size < geo.double_buffer:
        raise ValueError("FreeToken+SSD needs one host layer and two GPU prefill buffers")
    from freetoken.layers.moe import OffloadMoELayer

    classes = [OffloadMoELayer]
    if geo.quant == "ds_fp4":
        from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer

        classes.append(DSV4OffloadMoELayer)
    owning = [klass for klass in classes if "_prefill_routed" in vars(klass)]
    classes = owning or classes
    session = BaselineSession()
    session.remember(cache, "bank_sources", "banks", "bank_views", "ensure_experts", "copy_missing")
    try:
        manager = session.manager = FreeTokenSSD(cache, host_rows, reader, geo=geo)
        manager.crossover_guard = any(klass is not OffloadMoELayer for klass in classes)
        for klass in classes:
            original = klass._prefill_routed

            def prefill(
                layer,
                hidden,
                weights,
                ids,
                *,
                _native=original,
                _small=(geo.quant == "ds_fp4" and klass is not OffloadMoELayer),
            ):
                if _small and hidden.shape[0] * layer.top_k < geo.experts:
                    return _native(layer, hidden, weights, ids)
                return manager.prefill(layer, hidden, weights, ids)

            session.bind(klass, "_prefill_routed", prefill)
            if kernel_guard and geo.quant == "ds_fp4":
                try:
                    from freetoken.moe.fused_ds_fp4 import _GROUPED_MIN_ROUTES
                except ImportError:
                    _GROUPED_MIN_ROUTES = 768
                native_gemm = klass._expert_gemm

                def guarded(
                    layer,
                    cache_,
                    hidden,
                    weights,
                    ids,
                    *,
                    views,
                    n,
                    alphas,
                    is_prefill,
                    _native=native_gemm,
                    _floor=_GROUPED_MIN_ROUTES,
                ):
                    if is_prefill and n is None and hidden.shape[0] * layer.top_k >= _floor:
                        raise RuntimeError(
                            "Grouped FP4 prefill requires an explicit expert-bank size"
                        )
                    return _native(
                        layer,
                        cache_,
                        hidden,
                        weights,
                        ids,
                        views=views,
                        n=n,
                        alphas=alphas,
                        is_prefill=is_prefill,
                    )

                session.bind(klass, "_expert_gemm", guarded)
    except BaseException as error:
        try:
            session.uninstall()
        except BaseException as cleanup_error:
            error.add_note(f"Baseline cleanup failed: {cleanup_error!r}")
        raise
    return session
