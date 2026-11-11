"""Online physical KV/expert sharing for the native Tideway-Shift path.

Install once, while idle and with an empty prefix cache, after arena calibration.
Full and SWA tensors retain maximum shapes and stable virtual addresses; only
physical backing grows. Expert slots in the appended tail use the freed bytes.
Growth is driven by actual allocator requests, never a future output length.
"""

from __future__ import annotations

import gc
import math
import time

from .elastic_layout import (
    StripeLayout,
    bank_bytes,
    ceil_div,
    fit_slots,
    high_water_from_free,
    plan_swa_compaction,
)


class ElasticKV:
    def __init__(
        self,
        llm,
        *,
        initial_tokens=8192,
        growth_tokens=2048,
        max_slots=None,
        evict_unmatched_prefixes=True,
        swa_initial_tokens=None,
        compact_swa=False,
        chunk_mib=2,
        pool_physical_handles=False,
    ):
        import torch
        from freetoken.kvcache.mha_pool import MHAKVCache
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
        from .elastic_vmm import VMM

        if not isinstance(chunk_mib, int) or chunk_mib < 1 or chunk_mib & (chunk_mib - 1):
            raise ValueError("VMM chunk_mib must be a positive power of two")
        if initial_tokens < 1 or growth_tokens < 1:
            raise ValueError("elastic KV initial/growth tokens must be positive")
        if swa_initial_tokens is not None and swa_initial_tokens < 1:
            raise ValueError("elastic SWA initial tokens must be positive")
        self.llm, self.engine, self.cm = llm, llm.engine, llm.cache_manager
        eng, cm = self.engine, self.cm
        cache, pool = eng.moe_offload_cache, eng.kv_cache
        if getattr(eng, "elastic_kv", None) is not None:
            raise ValueError("elastic KV already installed")
        if cache is None or cache.shift is None or cache.decode_target != "gpu":
            raise ValueError("elastic KV requires the native GPU Tideway-Shift cache")
        # Bounded/hybrid managers replace this method on the instance. Their GPU
        # sole-owner experts need a release hook before physical pages can move.
        if "ensure_experts" in vars(cache):
            raise ValueError("bounded/hybrid elastic KV needs an expert ownership hook")
        if getattr(eng.config.tp_info, "size", 1) != 1:
            raise ValueError("elastic KV currently supports one GPU")
        if type(pool) not in (MHAKVCache, HybridSWAKVCache):
            raise ValueError(f"unsupported elastic KV pool {type(pool).__name__}")
        if llm.prefill_manager.runnable or llm.decode_manager.runnable:
            raise ValueError("install elastic KV while scheduler is idle")
        cm.check_integrity()
        if len(cm.free_slots) != cm.num_pages:
            raise ValueError("clear prefix caches before installing elastic KV")
        if cm.swa_paged and pool.swa_available_size() != pool.swa_num_tokens - 1:
            raise ValueError("clear SWA prefix cache before installing elastic KV")
        self.cache, self.pool = cache, pool
        self.device, self.page_size = eng.device, cm.page_size
        self.initial_tokens, self.growth_tokens = int(initial_tokens), int(growth_tokens)
        self.swa_initial_tokens = (
            int(swa_initial_tokens) if swa_initial_tokens is not None else None
        )
        if cm.swa_paged and self.swa_initial_tokens is None:
            from freetoken.scheduler.cache import _SWA_RETAIN_GAP, _SWA_EVICTION_INTERVAL

            # Initial physical backing is a single request's current window plus
            # the engine's retention/eviction margin. Concurrent requests and
            # prefill chunks grow it online through the actual allocator demand.
            self.swa_initial_tokens = (
                int(cm.sliding_window_size)
                + _SWA_RETAIN_GAP
                + _SWA_EVICTION_INTERVAL
                + 2 * self.page_size
                + 1
            )
        self.compact_swa = bool(compact_swa and cm.swa_paged)
        self.pending_compaction = False
        self._pending_swa_trim = []
        self.swa_compactions = self.swa_compacted_tokens = self.swa_compaction_d2d_bytes = 0
        self.swa_compaction_metadata_peak_bytes = self.swa_phase_trimmed_tokens = 0
        self.swa_compaction_seconds = 0.0
        self.swa_compaction_events = []
        self.evict_unmatched_prefixes = bool(evict_unmatched_prefixes)
        self._full_prefix_pins = {}
        self.full_prefix_pin_events = []
        self.full_prefix_capacity_fallbacks = 0
        self.base_slots = cache.cache_size
        self.row_bytes = [b.stride(0) * b.element_size() for _, b in cache.banks]
        self.bank_regions = {}
        self.kv_regions = {}
        self.layouts, self.shapes, self.dtypes = {}, {}, {}
        self.mapped = {}
        self.ready = {"full": 0, "swa": 0}
        self.events = []
        self.grows = self.shrinks = self.invalidated_experts = self.evicted_prefix_tokens = 0
        self.remap_seconds = 0.0
        self._failed = False
        old_hw = dict(cache.shift.arena_hw)

        def describe(name, tensor, dummy_tail):
            shape = tuple(tensor.shape)
            self.shapes[name], self.dtypes[name] = shape, tensor.dtype
            self.layouts[name] = StripeLayout(
                shape[0] * shape[1],
                shape[2],
                math.prod(shape[3:]) * tensor.element_size(),
                dummy_tail,
            )

        if isinstance(pool, MHAKVCache):
            describe("full", pool._kv_buffer, 1)
        else:
            describe("full", pool.full_kv_pool.buffer, 1)
            describe("swa", pool.swa_kv_pool.buffer, 0)
        for name, layout in self.layouts.items():
            floor = (
                ceil_div(initial_tokens, self.page_size)
                if name == "full"
                else self.swa_initial_tokens
            )
            self.mapped[name] = min(layout.usable, max(1, floor))

        # Validate driver support before tearing down any live allocation.
        torch.cuda.synchronize(self.device)
        self.vmm = VMM(self.device.index or 0, chunk_bytes=chunk_mib << 20)
        self.chunk = self.vmm.chunk_bytes
        full_kv_bytes = sum(
            ceil_div(layout.nbytes, self.chunk) * self.chunk for layout in self.layouts.values()
        )
        self.physical_budget = (
            bank_bytes(self.base_slots, self.row_bytes, self.chunk) + full_kv_bytes
        )
        if pool_physical_handles:
            self.vmm.enable_pool(self.physical_budget)
        initial_kv_bytes = sum(
            len(layout.chunks(self.mapped[n], self.chunk)) * self.chunk
            for n, layout in self.layouts.items()
        )
        self.max_slots = fit_slots(
            self.physical_budget - initial_kv_bytes,
            self.row_bytes,
            self.chunk,
            cache.num_layers * cache.num_experts,
        )
        if max_slots is not None:
            if int(max_slots) < self.base_slots:
                raise ValueError(
                    "elastic max_slots cannot be smaller than the calibrated base cache"
                )
            self.max_slots = min(self.max_slots, int(max_slots))
        self.active_slots = self.max_slots
        self.min_active_slots = self.max_slots
        self.max_mapped = dict(self.mapped)

        prior_graph_bs = list(eng.graph_runner.graph_bs_list)
        eng.attn_backend.reset_capture()
        eng.graph_runner.destroy_cuda_graphs()
        # A calibrated arena MemPool still owns cached segments pointing into
        # the old expert banks. Release it before replacing those banks, exactly
        # as the Shift geometry rebuild path does.
        from freetoken.moe import fused as _fused

        _fused.set_shift_workspace(None)
        from .arena_lifetime import guard_arena_bank_rebuild

        self.arena_rebuild_lifetime = guard_arena_bank_rebuild(llm)
        cache.shift.pool = None
        # Installation is the only cold transition. Serving never rebinds KV,
        # banks, page tables, masks, or graph addresses.
        if isinstance(pool, MHAKVCache):
            pool._kv_buffer = pool._k_buffer = pool._v_buffer = None
        else:
            pool.full_kv_pool = pool.swa_kv_pool = None
            pool._storages = {}
        gc.collect()
        torch.cuda.empty_cache()
        for name in self.layouts:
            tensor, region = self.vmm.tensor(self.shapes[name], self.dtypes[name])
            if region.chunk != self.chunk:
                raise RuntimeError("device VMM granularity differs from planned chunk")
            self.kv_regions[name] = region
            region.set_chunks(self.layouts[name].chunks(self.mapped[name], self.chunk))
            if name == "full" and isinstance(pool, MHAKVCache):
                pool._kv_buffer, pool._k_buffer, pool._v_buffer = tensor, tensor[0], tensor[1]
            else:
                from freetoken.kvcache.hybrid_swa_pool import _KVGroupStorage

                shape = self.shapes[name]
                storage = _KVGroupStorage(
                    tensor, tensor[0], tensor[1], (shape[2] * shape[3], shape[4], shape[5])
                )
                pool._storages[name] = storage
                setattr(pool, f"{name}_kv_pool", storage)
        cache.elastic_base_slots = self.base_slots
        cache.elastic_kv = self
        cache._elastic_bank_factory = self._bank_factory
        try:
            cache.rebuild(self.max_slots)
        finally:
            del cache._elastic_bank_factory
        cache.shift.arena_hw.update(old_hw)
        # The calibration prefill compiled seed metadata for the OLD slot count.
        # VMM's appended expert tail changes the constexpr C. Warm only metadata
        # on isolated scratch now, before the first measured request/graph work.
        seed = getattr(cache.shift, "empty_seed", None)
        self.seed_metadata_warmup = seed.warmup_metadata(cache) if seed is not None else None
        cm.elastic_kv = eng.elastic_kv = self
        if cm.swa_paged:
            pool.elastic_kv = self
        object.__setattr__(eng.config, "moe_cache_size", self.max_slots)
        # A completely free pool need not have a sorted free list (especially
        # SWA after eviction). Derive the ready prefix from actual indices.
        for name in self.layouts:
            self._refresh(name)
        if self.compact_swa:
            from .elastic_swa import warmup_swa_copy

            warmup_swa_copy(pool.swa_kv_pool.buffer)
        self._addresses = self._current_addresses()
        from freetoken.engine.graph import GraphRunner

        eng.graph_runner = GraphRunner(
            stream=eng.stream,
            device=eng.device,
            model=eng.model,
            attn_backend=eng.attn_backend,
            cuda_graph_bs=prior_graph_bs,
            cuda_graph_max_bs=eng.config.cuda_graph_max_bs,
            free_memory=torch.cuda.mem_get_info(eng.device)[0],
            max_seq_len=eng.page_table.shape[1],
            vocab_size=eng.config.model_config.vocab_size,
            dummy_req=eng.dummy_req,
            moe_offload_cache=cache,
        )
        self._assert_invariants()

    def _bank_factory(self, name, shape, dtype):
        tensor, region = self.vmm.tensor(shape, dtype)
        self.bank_regions[name] = region
        row = math.prod(shape[1:]) * tensor.element_size()
        region.set_chunks(set(range(ceil_div(self.active_slots * row, self.chunk))))
        return tensor

    def _current_addresses(self):
        pool = self.pool
        full = pool.full_kv_pool.buffer if hasattr(pool, "full_kv_pool") else pool._kv_buffer
        result = {
            "full": full.data_ptr(),
            "page_table": self.engine.page_table.data_ptr(),
            "pinned": self.cache.shift.pinned.data_ptr(),
            "slot_for_id": self.cache.slot_for_id.data_ptr(),
            "id_of_slot": self.cache.id_of_slot.data_ptr(),
            "usage": self.cache.usage.data_ptr(),
            "step": self.cache.step.data_ptr(),
            "copy_src_indices": self.cache.src_indices.data_ptr(),
            "copy_dst_indices": self.cache.evict_slots.data_ptr(),
            "copy_count": self.cache.num_indices.data_ptr(),
        }
        if "swa" in self.layouts:
            result["swa"] = pool.swa_kv_pool.buffer.data_ptr()
            result["swa_mapping"] = pool.full_to_swa_index_mapping.data_ptr()
        result.update({f"expert_{n}": b.data_ptr() for n, b in self.cache.bank_caches.items()})
        return result

    def _assert_invariants(self):
        if self._failed:
            raise RuntimeError("elastic KV mapping failed; engine cannot safely continue")
        assert self._current_addresses() == self._addresses, "CUDA graph address changed"
        assert self.base_slots <= self.active_slots <= self.max_slots
        assert self.committed_bytes <= self.physical_budget
        pool = self.vmm.pool_snapshot()
        if pool is not None:
            assert pool["physical_owned_bytes"] <= self.physical_budget
            assert pool["physical_mapped_bytes"] == self.committed_bytes
            assert (
                pool["physical_owned_bytes"] == self.committed_bytes + pool["physical_idle_bytes"]
            )
        for name, layout in self.layouts.items():
            assert self.kv_regions[name].mapped == layout.chunks(self.mapped[name], self.chunk)
        for row, region in zip(self.row_bytes, self.bank_regions.values()):
            assert region.mapped == set(range(ceil_div(self.active_slots * row, self.chunk)))

    @property
    def committed_bytes(self):
        return sum(
            len(r.mapped) * r.chunk
            for r in (*self.kv_regions.values(), *self.bank_regions.values())
        )

    def _rounded(self, name, units):
        units = int(units)
        step = (
            max(1, ceil_div(self.growth_tokens, self.page_size))
            if name == "full"
            else self.growth_tokens
        )
        floor = min(
            self.layouts[name].usable,
            ceil_div(self.initial_tokens, self.page_size)
            if name == "full"
            else self.swa_initial_tokens,
        )
        return min(self.layouts[name].usable, max(floor, ceil_div(units, step) * step))

    def _resize(self, targets, reason):
        import torch
        from freetoken.moe.shift import _invalidate_range

        targets = {n: self._rounded(n, targets.get(n, self.mapped[n])) for n in self.layouts}
        if targets == self.mapped:
            return
        self._assert_invariants()
        started = time.perf_counter()
        kv_maps = {n: layout.chunks(targets[n], self.chunk) for n, layout in self.layouts.items()}
        kv_bytes = sum(len(m) * self.chunk for m in kv_maps.values())
        slots = fit_slots(
            self.physical_budget - kv_bytes, self.row_bytes, self.chunk, self.max_slots
        )
        if slots < self.base_slots:
            raise RuntimeError("elastic KV growth would consume the activation arena/base experts")
        bank_maps = {
            name: set(range(ceil_div(slots * row, self.chunk)))
            for name, row in zip(self.bank_regions, self.row_bytes)
        }
        previous = dict(self.mapped)
        old_slots = self.active_slots
        ownership_changed = slots != old_slots or any(
            kv_maps[n] != self.kv_regions[n].mapped for n in self.kv_regions
        )
        if ownership_changed:
            # Also drains prior overlap decode and prefill copy streams. No mapped
            # physical page changes ownership while a GPU operation can reference it.
            torch.cuda.synchronize(self.device)
            if slots < old_slots:
                self.invalidated_experts += int(
                    (self.cache.id_of_slot[slots:old_slots] >= 0).sum().item()
                )
                self.cache.shift.pinned[slots:] = 1
                _invalidate_range(self.cache, slots, old_slots)
                torch.cuda.synchronize(self.device)
            try:
                # Release both kinds before any create: physical usage cannot exceed
                # the fixed budget, even transiently during a bidirectional exchange.
                for name, region in self.bank_regions.items():
                    region.set_chunks(bank_maps[name], release_only=True)
                for name, region in self.kv_regions.items():
                    region.set_chunks(kv_maps[name], release_only=True)
                for name, region in self.kv_regions.items():
                    region.set_chunks(kv_maps[name])
                for name, region in self.bank_regions.items():
                    region.set_chunks(bank_maps[name])
            except Exception:
                self._failed = True
                raise
            if slots > old_slots:
                # Rows being re-enabled are empty; no stale expert identity survives
                # a previous KV ownership interval.
                _invalidate_range(self.cache, old_slots, slots)
                self.cache.shift.pinned[old_slots:slots] = 0
            self.cache.shift.pinned[slots:] = 1
        self.mapped, self.active_slots = targets, slots
        self.min_active_slots = min(self.min_active_slots, slots)
        for name in targets:
            self.max_mapped[name] = max(self.max_mapped[name], targets[name])
        self.grows += int(any(targets[n] > previous[n] for n in targets))
        self.shrinks += int(any(targets[n] < previous[n] for n in targets))
        elapsed = time.perf_counter() - started
        self.remap_seconds += elapsed
        self.events.append(
            dict(
                reason=reason,
                previous=previous,
                mapped=dict(targets),
                old_expert_slots=old_slots,
                expert_slots=slots,
                seconds=elapsed,
                ownership_changed=ownership_changed,
                committed_bytes=self.committed_bytes,
            )
        )
        self._assert_invariants()

    def _free_tensor(self, name):
        return self.cm.free_slots if name == "full" else self.pool._swa_free

    def _set_free_tensor(self, name, tensor):
        if name == "full":
            self.cm.free_slots = tensor
        else:
            self.pool._swa_free = tensor

    def _refresh(self, name):
        """Sort at phase/growth boundaries, never on each decode token."""
        tensor = self._free_tensor(name).sort().values
        self._set_free_tensor(name, tensor)
        stride = self.page_size if name == "full" else 1
        values = tensor.cpu().numpy() // stride
        import bisect

        self.ready[name] = bisect.bisect_left(values, self.mapped[name])
        return values

    def _before_allocate(self, name, count):
        if count <= 0:
            return
        if self._failed:
            raise RuntimeError("elastic KV controller is failed")
        if count > self.ready[name]:
            free = self._refresh(name)
            if count > len(free):
                raise RuntimeError(f"elastic {name} allocation exceeds logical free capacity")
            required = int(free[count - 1]) + 1
            if required > self.mapped[name]:
                self._resize({name: required}, f"{name}_allocate")
                import bisect

                self.ready[name] = bisect.bisect_left(free, self.mapped[name])
        self.ready[name] -= count

    def before_full_allocate(self, cm, count):
        assert cm is self.cm
        self._before_allocate("full", count)

    def before_swa_allocate(self, pool, count):
        assert pool is self.pool
        self._before_allocate("swa", count)

    def _reconcile_free(self, reason):
        targets, frees = {}, {}
        for name, layout in self.layouts.items():
            free = self._refresh(name)
            frees[name] = free
            targets[name] = high_water_from_free(free, layout.usable, int(name == "swa"))
        # Hysteresis: retain one growth quantum when only a tiny tail was freed.
        for name in targets:
            step = (
                max(1, ceil_div(self.growth_tokens, self.page_size))
                if name == "full"
                else self.growth_tokens
            )
            if self.mapped[name] - self._rounded(name, targets[name]) < 2 * step:
                targets[name] = self.mapped[name]
        self._resize(targets, reason)
        import bisect

        for name, free in frees.items():
            self.ready[name] = bisect.bisect_left(free, self.mapped[name])

    def after_prefill(self, cm, req):
        """Defer relocation until no prepared attention metadata can become stale."""
        pin = self._full_prefix_pins.get(req.uid)
        if (
            pin is not None
            and not pin.hold_until_finish
            and req.cache_handle.cached_len >= pin.matched_tokens
        ):
            self._release_full_prefix(cm, req.uid, "prefill_commit")
        if not self.compact_swa:
            return
        assert cm is self.cm
        self.pending_compaction = True
        if cm.is_swa and req.mm_embeds is None:
            from freetoken.scheduler.cache import _SWA_RETAIN_GAP

            committed = (req.cached_len // self.page_size) * self.page_size
            keep_from = max(0, committed - cm.sliding_window_size - _SWA_RETAIN_GAP)
            keep_from = (keep_from // self.page_size) * self.page_size
            if keep_from:
                # CPU token ids; preserve this exact committed prefix even if
                # the request finishes before the next preparation safe point.
                self._pending_swa_trim.append((req.input_ids[:committed].clone(), keep_from))

    def _trim_swa_heads(self):
        if not self._pending_swa_trim:
            return
        for ids, keep_from in self._pending_swa_trim:
            full_indices = self.cm.prefix_cache.trim_head_swa(ids, keep_from)
            self.swa_phase_trimmed_tokens += int(full_indices.numel())
            self.cm._free_swa(full_indices)
        self._pending_swa_trim.clear()

    def _compact_swa(self, reason):
        """Caller holds a pre-metadata safe point and has drained GPU work."""
        if not self.compact_swa:
            return
        import torch

        started = time.perf_counter()
        capacity = self.layouts["swa"].usable
        live_count = capacity - 1 - int(self.pool._swa_free.numel())
        target = self._rounded("swa", live_count + 1)
        if target >= self.mapped["swa"]:
            return
        mapping = self.pool.full_to_swa_index_mapping
        plan = plan_swa_compaction(
            mapping.cpu().numpy(), self.pool._swa_free.cpu().numpy(), capacity, self.mapped["swa"]
        )
        moved = len(plan.source)
        if not moved:
            return  # _reconcile_free already shrinks a free contiguous tail.
        source = torch.as_tensor(plan.source, dtype=torch.int64, device=self.device)
        destination = torch.as_tensor(plan.destination, dtype=torch.int64, device=self.device)
        full_indices = torch.as_tensor(plan.full_indices, dtype=torch.int64, device=self.device)
        try:
            from .elastic_swa import copy_disjoint_swa_slots

            copy_disjoint_swa_slots(self.pool.swa_kv_pool.buffer, source, destination)
            # Only this indirection changes. Radix nodes and full attention page
            # tables hold canonical FULL indices and remain untouched.
            mapping[full_indices] = destination
            self.pool._swa_free = torch.arange(
                plan.live_count + 1, capacity, dtype=torch.int32, device=self.device
            )
            torch.cuda.synchronize(self.device)
        except Exception:
            self._failed = True
            raise
        elapsed = time.perf_counter() - started
        d2d_bytes = moved * self.layouts["swa"].stripes * self.layouts["swa"].unit_bytes
        # Three relocation vectors plus a replacement free list can temporarily
        # coexist with the old one. There is no KV payload scratch allocation.
        metadata_bytes = moved * 24 + int(self.pool._swa_free.numel()) * 8
        self.swa_compactions += 1
        self.swa_compacted_tokens += moved
        self.swa_compaction_d2d_bytes += d2d_bytes
        self.swa_compaction_metadata_peak_bytes = max(
            self.swa_compaction_metadata_peak_bytes, metadata_bytes
        )
        self.swa_compaction_seconds += elapsed
        self.swa_compaction_events.append(
            dict(
                reason=reason,
                live_tokens=plan.live_count,
                moved_tokens=moved,
                old_high_water=plan.high_water,
                new_high_water=plan.live_count + 1,
                d2d_bytes=d2d_bytes,
                metadata_transient_bytes=metadata_bytes,
                seconds=elapsed,
            )
        )

    def before_prepare_batch(self, cm):
        """Runs BEFORE allocation and attention metadata, including overlap mode."""
        assert cm is self.cm
        if not self.pending_compaction:
            return
        import torch

        # Last-data processing may mark this pending while the next forward has
        # already launched. Drain it here; its metadata is no longer reusable.
        torch.cuda.synchronize(self.device)
        self._trim_swa_heads()
        self._compact_swa("phase_boundary")
        self._reconcile_free("phase_boundary")
        self.pending_compaction = False

    def _release_full_prefix(self, cm, uid, reason):
        pin = self._full_prefix_pins.pop(uid, None)
        if pin is not None:
            if pin.cache is not cm.prefix_cache:
                raise RuntimeError("FULL-prefix pin outlived its prefix tree")
            pin.release()
            self.full_prefix_pin_events.append(
                dict(
                    uid=int(uid), reason=reason, matched_tokens=pin.matched_tokens, action="release"
                )
            )

    def on_admission_cancel(self, cm, pending_req):
        self._release_full_prefix(cm, pending_req.uid, "admission_cancel")

    def after_request_release(self, cm, req):
        self._release_full_prefix(cm, req.uid, "request_release")

    def on_admission(self, cm, pending_req=None, *, reused_tokens=0, required_tokens=0):
        """The caller has locked this request's matched prefix before calling us."""
        if self.compact_swa:
            import torch

            torch.cuda.synchronize(self.device)
            self._trim_swa_heads()
        purge_allowed = self.evict_unmatched_prefixes
        if purge_allowed and cm.is_hybrid:
            # Hybrid admission resumes only at a GDN checkpoint. Native commit
            # may additionally deduplicate a longer, matching FULL-KV prefix.
            # Do not remove its canonical bytes merely because GDN cannot yet
            # skip their recomputation.
            if pending_req is None or getattr(self.engine.config, "max_running_req", 1) > 1:
                # Extra canonical-byte retention is currently proven for the
                # single active request / personal-agent serving configuration.
                purge_allowed = False
            elif pending_req.mm_embeds is None:
                from .full_prefix import reserve_full_prefix

                if pending_req.uid in self._full_prefix_pins:
                    raise RuntimeError("duplicate FULL-prefix reservation for request uid")
                decision = reserve_full_prefix(
                    cm.prefix_cache,
                    pending_req.input_ids,
                    reused_tokens=reused_tokens,
                    required_tokens=required_tokens,
                    available_tokens=cm.available_size,
                    allow_pinning=cm.linear_state_pool.num_free_slots >= 4,
                )
                # Native GDN-pressure eviction can delete FULL leaves too.
                # Additional FULL refs would turn that deletion into tombstoning.
                # Avoid extra pins if live/ping-pong + checkpoint replacement
                # cannot fit without such an eviction. A wholly unrelated new
                # request needs no pin and can still purge unrelated branches.
                gdn_pressure = decision.fallback_reason == "gdn_pressure"
                purge_allowed = decision.purge_allowed
                if decision.pin is not None:
                    self._full_prefix_pins[pending_req.uid] = decision.pin
                self.full_prefix_capacity_fallbacks += int(not purge_allowed)
                self.full_prefix_pin_events.append(
                    dict(
                        uid=int(pending_req.uid),
                        action="admission",
                        matched_tokens=decision.matched_tokens,
                        reused_tokens=int(reused_tokens),
                        extra_tokens=decision.extra_tokens,
                        slack_tokens=decision.slack_tokens,
                        purge_allowed=purge_allowed,
                        pinned=decision.pin is not None,
                        hold_until_finish=bool(decision.pin and decision.pin.hold_until_finish),
                        gdn_pressure_fallback=gdn_pressure,
                    )
                )
        if purge_allowed:
            evictable = (
                cm.prefix_cache.full_evictable_size
                if cm.is_hybrid or cm.is_swa
                else cm.prefix_cache.size_info.evictable_size
            )
            if evictable:
                if cm.is_swa:
                    ev = cm.prefix_cache.evict_full(evictable)
                    cm._free_swa(ev.swa_indices)
                    indices = ev.kv_indices
                elif cm.is_hybrid:
                    ev = cm.prefix_cache.evict_full(evictable)
                    if ev.mamba_slots:
                        cm.linear_state_pool.free(ev.mamba_slots)
                    indices = ev.kv_indices
                else:
                    indices = cm.prefix_cache.evict(evictable)
                cm._free(indices)
                self.evicted_prefix_tokens += int(indices.numel())
        if self.compact_swa:
            self._compact_swa("admission")
            self.pending_compaction = False
        self._reconcile_free("admission")

    def after_finish(self, cm):
        # Keep the just-produced conversation prefix. Only entirely free top
        # ranges shrink here; unrelated prefixes are considered at admission.
        self._reconcile_free("finish")
        # An overlapped forward may already have prepared its translated SWA
        # metadata. Do not relocate from last-data processing itself.
        self.pending_compaction = self.compact_swa

    def before_cache_reset(self, cm):
        if self._full_prefix_pins:
            raise RuntimeError("cannot reset prefix cache with live FULL-prefix reservations")

    def on_cache_reset(self, cm):
        self.before_cache_reset(cm)
        self.pending_compaction = False
        self._pending_swa_trim.clear()
        self.ready = {"full": 0, "swa": 0}
        self._reconcile_free("cache_reset")

    def snapshot(self):
        self._assert_invariants()
        return dict(
            mode="online_vmm",
            max_context_tokens=self.engine.max_seq_len,
            base_expert_slots=self.base_slots,
            max_expert_slots=self.max_slots,
            active_expert_slots=self.active_slots,
            min_active_expert_slots=self.min_active_slots,
            full_mapped_tokens=self.mapped["full"] * self.page_size,
            swa_mapped_tokens=self.mapped.get("swa"),
            full_peak_mapped_tokens=self.max_mapped["full"] * self.page_size,
            swa_peak_mapped_tokens=self.max_mapped.get("swa"),
            physical_budget_bytes=self.physical_budget,
            physical_committed_bytes=self.committed_bytes,
            mapping_granularity_bytes=self.chunk,
            initial_tokens=self.initial_tokens,
            physical_handle_pool=self.vmm.pool_snapshot(),
            swa_initial_tokens=self.swa_initial_tokens,
            growth_tokens=self.growth_tokens,
            grows=self.grows,
            shrinks=self.shrinks,
            compact_swa=self.compact_swa,
            swa_compactions=self.swa_compactions,
            swa_compacted_tokens=self.swa_compacted_tokens,
            swa_compaction_d2d_bytes=self.swa_compaction_d2d_bytes,
            swa_compaction_kv_scratch_bytes=0,
            swa_compaction_metadata_peak_bytes=self.swa_compaction_metadata_peak_bytes,
            swa_compaction_seconds=self.swa_compaction_seconds,
            swa_phase_trimmed_tokens=self.swa_phase_trimmed_tokens,
            swa_compaction_events=list(self.swa_compaction_events),
            arena_rebuild_lifetime=self.arena_rebuild_lifetime,
            seed_metadata_warmup=self.seed_metadata_warmup,
            invalidated_experts=self.invalidated_experts,
            evicted_unmatched_prefix_tokens=self.evicted_prefix_tokens,
            pending_full_prefix_pins=len(self._full_prefix_pins),
            full_prefix_capacity_fallbacks=self.full_prefix_capacity_fallbacks,
            full_prefix_pin_events=list(self.full_prefix_pin_events),
            remap_seconds=self.remap_seconds,
            graph_addresses_stable=True,
            graph_recaptures_during_serving=0,
            events=list(self.events),
        )


def install_elastic_kv(
    llm,
    *,
    initial_tokens=8192,
    growth_tokens=2048,
    max_slots=None,
    evict_unmatched_prefixes=True,
    swa_initial_tokens=None,
    compact_swa=False,
    chunk_mib=2,
    pool_physical_handles=False,
):
    import torch

    # Match the engine's original graph-capture context: inference tensors and
    # all warm-up/capture work issued on the engine stream.
    with torch.inference_mode(), torch.cuda.stream(llm.engine.stream):
        return ElasticKV(
            llm,
            initial_tokens=initial_tokens,
            growth_tokens=growth_tokens,
            max_slots=max_slots,
            evict_unmatched_prefixes=evict_unmatched_prefixes,
            swa_initial_tokens=swa_initial_tokens,
            compact_swa=compact_swa,
            chunk_mib=chunk_mib,
            pool_physical_handles=pool_physical_handles,
        )
