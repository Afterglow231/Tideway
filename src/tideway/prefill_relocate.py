"""Bounded D2D retention before native Shift Prefill invalidates its workspace.

Explicit opt-in; no host payload upload/writeback, no large payload temporary.
Selection uses existing LRU recency, not a Decode EMA/frequency or future route.
Only one native GPU owner with a complete BF16 CPU master is supported.
"""

import os
import time

import torch

_TRUE = ("1", "true", "yes", "on")


def _require_cuda(cache):
    if cache.device.type != "cuda":
        raise ValueError("Prefill relocation requires a CUDA cache")


def select_moves(
    inverse,
    lookup,
    usage,
    pinned,
    *,
    active,
    experts,
    arena_lo,
    arena_hi,
    head,
    pin_base,
    pin_total,
    max_rows,
):
    """Pure bounded metadata plan; all selected row pairs have disjoint storage."""
    size = len(inverse)
    if not (
        len(usage) == len(pinned) == size and 2 * experts <= arena_lo < arena_hi <= active <= size
    ):
        raise RuntimeError("invalid active/arena/cache bounds")
    if head < 1 or arena_lo + head > arena_hi:
        raise RuntimeError("invalid permanent arena head")
    if (
        any(gid < -1 for gid in inverse)
        or any(slot < -1 for slot in lookup)
        or any(age < 0 for age in usage)
    ):
        raise RuntimeError("unsupported negative ownership or LRU metadata")
    for slot, gid in enumerate(inverse):
        if slot >= active and (gid >= 0 or not pinned[slot]):
            raise RuntimeError("unmapped tail has a live owner or is not protected")
        if arena_lo <= slot < arena_lo + head and (gid >= 0 or not pinned[slot]):
            raise RuntimeError("arena head cannot contain an expert or be unprotected")
        if gid >= 0 and (gid >= len(lookup) or lookup[gid] != slot):
            raise RuntimeError("inverse expert ownership disagrees with lookup")
    for gid, slot in enumerate(lookup):
        if slot >= 0 and (slot >= active or inverse[slot] != gid):
            raise RuntimeError("lookup refers to invalid/inconsistent physical ownership")

    def core(slot):
        return pin_base <= slot < pin_base + pin_total

    def overwritten(slot):
        return slot < 2 * experts or arena_lo <= slot < arena_hi

    sources = [
        slot
        for slot in range(active)
        if overwritten(slot)
        and inverse[slot] >= 0
        and not pinned[slot]
        and not core(slot)
        and usage[slot] > 0
    ]
    targets = [
        slot
        for slot in range(2 * experts, active)
        if not overwritten(slot) and not pinned[slot] and not core(slot)
    ]
    sources.sort(key=lambda slot: (-usage[slot], slot))
    targets.sort(key=lambda slot: (inverse[slot] >= 0, usage[slot], slot))
    pairs = []
    for source, target in zip(sources, targets):
        if len(pairs) >= max_rows:
            break
        if inverse[target] >= 0 and usage[source] <= usage[target]:
            break
        pairs.append(
            dict(
                source=source,
                destination=target,
                expert=inverse[source],
                displaced_expert=inverse[target],
                source_usage=usage[source],
                destination_usage=usage[target],
            )
        )
    src, dst = {p["source"] for p in pairs}, {p["destination"] for p in pairs}
    if len(src) != len(pairs) or len(dst) != len(pairs) or src & dst:
        raise RuntimeError("overlapping/nonunique relocation transaction")
    return pairs


class PrefillRelocation:
    def __init__(self, cache, shift_module, max_rows=64):
        if type(max_rows) is not int or max_rows not in (64, 128, 256):
            raise ValueError("max_rows must be 64, 128 or 256")
        _require_cuda(cache)
        st = cache.shift
        if (
            st is None
            or cache.decode_target != "gpu"
            or "ensure_experts" in vars(cache)
            or cache.quant_format != "bf16"
            or not cache.prefill_overlap
        ):
            raise ValueError("relocation requires native GPU Shift BF16 with full-layer buffers")
        if st.policy != "lru" or st.arena <= 0 or st.arena_lib is None:
            raise ValueError("relocation requires a native LRU activation arena")
        if st.pin_total != 0:
            raise ValueError("initial relocation arm requires zero fixed seed/core rows")
        if os.getenv("TIDEWAY_SHIFT_PREBEGIN", "1").strip().lower() in _TRUE:
            raise ValueError(
                "disable idle PREBEGIN before installing relocation; already overwritten rows cannot be rescued"
            )
        if os.getenv("FREETOKEN_SKIP_FAST_INDEX_COPY", "").strip().lower() in _TRUE:
            raise ValueError("relocation cannot run with payload-copy skipping")
        if (
            not cache._copy_fused_ok
            or cache._copy_dst_ptrs is None
            or cache._copy_feat_bytes is None
        ):
            raise ValueError("complete fused multi-bank GPU copy plan is required")
        if not cache.banks or len(cache.banks) != len(cache._copy_feat_bytes_host):
            raise ValueError("all payload banks must be present in the copy descriptor")
        if (
            cache._copy_dst_ptrs.detach().cpu().tolist() != cache._copy_dst_ptrs_host
            or cache._copy_feat_bytes.detach().cpu().tolist() != cache._copy_feat_bytes_host
        ):
            raise ValueError("device copy descriptor differs from its validated bank metadata")
        for index, (host, bank) in enumerate(cache.banks):
            row_bytes = bank[0].numel() * bank.element_size()
            if (
                bank.device != cache.device
                or not bank.is_contiguous()
                or bank.dtype != torch.bfloat16
                or bank.shape[0] != cache.cache_size
                or row_bytes != cache._copy_feat_bytes_host[index]
                or bank.data_ptr() != cache._copy_dst_ptrs_host[index]
            ):
                raise ValueError(
                    "GPU bank geometry/pointer differs from the complete copy descriptor"
                )
            if len(host) != cache.num_layers or any(
                t.device.type != "cpu"
                or t.dtype != bank.dtype
                or t.shape != (cache.num_experts, *bank.shape[1:])
                for t in host
            ):
                raise ValueError("each bank requires the complete per-layer BF16 CPU master")
        self.cache, self.module, self.owner = cache, shift_module, st
        self.max_rows = max_rows
        self.row_bytes = sum(cache._copy_feat_bytes_host)
        self.original_prepare, self.original_invalidate = (
            shift_module.arena_prepare,
            shift_module._invalidate_range,
        )
        self.bank_signature = tuple(
            (id(bank), bank.data_ptr(), tuple(bank.shape), bank.dtype) for _, bank in cache.banks
        )
        self.ptrs_signature = (id(cache._copy_dst_ptrs), id(cache._copy_feat_bytes))
        self.src = torch.empty(max_rows, dtype=torch.int32, device=cache.device)
        self.dst = torch.empty_like(self.src)
        self.num = torch.zeros((1,), dtype=torch.int64, device=cache.device)
        self.host_src = torch.empty(max_rows, dtype=torch.int32, pin_memory=True)
        self.host_dst = torch.empty_like(self.host_src, pin_memory=True)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.state, self.failure, self.inside, self.tokens = "installed", None, False, None
        self.pending_stream = None
        self.events = []
        self.stats = dict(
            prefills=0,
            migrations=0,
            moved_rows=0,
            d2d_payload_bytes=0,
            d2d_gpu_ms=0.0,
            hook_wall_s=0.0,
            metadata_dtoh_bytes=0,
            index_upload_bytes=0,
            skipped_small_prefill=0,
            failures=0,
        )

        def prepare(owner_cache, n_tok):
            if owner_cache is not self.cache:
                return self.original_prepare(owner_cache, n_tok)
            if self.state != "installed":
                raise RuntimeError(f"relocation is {self.state}: {self.failure}")
            if self.inside:
                raise RuntimeError("nested Prefill relocation is unsupported")
            self.inside, self.tokens = True, int(n_tok)
            try:
                self._validate_live()
                return self.original_prepare(owner_cache, n_tok)
            except BaseException as error:
                self._fail(error)
                raise
            finally:
                self.inside, self.tokens = False, None

        def invalidate(owner_cache, lo, hi):
            if self.inside and owner_cache is self.cache:
                if lo != self.owner.elastic_lo or hi != self.owner.arena_inval_hi:
                    raise RuntimeError("native arena selected an unexpected invalidation range")
                self._relocate(int(lo), int(hi))
            return self.original_invalidate(owner_cache, lo, hi)

        self.prepare_hook, self.invalidate_hook = prepare, invalidate
        try:
            shift_module.arena_prepare = prepare
            shift_module._invalidate_range = invalidate
        except BaseException:
            if shift_module.arena_prepare is prepare:
                shift_module.arena_prepare = self.original_prepare
            if shift_module._invalidate_range is invalidate:
                shift_module._invalidate_range = self.original_invalidate
            raise

    def _fail(self, error):
        if self.state != "failed":
            self.stats["failures"] += 1
        self.state, self.failure = "failed", f"{type(error).__name__}: {error}"

    def _validate_live(self):
        cache, st = self.cache, self.owner
        if cache.shift is not st or st.policy != "lru" or st.pin_total != 0:
            raise RuntimeError("Shift owner/policy/core geometry changed")
        if cache._prestaged or not all(cache._prefill_buffer_released):
            raise RuntimeError(
                "idle-prestaged or in-flight Prefill buffers cannot be relocated safely"
            )
        if os.getenv("TIDEWAY_SHIFT_PREBEGIN", "1").strip().lower() in _TRUE:
            raise RuntimeError("idle PREBEGIN enabled after relocation installation")
        if (
            os.getenv("FREETOKEN_SKIP_FAST_INDEX_COPY", "").strip().lower() in _TRUE
            or not cache._copy_fused_ok
        ):
            raise RuntimeError("the complete fused payload copy path was disabled")
        if (
            tuple(
                (id(bank), bank.data_ptr(), tuple(bank.shape), bank.dtype)
                for _, bank in cache.banks
            )
            != self.bank_signature
        ):
            raise RuntimeError("expert bank storage changed after relocation installation")
        if (id(cache._copy_dst_ptrs), id(cache._copy_feat_bytes)) != self.ptrs_signature:
            raise RuntimeError("fused copy descriptor changed after relocation installation")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prefill relocation cannot execute inside a captured graph")

    @torch.no_grad()
    def _relocate(self, lo, hi):
        cache, st = self.cache, self.owner
        started = time.perf_counter()
        self.stats["prefills"] += 1
        if self.tokens <= st.small_prefill_max:
            self.stats["skipped_small_prefill"] += 1
            return
        stream = torch.cuda.current_stream(cache.device)
        self.pending_stream = stream
        # Complete earlier routed compute and old Prefill copies before reading
        # owner metadata. No payload source can be concurrently overwritten.
        stream.synchronize()
        if cache.prefill_copy_stream is not None:
            cache.prefill_copy_stream.synchronize()
        active = int(getattr(getattr(cache, "elastic_kv", None), "active_slots", cache.cache_size))
        inverse = cache.id_of_slot.detach().cpu().tolist()
        lookup = cache.slot_for_id.detach().cpu().reshape(-1).tolist()
        usage = cache.usage.detach().cpu().tolist()
        pinned = st.pinned.detach().cpu().tolist()
        meta_bytes = sum(
            t.numel() * t.element_size()
            for t in (cache.id_of_slot, cache.slot_for_id, cache.usage, st.pinned)
        )
        self.stats["metadata_dtoh_bytes"] += meta_bytes
        pairs = select_moves(
            inverse,
            lookup,
            usage,
            pinned,
            active=active,
            experts=cache.num_experts,
            arena_lo=lo,
            arena_hi=hi,
            head=st.arena_head,
            pin_base=st.pin_base,
            pin_total=st.pin_total,
            max_rows=self.max_rows,
        )
        n = len(pairs)
        elapsed_ms = 0.0
        if n:
            source = [p["source"] for p in pairs]
            target = [p["destination"] for p in pairs]
            self.host_src[:n].copy_(torch.tensor(source, dtype=torch.int32))
            self.host_dst[:n].copy_(torch.tensor(target, dtype=torch.int32))
            self.src[:n].copy_(self.host_src[:n], non_blocking=True)
            self.dst[:n].copy_(self.host_dst[:n], non_blocking=True)
            self.num.fill_(n)
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            self.start_event.record(stream)
            fast_index_copy_multi_jit(
                cache._copy_dst_ptrs,
                cache._copy_dst_ptrs,
                cache._copy_feat_bytes,
                self.dst,
                self.src,
                self.num,
                blocks_per_bank=64,
            )
            self.end_event.record(stream)
            # Metadata-only gathers, never an expert payload scratch. All
            # publication kernels follow payload copy on this same stream.
            src, dst = self.src[:n].long(), self.dst[:n].long()
            saved_usage = cache.usage.index_select(0, src)
            gids = cache.id_of_slot.index_select(0, src)
            displaced = [p["displaced_expert"] for p in pairs if p["displaced_expert"] >= 0]
            flat = cache.slot_for_id.view(-1)
            if displaced:
                flat[torch.tensor(displaced, dtype=torch.long, device=cache.device)] = -1
            cache.id_of_slot[src] = -1
            cache.usage[src] = 0
            cache.id_of_slot[dst] = gids
            flat[gids.long()] = self.dst[:n]
            cache.usage[dst] = saved_usage
            # Draining keeps the pinned index source safe and prevents the
            # subsequent arena invalidation/allocations from racing relocation.
            stream.synchronize()
            elapsed_ms = float(self.start_event.elapsed_time(self.end_event))
            self.stats["migrations"] += 1
            self.stats["moved_rows"] += n
            self.stats["d2d_payload_bytes"] += n * self.row_bytes
            self.stats["d2d_gpu_ms"] += elapsed_ms
            self.stats["index_upload_bytes"] += 8 * n + 8 * len(displaced)
        wall = time.perf_counter() - started
        self.stats["hook_wall_s"] += wall
        self.pending_stream = None
        if len(self.events) < 256:
            self.events.append(
                dict(
                    tokens=self.tokens,
                    active_slots=active,
                    arena_lo=lo,
                    arena_hi=hi,
                    arena_head=st.arena_head,
                    max_rows=self.max_rows,
                    moves=pairs,
                    moved_rows=n,
                    d2d_payload_bytes=n * self.row_bytes,
                    d2d_gpu_ms=elapsed_ms,
                    hook_wall_s=wall,
                    metadata_dtoh_bytes=meta_bytes,
                )
            )

    def snapshot(self):
        return dict(
            enabled=True,
            state=self.state,
            failure=self.failure,
            policy="bounded LRU-recency D2D relocation before native Prefill workspace invalidation; not Decode EMA",
            max_rows=self.max_rows,
            row_bytes=self.row_bytes,
            counters=dict(self.stats),
            events=list(self.events),
            events_truncated=self.stats["prefills"] > len(self.events),
            payload_scratch_bytes=0,
            host_payload_upload_bytes=0,
            host_payload_writeback_bytes=0,
            metadata_scratch_capacity_bytes=sum(
                t.numel() * t.element_size()
                for t in (self.src, self.dst, self.num, self.host_src, self.host_dst)
            ),
            latency_charged_to="same Prefill request P/TTFT and E2E; no forward retry",
            changes_KV_mapping=False,
            changes_decode_graph_addresses=False,
        )

    def close(self):
        if self.state == "closed":
            return
        if self.pending_stream is not None:
            try:
                self.pending_stream.synchronize()
            except BaseException as error:
                self._fail(error)
                raise
            self.pending_stream = None
        if (
            self.module.arena_prepare is not self.prepare_hook
            or self.module._invalidate_range is not self.invalidate_hook
        ):
            raise RuntimeError("Prefill relocation hooks changed while installed")
        self.module.arena_prepare = self.original_prepare
        self.module._invalidate_range = self.original_invalidate
        self.state = "closed"


def install_prefill_relocation(llm, max_rows=64):
    from freetoken.moe import shift

    cache = llm.engine.moe_offload_cache
    if cache is None or getattr(cache, "prefill_relocation", None) is not None:
        raise ValueError("a fresh native Shift cache owner is required")
    controller = PrefillRelocation(cache, shift, max_rows=max_rows)
    try:
        cache.prefill_relocation = controller
    except BaseException as error:
        try:
            controller.close()
        except BaseException as cleanup_error:
            error.add_note(f"Relocation hook rollback failed: {cleanup_error!r}")
        raise
    return controller
