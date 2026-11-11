"""Optional prefill retention that only fills empty, currently usable slots.

No resident is evicted or promoted. A chunk-start quota prevents the first
layers from consuming the free pool; unused quotas stay free until next chunk.
All per-layer scratch is fixed at attach time. Serving has no host readback,
dynamic output shape, histogram allocation, or change to decode CUDA graphs.
"""

from __future__ import annotations

import os
import time

import torch
import triton
import triton.language as tl


def enabled() -> bool:
    return os.getenv("TIDEWAY_SHIFT_SEED_EMPTY", "0").strip().lower() in ("1", "true", "yes", "on")


def routing_layout(ids):
    """Only already-observed tensor metadata, never token contents or values."""
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected int32/int64 expert routing ids")
    if ids.ndim == 2:
        width, row_stride, col_stride = ids.shape[1], *ids.stride()
    elif ids.ndim == 1:
        width, row_stride, col_stride = 1, ids.stride(0), 0
    else:
        raise ValueError("Expected one- or two-dimensional expert routing ids")
    if width <= 0 or min(row_stride, col_stride) < 0:
        raise ValueError("Invalid expert routing layout")
    # Triton specializes pointer alignment as well as the pointer element type.
    # The standard routers are aligned; retain non-aligned sliced views too.
    address = ids.data_ptr()
    alignment = next(n for n in (16, 8, 4, 2, 1) if address % n == 0)
    return ids.dtype, int(width), int(row_stride), int(col_stride), alignment


def observe_routing_layout(cache, ids):
    layout = routing_layout(ids)
    layouts = getattr(cache, "_seed_empty_route_layouts", None)
    if layouts is None:
        layouts = cache._seed_empty_route_layouts = set()
    layouts.add(layout)  # cache survives the VMM bank rebuild / Shift re-attach.
    return layout


def _dummy_routing(layout, device):
    dtype, width, row_stride, col_stride, alignment = layout
    element = dtype.itemsize
    if alignment < element:
        raise ValueError("Routing pointer alignment is smaller than its element")
    offset = 0 if alignment == 16 else alignment // element
    # One row is enough: quota==0 means the warmup never reads any routing id.
    storage = torch.empty(offset + (width - 1) * col_stride + 1, dtype=dtype, device=device)
    return torch.as_strided(storage, (1, width), (row_stride, col_stride), storage_offset=offset)


def excluded_interval(st, cache_size: int) -> tuple[int, int]:
    """Current activation lease. The pinned mask separately excludes KV pages."""
    if st.elastic <= 0:
        return cache_size, cache_size
    lo, end = st.elastic_lo, st.elastic_lo + st.elastic
    if st.arena > 0 and st.arena_lib is not None:
        hi = st.arena_inval_hi
        if not lo + st.arena_head <= hi <= end:
            raise RuntimeError("Empty seeding requires the current prefill arena lease")
        return lo, hi
    return lo, end


@triton.jit(do_not_specialize=["rotation", "exclude_lo", "exclude_hi"])
def _prepare_quotas(
    owners,
    pinned,
    quotas,
    counters,
    clock,
    rotation,
    exclude_lo,
    exclude_hi,
    E: tl.constexpr,
    L: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    s = tl.arange(0, BLOCK_C)
    owner = tl.load(owners + s, s < C, -1)
    pin = tl.load(pinned + s, s < C, 1)
    allowed = (s >= 2 * E) & (s < C) & ((s < exclude_lo) | (s >= exclude_hi))
    free = allowed & (owner < 0) & (pin == 0)
    count = tl.sum(free.to(tl.int32))
    layer = tl.arange(0, BLOCK_L)
    # Rotate only the remainder; every layer has the same base allocation.
    extra = ((layer + L - rotation) % L) < (count % L)
    quota = tl.minimum(E, count // L + extra.to(tl.int32))
    tl.store(quotas + layer, quota, layer < L)
    tl.store(counters + 0, count.to(tl.int64))
    # Advance the actual LRU clock once for this phase. Seeded rows are newer
    # than usage-zero empty slots, but never have a timestamp in the future;
    # the next ensure increments again, so usage==step protects only its hits.
    if count > 0:
        tl.store(clock, tl.load(clock) + 1)
    planned = tl.sum(tl.where(layer < L, quota, 0)).to(tl.int64)
    tl.store(counters + 1, tl.load(counters + 1) + planned)


@triton.jit
def _zero_histogram(hist, E: tl.constexpr, BLOCK_E: tl.constexpr):
    e = tl.arange(0, BLOCK_E)
    tl.store(hist + e, 0, e < E)


@triton.jit(do_not_specialize=["layer", "N"])
def _route_histogram(
    ids,
    hist,
    quotas,
    layer,
    N,
    WIDTH: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(quotas + layer) > 0:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offset = (i // WIDTH) * ROW_STRIDE + (i % WIDTH) * COL_STRIDE
        e = tl.load(ids + offset, i < N, 0).to(tl.int32)
        h = tl.histogram(e, BLOCK_E, mask=(i < N) & (e >= 0) & (e < E))
        bins = tl.arange(0, BLOCK_E)
        tl.atomic_add(hist + bins, h, bins < E)


@triton.jit(do_not_specialize=["layer", "exclude_lo", "exclude_hi"])
def _admit_empty(
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
    counters,
    layer,
    exclude_lo,
    exclude_hi,
    E: tl.constexpr,
    C: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Select at most quota empty victims; unlike ensure, shortage is success."""
    quota = tl.load(quotas + layer)
    n = tl.full((), 0, tl.int32)
    if quota > 0:
        e = tl.arange(0, BLOCK_E)
        h = tl.load(hist + e, e < E, 0).to(tl.int64)
        previous = tl.load(slots + layer * E + e, e < E, 0)
        candidate = (e < E) & (h > 0) & (previous < 0)
        # Unique ranks, low expert id breaks ties; old residents are excluded
        # before top-k selection, so their frequency cannot waste the quota.
        keys = tl.where(candidate, h * E + E - 1 - e, -1)
        ranked = tl.sort(keys, descending=True)
        expert = tl.where(ranked >= 0, E - 1 - ranked % E, 0).to(tl.int32)

        s = tl.arange(0, BLOCK_C)
        owner = tl.load(owners + s, s < C, -1)
        pin = tl.load(pinned + s, s < C, 1)
        allowed = (s >= 2 * E) & (s < C) & ((s < exclude_lo) | (s >= exclude_hi))
        free = allowed & (owner < 0) & (pin == 0)
        free_count = tl.sum(free.to(tl.int32))
        candidate_count = tl.sum(candidate.to(tl.int32))
        n = tl.minimum(quota, tl.minimum(free_count, candidate_count))

        free_rank = tl.cumsum(free.to(tl.int32)) - 1
        tl.store(dst + free_rank, s, free & (free_rank < n))
        # This CTA publishes its compact destination list before reading it.
        tl.debug_barrier()
        target = tl.load(dst + e, e < n, 0)
        gid = layer * E + expert
        tl.store(slots + gid, target, e < n)
        tl.store(owners + target, gid, e < n)
        tl.store(usage + target, tl.load(clock), e < n)
        # The staging rows occupy the first 2E slots of these same banks.
        tl.store(src + e, (layer % 2) * E + expert, e < n)
    tl.store(count_out, n.to(tl.int64))
    tl.store(counters + 2, tl.load(counters + 2) + n.to(tl.int64))
    tl.store(counters + 3, tl.load(counters + 3) + (n > 0).to(tl.int64))


class EmptySeedState:
    def __init__(self, cache, st):
        if st.pin_n != 0 or st.policy != "lru":
            raise ValueError("Empty-only seeding requires PIN_PER_LAYER=0 and POLICY=lru")
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if not cache._copy_fused_ok or _skip_fast_index_copy_enabled():
            raise ValueError("Empty-only seeding requires the working fused bank-copy path")
        self.E, self.L, self.C = cache.num_experts, cache.num_layers, cache.cache_size
        self.hist = torch.zeros((self.E,), dtype=torch.int32, device=cache.device)
        self.quotas = torch.zeros((self.L,), dtype=torch.int32, device=cache.device)
        self.src = torch.zeros((self.E,), dtype=torch.int32, device=cache.device)
        self.dst = torch.zeros((self.E,), dtype=torch.int32, device=cache.device)
        self.count = torch.zeros((1,), dtype=torch.int64, device=cache.device)
        # Last free capacity; cumulative planned rows, actual rows, seeded layers.
        self.counters = torch.zeros((4,), dtype=torch.int64, device=cache.device)
        self.chunks = 0
        self.exclude = (self.C, self.C)
        self.row_bytes = sum(bank[0].numel() * bank.element_size() for _, bank in cache.banks)
        self.route_tokens = int(os.environ.get("TIDEWAY_SHIFT_SEED_ROUTE_TOKENS", "0"))
        self.refresh_rows = int(os.environ.get("TIDEWAY_SHIFT_SEED_REFRESH_ROWS", "0"))
        if self.route_tokens not in (0, 128, 256) or self.refresh_rows not in (0, 4, 8):
            raise ValueError("Seed tail tokens must be 0/128/256; refresh rows must be 0/4/8")
        if self.refresh_rows and not self.route_tokens:
            raise ValueError("Current-layer refresh requires an explicit 128/256-token tail")
        self.phase_is_final = False
        self.refresh_counters = (
            torch.zeros((2,), dtype=torch.int64, device=cache.device) if self.refresh_rows else None
        )

    def warmup_metadata(self, cache):
        """Compile/load the new-C metadata kernels without touching serving state.

        Run after a bank rebuild and before measured requests, outside any graph
        capture or activation MemPool. Shapes/strides/types come from calibration
        already executed on this cache; no next-request length/route is consulted.
        Every pointer passed below refers to fresh scratch, never serving state.
        """
        if cache.cache_size != self.C:
            raise RuntimeError("Seed state and rebuilt cache geometry disagree")
        layouts = tuple(sorted(getattr(cache, "_seed_empty_route_layouts", ()), key=str))
        if not layouts:
            raise RuntimeError(
                "Seed metadata warmup requires routing metadata from prior calibration"
            )
        # Match a real observed arena lease where available. Every scratch slot
        # is pinned, so the quota stays zero regardless of these runtime bounds.
        exclude = getattr(cache, "_seed_empty_last_exclude", (2 * self.E, self.C))
        started = time.perf_counter()
        dev = cache.device
        owners = torch.full((self.C,), -1, dtype=cache.id_of_slot.dtype, device=dev)
        pinned = torch.ones((self.C,), dtype=torch.int32, device=dev)
        slots = torch.full((self.L, self.E), -1, dtype=cache.slot_for_id.dtype, device=dev)
        usage = torch.zeros((self.C,), dtype=cache.usage.dtype, device=dev)
        clock = torch.zeros((), dtype=cache.step.dtype, device=dev)
        hist = torch.zeros_like(self.hist)
        quotas = torch.zeros_like(self.quotas)
        src, dst = torch.zeros_like(self.src), torch.zeros_like(self.dst)
        count, counters = torch.zeros_like(self.count), torch.zeros_like(self.counters)
        _prepare_quotas[(1,)](
            owners,
            pinned,
            quotas,
            counters,
            clock,
            0,
            *exclude,
            self.E,
            self.L,
            self.C,
            triton.next_power_of_2(self.C),
            triton.next_power_of_2(self.L),
            num_warps=4,
        )
        _zero_histogram[(1,)](hist, self.E, triton.next_power_of_2(self.E), num_warps=4)
        route_scratch = []
        for layout in layouts:
            ids = _dummy_routing(layout, dev)
            route_scratch.append(ids)
            _, width, row_stride, col_stride, _ = layout
            _route_histogram[(1,)](
                ids,
                hist,
                quotas,
                0,
                0,
                width,
                row_stride,
                col_stride,
                self.E,
                triton.next_power_of_2(self.E),
                1024,
                num_warps=4,
            )
        _admit_empty[(1,)](
            hist,
            slots,
            owners,
            usage,
            clock,
            pinned,
            quotas,
            src,
            dst,
            count,
            counters,
            0,
            *exclude,
            self.E,
            self.C,
            triton.next_power_of_2(self.E),
            triton.next_power_of_2(self.C),
            num_warps=4,
        )
        refresh_scratch = None
        if getattr(self, "refresh_rows", 0):
            from freetoken.moe.phase_seed import (
                _advance_refresh_clock_if_full,
                _route_phase_histogram,
                _admit_empty_and_refresh,
            )

            refresh_scratch = torch.zeros_like(self.refresh_counters)
            _advance_refresh_clock_if_full[(1,)](
                quotas, clock, self.L, triton.next_power_of_2(self.L), num_warps=4
            )
            for layout, ids in zip(layouts, route_scratch):
                _, width, row_stride, col_stride, _ = layout
                _route_phase_histogram[(1,)](
                    ids,
                    hist,
                    quotas,
                    0,
                    0,
                    0,
                    width,
                    row_stride,
                    col_stride,
                    self.E,
                    triton.next_power_of_2(self.E),
                    1024,
                    num_warps=4,
                )
            _admit_empty_and_refresh[(1,)](
                hist,
                slots,
                owners,
                usage,
                clock,
                pinned,
                quotas,
                src,
                dst,
                count,
                counters,
                refresh_scratch,
                0,
                *exclude,
                0,
                self.E,
                self.C,
                triton.next_power_of_2(self.E),
                triton.next_power_of_2(self.C),
                num_warps=4,
            )
        if dev.type == "cuda":
            torch.cuda.current_stream(dev).synchronize()
        scratch = [owners, pinned, slots, usage, clock, hist, quotas, src, dst, count, counters]
        scratch_bytes = sum(t.numel() * t.element_size() for t in scratch)
        scratch_bytes += sum(t.untyped_storage().nbytes() for t in route_scratch)
        if refresh_scratch is not None:
            scratch_bytes += refresh_scratch.numel() * refresh_scratch.element_size()
        return dict(
            complete=True,
            cache_slots=self.C,
            metadata_only=True,
            model_forwards=0,
            expert_bytes_copied=0,
            scratch_bytes=scratch_bytes,
            warmup_excluded_interval=list(exclude),
            phase_refresh_warmed=refresh_scratch is not None,
            seconds=time.perf_counter() - started,
            routing_layouts=[
                dict(
                    dtype=str(x[0]),
                    width=x[1],
                    row_stride=x[2],
                    col_stride=x[3],
                    alignment_bytes=x[4],
                )
                for x in layouts
            ],
        )

    def prepare(self, cache, st):
        self.exclude = excluded_interval(st, self.C)
        cache._seed_empty_last_exclude = self.exclude
        self.phase_is_final = False
        if getattr(self, "route_tokens", 0):
            from freetoken.moe.phase_seed import current_phase_is_final

            self.phase_is_final = current_phase_is_final()
        _prepare_quotas[(1,)](
            cache.id_of_slot,
            st.pinned,
            self.quotas,
            self.counters,
            cache.step,
            self.chunks % self.L,
            *self.exclude,
            self.E,
            self.L,
            self.C,
            triton.next_power_of_2(self.C),
            triton.next_power_of_2(self.L),
            num_warps=4,
        )
        self.chunks += 1
        if self.phase_is_final and getattr(self, "refresh_rows", 0):
            from freetoken.moe.phase_seed import _advance_refresh_clock_if_full

            _advance_refresh_clock_if_full[(1,)](
                self.quotas, cache.step, self.L, triton.next_power_of_2(self.L), num_warps=4
            )

    def seed_layer(self, cache, st, layer_id, topk_ids):
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        buffer_id = layer_id % 2
        if (
            cache._prefill_buffer_layer[buffer_id] != layer_id
            or cache._prefill_buffer_released[buffer_id]
        ):
            raise RuntimeError("Empty seeding must precede release of this layer's staging bank")
        if topk_ids.numel() == 0:
            return
        # A view of already-computed routes, no new router work or GPU payload
        # allocation. Non-final chunks keep the original full-chunk policy.
        if self.phase_is_final and self.route_tokens:
            from freetoken.moe.phase_seed import tail_routing_view

            topk_ids = tail_routing_view(topk_ids, self.route_tokens, True)
        _, width, row_stride, col_stride, _ = observe_routing_layout(cache, topk_ids)
        _zero_histogram[(1,)](self.hist, self.E, triton.next_power_of_2(self.E), num_warps=4)
        refresh = self.refresh_rows if self.phase_is_final and topk_ids.ndim == 2 else 0
        if refresh:
            from freetoken.moe.phase_seed import _route_phase_histogram, _admit_empty_and_refresh

            _route_phase_histogram[(triton.cdiv(topk_ids.numel(), 1024),)](
                topk_ids,
                self.hist,
                self.quotas,
                layer_id,
                topk_ids.numel(),
                refresh,
                width,
                row_stride,
                col_stride,
                self.E,
                triton.next_power_of_2(self.E),
                1024,
                num_warps=4,
            )
            _admit_empty_and_refresh[(1,)](
                self.hist,
                cache.slot_for_id,
                cache.id_of_slot,
                cache.usage,
                cache.step,
                st.pinned,
                self.quotas,
                self.src,
                self.dst,
                self.count,
                self.counters,
                self.refresh_counters,
                layer_id,
                *self.exclude,
                refresh,
                self.E,
                self.C,
                triton.next_power_of_2(self.E),
                triton.next_power_of_2(self.C),
                num_warps=4,
            )
        else:
            _route_histogram[(triton.cdiv(topk_ids.numel(), 1024),)](
                topk_ids,
                self.hist,
                self.quotas,
                layer_id,
                topk_ids.numel(),
                width,
                row_stride,
                col_stride,
                self.E,
                triton.next_power_of_2(self.E),
                1024,
                num_warps=4,
            )
            _admit_empty[(1,)](
                self.hist,
                cache.slot_for_id,
                cache.id_of_slot,
                cache.usage,
                cache.step,
                st.pinned,
                self.quotas,
                self.src,
                self.dst,
                self.count,
                self.counters,
                layer_id,
                *self.exclude,
                self.E,
                self.C,
                triton.next_power_of_2(self.E),
                triton.next_power_of_2(self.C),
                num_warps=4,
            )
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs,
            cache._copy_dst_ptrs,
            cache._copy_feat_bytes,
            self.dst,
            self.src,
            self.count,
            blocks_per_bank=64,
        )

    def statistics(self):
        # Request-boundary reporting only; never called from per-layer serving.
        free, planned, actual, layers = self.counters.cpu().tolist()
        result = dict(
            seed_empty=True,
            seed_empty_free_last=free,
            seed_empty_planned_rows_total=planned,
            seed_empty_rows_total=actual,
            seed_empty_D2D_bytes_total=actual * self.row_bytes,
            seed_empty_layers_total=layers,
            seed_empty_chunks_total=self.chunks,
        )
        if getattr(self, "route_tokens", 0):
            result.update(
                seed_final_route_tokens=self.route_tokens,
                seed_current_layer_refresh_limit=self.refresh_rows,
            )
        if getattr(self, "refresh_counters", None) is not None:
            replaced, refreshed_layers = self.refresh_counters.cpu().tolist()
            result.update(
                seed_refresh_rows_total=replaced,
                seed_refresh_layers_total=refreshed_layers,
                seed_refresh_D2D_bytes_total=replaced * self.row_bytes,
            )
        return result
