"""Bounded expert transport for SSD-LLaMA and FlashMoE."""

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from queue import Queue
import time
import numpy as np
import torch

from .codec import Codec, HEADER
from .pack import PackReader, CheckpointRecordReader, padded
from .policies import GPUPlan


class ExpertPager:
    def __init__(
        self,
        cache,
        geo,
        pack,
        mode,
        host_rows,
        lanes=8,
        predictor=None,
        raw_store="pack",
        host_budget_bytes=None,
        exact_native_prefill=False,
    ):
        from freetoken.kernel.pinned import alloc_pinned_tensor

        self.cache, self.geo, self.mode = cache, geo, mode
        self.exact_native_prefill = bool(exact_native_prefill)
        if self.exact_native_prefill and (mode != "flashmoe" or geo.quant != "ds_fp4"):
            raise ValueError(
                "Exact native prefill is the declared DeepSeek FlashMoE adaptation only"
            )
        if mode == "flashmoe" and hasattr(predictor, "experts"):
            if (predictor.experts, predictor.layers) != (geo.experts, geo.layers):
                raise ValueError("Learned predictor geometry differs from the model")
        if raw_store not in ("pack", "checkpoint"):
            raise ValueError("Unknown raw store")
        self.reader = (
            CheckpointRecordReader(geo, pack, lanes)
            if raw_store == "checkpoint" and mode != "ssd_llama"
            else PackReader(pack, raw_comparison=mode != "ssd_llama")
        )
        if self.reader.manifest["row_bytes"] != geo.row_bytes:
            raise ValueError("Pack geometry differs from runtime")
        if len(self.reader.manifest["rows"]) != geo.total_experts:
            raise ValueError("Actual model execution requires the complete expert pack")
        self.codec = Codec()
        if mode == "ssd_llama":
            self.codec.enable_cuda()
        self.plan = GPUPlan(mode, geo.layers, geo.experts, cache.cache_size, predictor)
        # RAM is a byte budget. Compressed records use the largest aligned
        # compressed record as slot stride, not a wasted full raw expert row.
        # There is no uncharged host master or implicit filesystem cache.
        raw_stride = padded(geo.row_bytes + HEADER.size)
        self.stride = self.reader.max_size
        self.reader_workspace_bytes = getattr(self.reader, "workspace_bytes", 0)
        budget = (
            int(host_rows) * raw_stride if host_budget_bytes is None else int(host_budget_bytes)
        )
        self.capacity = min(
            geo.total_experts, (budget - self.reader_workspace_bytes) // self.stride
        )
        if self.capacity < min(geo.total_experts, geo.experts + lanes):
            raise ValueError("This adapter requires at least one layer plus staging in RAM")
        allocated = alloc_pinned_tensor(self.capacity * self.stride + 4096, dtype=torch.uint8)
        shift = (-allocated.data_ptr()) % 4096
        self.host = allocated[shift : shift + self.capacity * self.stride].view(
            self.capacity, self.stride
        )
        self.host.zero_()  # Charge physical resident RAM, not merely virtual capacity.
        self.where, self.lru = {}, OrderedDict()
        self.free = list(range(self.capacity - 1, -1, -1))
        self.parsed = {}
        self.pool = ThreadPoolExecutor(max_workers=lanes)
        self.lanes, self.free_lanes = [], Queue()
        for i in range(lanes):
            self.lanes.append(
                dict(
                    device=torch.empty(self.stride, dtype=torch.uint8, device="cuda"),
                    errors=torch.zeros(1, dtype=torch.int32, device="cuda"),
                    stream=torch.cuda.Stream(),
                )
            )
            self.free_lanes.put(i)
        self.stats = Counter()
        self.trace = None
        self.last_sources = {}

    def _host_slot(self, expert, protected):
        if expert in self.where:
            self.lru.move_to_end(expert)
            return self.where[expert], True
        if self.free:
            slot = self.free.pop()
        else:
            victim = next((e for e in self.lru if e not in protected), None)
            if victim is None:
                raise RuntimeError("All RAM slots leased by the active batch")
            slot = self.where.pop(victim)
            del self.lru[victim]
            self.parsed.pop(victim, None)
            self.stats["ram_evictions"] += 1
        self.where[expert] = slot
        self.lru[expert] = None
        return slot, False

    def ensure(self, layer, ids):
        began = time.perf_counter()
        mapping, loads, temporary = self.plan.plan(layer, ids)
        self.stats["policy_s"] += time.perf_counter() - began
        self.stats["gpu_requests"] += len(mapping)
        self.stats["gpu_hits"] += len(mapping) - len(loads)
        self.stats["temporary_admissions"] += len(temporary)
        protected = {layer * self.geo.experts + e for e in mapping}
        self.stats["gpu_evictions"] += sum(v is not None for _, _, v in loads)
        # GPU GEMMs from the caller must finish before any destination is reused.
        prior = torch.cuda.Event()
        prior.record(torch.cuda.current_stream())
        jobs = []
        for expert, gpu_slot, victim in loads:
            # Publication says a victim "may remain in RAM", but does not
            # specify active DtoH demotion. Retain an existing host duplicate;
            # otherwise discard the GPU-only copy. Do not add Tideway writeback.
            host_slot, hit = self._host_slot(expert, protected)
            jobs.append((expert, gpu_slot, host_slot, hit))

        def load(job):
            expert, gpu_slot, host_slot, hit = job
            lane_id = self.free_lanes.get()
            lane = self.lanes[lane_id]
            try:
                host = self.host[host_slot]
                if hit:
                    parsed = self.parsed[expert]
                else:
                    data = self.reader.read_into(expert, host)
                    parsed = self.codec.parse(data)
                    self.parsed[expert] = parsed
                outputs = [self.cache.bank_caches[b.name][gpu_slot] for b in self.geo.banks]
                stream = lane["stream"]
                begin = torch.cuda.Event(enable_timing=True)
                copied = torch.cuda.Event(enable_timing=True)
                done = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    stream.wait_event(prior)
                    lane["errors"].zero_()
                    begin.record(stream)
                    if parsed["raw"]:
                        at = parsed["offset"]
                        for dst in outputs:
                            view = dst.view(torch.uint8).reshape(-1)
                            view.copy_(host[at : at + view.numel()], non_blocking=True)
                            at += view.numel()
                        size = self.geo.row_bytes
                        copied.record(stream)
                    else:
                        size = parsed["offset"] + parsed["payload"]
                        lane["device"][:size].copy_(host[:size], non_blocking=True)
                        copied.record(stream)
                        self.codec.decode_cuda(
                            lane["device"], parsed, outputs, stream, lane["errors"]
                        )
                    done.record(stream)
                done.synchronize()
                if lane["errors"].item():
                    raise RuntimeError(f"Decoder integrity failure for expert {expert}")
                return dict(
                    ssd_bytes=0 if hit else self.reader.size(expert),
                    h2d_bytes=size,
                    h2d_ms=begin.elapsed_time(copied),
                    decode_ms=copied.elapsed_time(done),
                    ram_hits=int(hit),
                    ram_misses=int(not hit),
                )
            finally:
                self.free_lanes.put(lane_id)

        start = time.perf_counter()
        for counters in self.pool.map(load, jobs):
            self.stats.update(counters)
        self.stats["delivery_s"] += time.perf_counter() - start
        for expert, slot, _, _ in jobs:
            self.last_sources[slot] = expert
        return mapping

    def native_full_prefill(self, layer, hidden, weights, ids):
        """Preserve DeepSeek's original FP4 grouped-prefill arithmetic.

        FlashMoE's published contribution is its learned cache replacement.  It
        does not require changing the model's expert kernel.  Expert-major
        regrouping changes the FP4 grouped-GEMM tile shape and introduces small
        BF16 differences, so for DeepSeek prefill we materialize one complete
        layer in a charged scratch region and invoke the native kernel with the
        original [tokens, top_k] routing matrix. Decode still uses the learned
        FlashMoE plan.
        """
        began = time.perf_counter()
        E = self.geo.experts
        if self.cache.cache_size < E:
            raise RuntimeError("Native prefill scratch does not fit one expert layer")
        # Slots [0,E) are a transient one-layer scratch. Invalidate any learned
        # decode-cache entries they overwrite; slots [E,cache_size) remain hot.
        victims = 0
        for slot in range(E):
            victim = self.plan.reverse[slot]
            if victim >= 0:
                self.plan.where.pop(victim, None)
                self.plan.lru.pop(victim, None)
                self.plan.reverse[slot] = -1
                victims += 1
        self.stats["gpu_evictions"] += victims
        prior = torch.cuda.Event()
        prior.record(torch.cuda.current_stream())
        protected = {layer.layer_id * E + e for e in range(E)}
        jobs = []
        for local in range(E):
            expert = layer.layer_id * E + local
            host_slot, hit = self._host_slot(expert, protected)
            jobs.append((expert, local, host_slot, hit))

        def load(job):
            expert, gpu_slot, host_slot, hit = job
            lane_id = self.free_lanes.get()
            lane = self.lanes[lane_id]
            try:
                host = self.host[host_slot]
                if hit:
                    parsed = self.parsed[expert]
                else:
                    data = self.reader.read_into(expert, host)
                    parsed = self.codec.parse(data)
                    self.parsed[expert] = parsed
                outputs = [self.cache.bank_caches[b.name][gpu_slot] for b in self.geo.banks]
                stream = lane["stream"]
                begin = torch.cuda.Event(enable_timing=True)
                copied = torch.cuda.Event(enable_timing=True)
                done = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    stream.wait_event(prior)
                    lane["errors"].zero_()
                    begin.record(stream)
                    if parsed["raw"]:
                        at = parsed["offset"]
                        for dst in outputs:
                            view = dst.view(torch.uint8).reshape(-1)
                            view.copy_(host[at : at + view.numel()], non_blocking=True)
                            at += view.numel()
                        size = self.geo.row_bytes
                        copied.record(stream)
                    else:
                        size = parsed["offset"] + parsed["payload"]
                        lane["device"][:size].copy_(host[:size], non_blocking=True)
                        copied.record(stream)
                        self.codec.decode_cuda(
                            lane["device"], parsed, outputs, stream, lane["errors"]
                        )
                    done.record(stream)
                done.synchronize()
                if lane["errors"].item():
                    raise RuntimeError(f"Decoder integrity failure for expert {expert}")
                return dict(
                    ssd_bytes=0 if hit else self.reader.size(expert),
                    h2d_bytes=size,
                    h2d_ms=begin.elapsed_time(copied),
                    decode_ms=copied.elapsed_time(done),
                    ram_hits=int(hit),
                    ram_misses=int(not hit),
                )
            finally:
                self.free_lanes.put(lane_id)

        start = time.perf_counter()
        for counters in self.pool.map(load, jobs):
            self.stats.update(counters)
        self.stats["delivery_s"] += time.perf_counter() - start
        for expert, slot, _, _ in jobs:
            self.last_sources[slot] = expert
        self.stats["gpu_requests"] += E
        self.stats["compute_groups"] += 1
        self.stats["native_full_prefill_layers"] += 1
        self.stats["policy_s"] += time.perf_counter() - began - (time.perf_counter() - start)
        views = tuple(self.cache.bank_caches[b.name][:E] for b in self.geo.banks)
        return layer._expert_gemm(
            self.cache, hidden, weights, ids, views=views, n=E, alphas=None, is_prefill=True
        )

    def routed(self, layer, hidden, weights, ids, is_prefill):
        keys = [
            "ssd_bytes",
            "h2d_bytes",
            "gpu_hits",
            "gpu_requests",
            "ram_hits",
            "ram_misses",
            "gpu_evictions",
            "compute_groups",
        ]
        before = {k: self.stats[k] for k in keys}
        result = self._routed_impl(layer, hidden, weights, ids, is_prefill)
        phase = "prefill" if is_prefill else "decode"
        for key in keys:
            self.stats[phase + "_" + key] += self.stats[key] - before[key]
        return result

    def _routed_impl(self, layer, hidden, weights, ids, is_prefill):
        # Host-visible routing belongs to these cache algorithms; its overhead
        # remains inside measured request time. No access to future routes.
        local = ids.detach().cpu().numpy().astype(np.int64)
        base = 0 if self.mode == "ssd_llama" else layer.layer_id * self.plan.cap
        count = self.cache.cache_size if self.mode == "ssd_llama" else self.plan.cap
        views = tuple(self.cache.bank_caches[b.name][base : base + count] for b in self.geo.banks)
        self.plan.history.observe(layer.layer_id, local)
        if self.trace is not None:
            self.trace.append(
                dict(
                    layer=layer.layer_id,
                    phase="prefill" if is_prefill else "decode",
                    ids=local.tolist(),
                )
            )
        if self.exact_native_prefill and is_prefill:
            return self.native_full_prefill(layer, hidden, weights, ids)
        unique = np.unique(local)
        limit = self.geo.experts if self.mode == "ssd_llama" else self.plan.cap
        if len(unique) <= limit:
            mapping = self.ensure(layer.layer_id, local)
            mapped = torch.tensor(
                [[mapping[int(e)] - base for e in row] for row in local],
                dtype=ids.dtype,
                device=ids.device,
            )
            self.stats["compute_groups"] += 1
            return layer._expert_gemm(
                self.cache,
                hidden,
                weights,
                mapped,
                views=views,
                n=count if is_prefill else None,
                alphas=None,
                is_prefill=is_prefill,
            )
        # FlashMoE: expert-major prefill; no expert is delivered twice for this
        # iteration. Scatter each result to its ORIGINAL top-k position, then
        # use the original model reduction. Atomic index_add changes floating
        # reduction order and caused a long-prompt greedy-output divergence.
        route_outputs = torch.empty(
            (*ids.shape, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device
        )
        for begin in range(0, len(unique), limit):
            selected = unique[begin : begin + limit]
            mapping = self.ensure(layer.layer_id, selected)
            token, route = np.nonzero(np.isin(local, selected))
            token_dev = torch.tensor(token, device=hidden.device)
            route_dev = torch.tensor(route, device=hidden.device)
            mapped = torch.tensor(
                [[mapping[int(local[t, r])] - base] for t, r in zip(token, route)],
                dtype=ids.dtype,
                device=ids.device,
            )
            routed_hidden = hidden[token_dev].contiguous()
            routed_weights = weights[token_dev, route_dev, None].contiguous()
            partial = layer._expert_gemm(
                self.cache,
                routed_hidden,
                routed_weights,
                mapped,
                views=views,
                n=count,
                alphas=None,
                is_prefill=True,
            )
            route_outputs[token_dev, route_dev] = partial
            self.stats["compute_groups"] += 1
        if self.cache.quant_format == "ds_fp4":
            # Native DeepSeek FP4 kernels reduce BF16 route outputs through
            # torch.sum. Preserve that model-specific reduction exactly.
            return route_outputs.sum(dim=1)
        from freetoken.kernel import moe_sum_reduce_triton

        out = torch.empty_like(hidden)
        moe_sum_reduce_triton(route_outputs, out)
        return out

    def reset_stats(self):
        self.stats.clear()

    def reset_cold(self):
        torch.cuda.synchronize()
        self.where.clear()
        self.lru.clear()
        self.parsed.clear()
        self.last_sources.clear()
        self.free = list(range(self.capacity - 1, -1, -1))
        self.plan = GPUPlan(
            self.mode, self.geo.layers, self.geo.experts, self.cache.cache_size, self.plan.predictor
        )
        self.reset_stats()

    def snapshot(self):
        both = len(set(self.where) & set(self.plan.where))
        return dict(
            self.stats,
            ram_experts=len(self.where),
            gpu_experts=len(self.plan.where),
            duplicate_experts=both,
            unique_experts=len(set(self.where) | set(self.plan.where)),
            host_allocated_bytes=self.host.numel(),
            source_bounce_budget_bytes=self.reader_workspace_bytes,
            host_total_allocated_bytes=self.host.numel() + self.reader_workspace_bytes,
            compressed_staging_gpu_bytes=sum(lane["device"].numel() for lane in self.lanes),
            d2h_bytes=0,
            ssd_demote_gpu_only=False,
        )

    def audit_gpu(self):
        torch.cuda.synchronize()
        checked = 0
        # Full-byte SHA against independent source proof performed during packing.
        for slot, expert in self.last_sources.items():
            digest = hashlib.sha256()
            for bank in self.geo.banks:
                digest.update(
                    self.cache.bank_caches[bank.name][slot].view(torch.uint8).cpu().numpy()
                )
            if digest.hexdigest() != self.reader.manifest["rows"][str(expert)]["original_sha256"]:
                raise RuntimeError(f"GPU payload mismatch slot={slot} expert={expert}")
            checked += 1
        return dict(ok=True, checked_slots=checked)

    def close(self):
        self.pool.shutdown(wait=True)
        torch.cuda.synchronize()
        self.reader.close()


def install_pager(cache, geo, **kwargs):
    """Only redirects routed MoE movement; leaves original computational kernels."""
    from .._backend import require_pristine_backend
    from .._session import BaselineSession

    require_pristine_backend()
    from freetoken.layers.moe import OffloadMoELayer

    # Installed SGL alignment faults for the global 1800-slot view (independently
    # reproduced before GEMM). Use the engine's ORIGINAL Triton fallback for
    # large slot counts; no kernel source is changed or imported from Tideway.
    import freetoken.moe.fused as fused

    native_align = fused.moe_align_block_size

    def compatible_align(ids, block, experts):
        if experts >= 1024:
            from freetoken.kernel.triton.moe_align import moe_align_block_size

            return moe_align_block_size(ids, block, experts)
        return native_align(ids, block, experts)

    classes = [OffloadMoELayer]
    if geo.quant == "ds_fp4":
        from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer

        classes.append(DSV4OffloadMoELayer)
    session = BaselineSession()
    try:
        session.bind(fused, "moe_align_block_size", compatible_align)
        manager = session.manager = ExpertPager(cache, geo, **kwargs)
        for klass in classes:

            def prefill(layer, hidden, weights, ids):
                return manager.routed(layer, hidden, weights, ids, True)

            def decode(layer, hidden, weights, ids):
                return manager.routed(layer, hidden, weights, ids, False)

            session.bind(klass, "_prefill_routed", prefill)
            session.bind(klass, "_decode_routed", decode)
    except BaseException as error:
        try:
            session.uninstall()
        except BaseException as cleanup_error:
            error.add_note(f"Baseline cleanup failed: {cleanup_error!r}")
        raise
    return session
