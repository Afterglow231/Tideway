"""Opt-in native decode graphs split at the bounded expert pager.

The native model runs once during a real warmup decode. At each ensure boundary
we end capture, replay that segment exactly once, run the unchanged CPU pager,
and resume capture. Capture itself executes no kernels: KV/GDN state therefore
advances once, not twice. Subsequent tokens replay the same segments around the
same pager, using retained routing tensors and fixed expert banks.

This changes neither arithmetic nor cache policy. It supports one CUDA stream,
one GPU and text-only decode BS=1 for the Qwen/Gemma BF16 bounded manager. It
requires a separate numerical GPU gate before performance claims are made.
"""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
import time


def _layout(t):
    return (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


def _storage(t):
    return (int(t.data_ptr()), _layout(t))


def _owned_ensure(binding, manager):
    """Permit transparent telemetry wrappers around this exact pager owner."""
    seen = set()
    while binding is not None and id(binding) not in seen:
        seen.add(id(binding))
        if getattr(binding, "__self__", None) is manager and getattr(
            binding, "__func__", None
        ) is getattr(manager.ensure, "__func__", None):
            return True
        binding = getattr(binding, "__wrapped__", None)
    return False


@dataclass
class _Boundary:
    layer: int
    ids: object
    signature: tuple


class PagedDecodeGraph:
    """GraphRunner-compatible controller, installed before explicit warmup."""

    def __init__(self, engine, manager):
        import torch
        from freetoken.engine.graph import GraphCaptureBuffer
        from freetoken.engine.engine import _page_table_width
        from freetoken.moe.offload_cache import iter_offload_moe_layers
        from .runtime import TidewayOffload

        if not isinstance(manager, TidewayOffload) or manager.geo.quant != "bf16":
            raise TypeError("paged decode graphs require the bounded BF16 Tideway manager")
        if getattr(engine.device, "type", None) != "cuda":
            raise ValueError("paged decode graphs require CUDA")
        if getattr(getattr(engine.config, "tp_info", None), "size", 1) != 1:
            raise ValueError("paged decode graphs require one GPU")
        model_name = type(engine.model).__name__.lower()
        if "qwen" not in model_name and "gemma4" not in model_name:
            raise TypeError("paged decode graphs are limited to Qwen/Gemma4")
        if manager.writeback_ahead:
            raise ValueError("paged decode graphs require writeback_ahead=False")
        if getattr(engine, "cpu_moe_executor", None) is not None:
            raise ValueError("paged decode graphs cannot compose CPU MoE execution")
        if engine.moe_offload_cache is not manager.cache:
            raise RuntimeError("engine and bounded manager have different expert caches")
        native = engine.graph_runner
        if getattr(native, "graph_bs_list", []) or getattr(native, "graph_map", {}):
            raise ValueError("disable native whole-model graphs before installing paged graphs")
        for name in (
            "paged_decode_graph",
            "segmented_decode",
            "attention_decode_graph",
            "ffn_front_graph",
            "ffn_back_graph",
        ):
            if getattr(manager, name, None) is not None:
                raise ValueError("conflicting graph controller: " + name)
        if not _owned_ensure(manager.cache.ensure_experts, manager):
            raise RuntimeError("ensure_experts is not owned by the supplied bounded pager")
        layers = list(iter_offload_moe_layers(engine.model))
        self.expected_layers = tuple(int(layer.layer_id) for layer in layers)
        if (
            not layers
            or len(set(self.expected_layers)) != len(layers)
            or any(layer.offload_cache is not manager.cache for layer in layers)
        ):
            raise RuntimeError("invalid MoE boundary ownership or layer sequence")
        if engine.stream == torch.cuda.default_stream(engine.device):
            raise ValueError("paged graph capture requires the original nondefault engine stream")

        self.torch, self.engine, self.manager = torch, engine, manager
        self.cache, self.native_runner = manager.cache, native
        self.native_ensure = self.cache.ensure_experts
        self.native_copy_missing = self.cache.copy_missing
        self.model, self.attn, self.stream = engine.model, engine.attn_backend, engine.stream
        self.device = engine.device
        self.state, self.failure = "installed", None
        self.segments, self.boundaries = [], []
        self.failed_graphs = []
        self.pool = self.active_graph = self.capture_metadata = None
        self.inside_forward = False
        self.capture_seconds = 0.0
        self.memory_before = self._memory()
        self.memory_sealed = None
        self.stats = dict(
            captures=0,
            capture_steps=0,
            warmup_steps=0,
            measured_steps=0,
            warmup_replays=0,
            measured_replays=0,
            pager_calls=0,
            prefill_bypasses=0,
            failures=0,
        )
        self.reset_stats()
        # Standard GraphRunner fields used by the scheduler and reset audits.
        self.graph_bs_list, self.max_graph_bs = [1], 1
        self.graph_map = {}  # Segments are not whole-model graph_map entries.
        self.dummy_req = native.dummy_req
        self.buffer = None
        self._installed = False
        self._attn_initialized = False
        try:
            backend = getattr(self.attn, "decode_backend", self.attn)
            if getattr(backend, "capture", None) is not None:
                raise RuntimeError(
                    "attention capture buffers are already owned by another controller"
                )
            # Own partial initialization too, so an allocation failure can be
            # rolled back without leaving capture metadata attached to eager.
            self._attn_initialized = True
            self.attn.init_capture_graph(
                max_seq_len=_page_table_width(engine.max_seq_len, engine.config.page_size),
                bs_list=[1],
            )
            self.buffer = GraphCaptureBuffer.init(
                1, engine.config.model_config.vocab_size, self.device
            )
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            self.bank_signature = tuple(_storage(t) for t in self.cache.bank_views())
            self.model_signature = self._model_signature()
            self.runtime_signature = self._runtime_signature()
            self.ensure_binding = self._boundary
            self.cache.ensure_experts = self.ensure_binding
            engine.graph_runner = self
            manager.paged_decode_graph = self
            self._installed = True
        except BaseException:
            if self._attn_initialized:
                self.attn.reset_capture()
            raise

    def _memory(self):
        t = self.torch
        return dict(
            allocated=int(t.cuda.memory_allocated(self.device)),
            reserved=int(t.cuda.memory_reserved(self.device)),
            free=int(t.cuda.mem_get_info(self.device)[0]),
        )

    def _model_signature(self):
        return tuple((name, _storage(t)) for name, t in self.model.state_dict().items())

    def _runtime_signature(self):
        """Metadata/storage only; never read device values or synchronize."""
        t = self.torch
        roots = [
            ("kv", self.engine.ctx.kv_cache),
            ("linear", getattr(self.engine.ctx, "linear_state_pool", None)),
        ]
        backend = getattr(self.attn, "decode_backend", self.attn)
        roots.extend([("capture", getattr(backend, "capture", None)), ("buffer", self.buffer)])
        result = []
        for prefix, obj in roots:
            if obj is None:
                result.append((prefix, None))
                continue
            result.append((prefix, id(obj)))
            for name, value in vars(obj).items():
                # The SWA allocator deliberately slices/replaces its free-list
                # on each request. Graphs bind payloads and the translation map,
                # never that host-controlled allocator bookkeeping tensor.
                if prefix == "kv" and name not in (
                    "_kv_buffer",
                    "full_to_swa_index_mapping",
                    "full_kv_pool",
                    "swa_kv_pool",
                ):
                    continue
                if isinstance(value, t.Tensor):
                    result.append((prefix + "." + name, _storage(value)))
                elif name in ("full_kv_pool", "swa_kv_pool") and value is not None:
                    # Hybrid-SWA pools may own one tensor or a nested cache object.
                    for subname, subvalue in (
                        vars(value).items() if hasattr(value, "__dict__") else ()
                    ):
                        if isinstance(subvalue, t.Tensor):
                            result.append((prefix + "." + name + "." + subname, _storage(subvalue)))
        return tuple(result)

    def _fail(self, error):
        self.failure = f"{type(error).__name__}: {error}"
        self.state = "failed"
        self.stats["failures"] += 1

    def reset_stats(self):
        self.request_steps = self.request_replays = self.request_pager_calls = 0

    @contextmanager
    def warmup(self):
        if self.state != "installed":
            raise RuntimeError("paged graph warmup begins exactly once")
        self.state = "warming"
        try:
            yield self
            self.seal()
        except BaseException as error:
            self._fail(error)
            raise

    def seal(self):
        if self.state != "warming" or not self.segments or self.inside_forward:
            raise RuntimeError("paged graph seal requires completed real decode warmup")
        self._check_graph_structure()
        self._check_static()
        self.stream.synchronize()
        self.memory_sealed = self._memory()
        self.state = "sealed"

    def can_use_cuda_graph(self, batch):
        if self.state in ("failed", "closed"):
            raise RuntimeError(f"paged decode graphs are {self.state}: {self.failure}")
        if batch.is_prefill:
            self.stats["prefill_bypasses"] += 1
            return False
        if not batch.is_decode or batch.size != 1 or batch.padded_size != 1:
            raise ValueError("paged decode graphs require explicit decode BS=1")
        if self.state not in ("warming", "sealed"):
            raise RuntimeError("enter explicit paged graph warmup before decode")
        return True

    def pad_batch(self, batch):
        if batch.is_decode and batch.size != 1:
            raise ValueError("paged decode graphs require decode BS=1")
        self.native_runner.pad_batch(batch)

    def _check_static(self):
        if (
            self.engine.graph_runner is not self
            or self.engine.model is not self.model
            or self.engine.attn_backend is not self.attn
            or self.engine.stream != self.stream
            or self.engine.moe_offload_cache is not self.cache
            or self.manager.cache is not self.cache
            or self.cache.ensure_experts is not self.ensure_binding
            or self.cache.copy_missing is not self.native_copy_missing
        ):
            raise RuntimeError("paged graph runtime ownership changed")
        if self.manager.writeback_ahead:
            raise RuntimeError("paged graph writeback stream policy changed")
        if tuple(_storage(t) for t in self.cache.bank_views()) != self.bank_signature:
            raise RuntimeError("paged graph expert bank address/layout changed")
        if self._runtime_signature() != self.runtime_signature:
            raise RuntimeError("paged graph KV/GDN/metadata buffer storage changed")
        if self._model_signature() != self.model_signature:
            raise RuntimeError("paged graph model weight address/layout changed")
        for boundary in self.boundaries:
            if _storage(boundary.ids) != boundary.signature:
                raise RuntimeError("paged graph retained routing tensor storage changed")

    def _check_graph_structure(self):
        actual = tuple(b.layer for b in self.boundaries)
        if actual != self.expected_layers or len(self.segments) != len(actual) + 1:
            raise RuntimeError(f"incomplete paged graph boundary sequence: {actual}")

    def _begin_segment(self):
        if self.active_graph is not None:
            raise RuntimeError("nested paged graph segment capture")
        graph = self.torch.cuda.CUDAGraph()
        self.active_graph = graph
        graph.capture_begin(pool=self.pool, capture_error_mode="global")

    def _end_segment(self):
        graph = self.active_graph
        if graph is None:
            raise RuntimeError("no paged graph segment to finish")
        graph.capture_end()
        self.active_graph = None
        self.segments.append(graph)
        if self.pool is None:
            self.pool = graph.pool()
        self.stats["captures"] += 1
        # CUDA capture records commands but does not run them. In particular,
        # recurrent/KV writes execute HERE once, not again via an eager reference.
        graph.replay()
        self.stats["warmup_replays"] += 1

    def _boundary(self, layer, expert_ids):
        if not self.inside_forward:
            # Prefill retains the original installed pager and its native hooks.
            if self.state in ("failed", "closed"):
                raise RuntimeError("pager invoked after paged graph failure/close")
            return self.native_ensure(layer, expert_ids)
        if self.state != "warming" or self.active_graph is None:
            raise RuntimeError("unexpected native ensure while replaying paged graphs")
        index = len(self.boundaries)
        if index >= len(self.expected_layers) or int(layer) != self.expected_layers[index]:
            raise RuntimeError("native MoE ensure boundary repeated or reordered")
        if (
            expert_ids.device != self.device
            or expert_ids.ndim != 2
            or expert_ids.shape[0] != 1
            or not expert_ids.is_contiguous()
            or expert_ids.dtype not in (self.torch.int32, self.torch.int64)
        ):
            raise RuntimeError("invalid native routing tensor at paged graph boundary")
        # Keep routing tensors alive across all graphs, including shared-pool
        # captures. Native forward stack retains other live cross-boundary values.
        self.boundaries.append(_Boundary(int(layer), expert_ids, _storage(expert_ids)))
        self._end_segment()
        self.native_ensure(layer, expert_ids)
        self.stats["pager_calls"] += 1
        self._begin_segment()

    def _prepare_batch(self, batch, first):
        source = copy(batch)
        self.buffer.copy_from(source)
        if first:
            self.attn.prepare_for_capture(batch)
            self.capture_metadata = batch.attn_metadata
        # Copy this actual request's metadata into the native static capture
        # buffers. This is outside capture and must precede stateful execution.
        self.attn.prepare_for_replay(source)
        batch.attn_metadata = self.capture_metadata
        self.buffer.set_batch(batch)

    def _capture_forward(self):
        started = time.perf_counter()
        self._begin_segment()
        logits = self.model.forward()
        self.buffer.logits.copy_(logits)
        self._end_segment()
        self._check_graph_structure()
        self.stats["capture_steps"] += 1
        self.capture_seconds += time.perf_counter() - started

    def replay(self, batch):
        if not self.can_use_cuda_graph(batch):
            raise RuntimeError("prefill passed to paged graph replay")
        if self.inside_forward:
            raise RuntimeError("reentrant paged decode")
        if self.torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("paged decode must use its original engine stream")
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paged decode cannot run inside another CUDA graph")
        if getattr(batch, "mm_embeds", None) is not None:
            raise ValueError("paged decode does not support multimodal decode inputs")
        if batch.input_ids.numel() != 1 or batch.positions.numel() != 1:
            raise ValueError("paged decode requires one token")
        if any(
            req.extend_len != 1 or req.device_len > self.engine.max_seq_len for req in batch.reqs
        ):
            raise ValueError("paged decode request exceeds captured context geometry")
        saved = {
            name: getattr(batch, name, None)
            for name in (
                "input_ids",
                "out_loc",
                "positions",
                "linear_table_idx",
                "fla_metadata",
                "attn_metadata",
            )
        }
        try:
            self._check_static()
            first = not self.segments
            if first and self.state != "warming":
                raise RuntimeError("capture during measurement is forbidden")
            self._prepare_batch(batch, first)
            self.inside_forward = True
            if first:
                self._capture_forward()
            else:
                self._check_graph_structure()
                for index, graph in enumerate(self.segments):
                    graph.replay()
                    key = "warmup_replays" if self.state == "warming" else "measured_replays"
                    self.stats[key] += 1
                    if self.state == "sealed":
                        self.request_replays += 1
                    if index < len(self.boundaries):
                        boundary = self.boundaries[index]
                        self.native_ensure(boundary.layer, boundary.ids)
                        self.stats["pager_calls"] += 1
                        if self.state == "sealed":
                            self.request_pager_calls += 1
            key = "warmup_steps" if self.state == "warming" else "measured_steps"
            self.stats[key] += 1
            if self.state == "sealed":
                self.request_steps += 1
            return self.buffer.logits[: batch.size]
        except BaseException as error:
            # No eager retry: some previous segments may already have advanced
            # GDN/KV state. Retain graph/pool/tensor owners until a successful drain.
            if self.active_graph is not None:
                self.failed_graphs.append(self.active_graph)
                try:
                    self.active_graph.capture_end()
                except BaseException as cleanup:
                    error.add_note(f"failed segment capture cleanup: {cleanup!r}")
                self.active_graph = None
            self._fail(error)
            raise
        finally:
            self.inside_forward = False
            for name, value in saved.items():
                setattr(batch, name, value)

    def snapshot(self):
        return dict(
            enabled=True,
            state=self.state,
            failure=self.failure,
            scope="bounded BF16 Qwen/Gemma text decode BS=1; native Prefill bypass",
            boundary_layers=[b.layer for b in self.boundaries],
            expected_boundary_layers=list(self.expected_layers),
            segment_count=len(self.segments),
            boundary_count=len(self.boundaries),
            capture_seconds=self.capture_seconds,
            counters=dict(self.stats),
            request_steps=self.request_steps,
            request_replays=self.request_replays,
            request_pager_calls=self.request_pager_calls,
            memory_before=self.memory_before,
            memory_sealed=self.memory_sealed,
            static_input_bytes=0
            if self.buffer is None
            else sum(int(t.numel() * t.element_size()) for t in vars(self.buffer).values()),
            retained_routing_bytes=sum(
                int(b.ids.numel() * b.ids.element_size()) for b in self.boundaries
            ),
            extra_expert_payload_bytes=0,
            native_arithmetic=True,
            capture_first_real_step_executes_once=True,
            ensures_captured=False,
            shared_pool=True,
            strict_segment_order=True,
            captures_during_measurement=False,
            graph_pool_rule="serial replay in original capture order; routing tensors retained",
            validation="requires independent full-logit/token/state numerical gate",
        )

    def close(self):
        if self.state == "closed":
            return
        if self.inside_forward or self.active_graph is not None:
            raise RuntimeError("cannot close an active paged graph step")
        try:
            self.stream.synchronize()
        except BaseException as error:
            self._fail(error)
            raise
        if self.cache.ensure_experts is self.ensure_binding:
            self.cache.ensure_experts = self.native_ensure
        if self.engine.graph_runner is self:
            self.engine.graph_runner = self.native_runner
        if getattr(self.manager, "paged_decode_graph", None) is self:
            delattr(self.manager, "paged_decode_graph")
        self.segments.clear()
        self.boundaries.clear()
        self.failed_graphs.clear()
        self.pool = self.buffer = self.capture_metadata = None
        if self._attn_initialized:
            self.attn.reset_capture()
            self._attn_initialized = False
        self.state = "closed"

    def destroy_cuda_graphs(self):
        self.close()


def enable_paged_decode_graph(engine, manager):
    """Install before common warmup; capture only inside ``controller.warmup()``."""
    return PagedDecodeGraph(engine, manager)
