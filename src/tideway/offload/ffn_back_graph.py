"""DS-FP4 decode back graphs after expert paging.

Preserve routed-expert arithmetic, TP1 reduction, shared addition and HC post.
CPU policy, SSD reads, expert DMA and slot-map upload stay outside capture.
Graphs bind the FFN-front controller's stable output tensors.
"""

from dataclasses import dataclass
import time

import torch

from .attention_graph import _layout, _storage, _nbytes, _byte_equal


def _owner_ensure(binding, manager):
    """Allow transparent telemetry @wraps, never a different pager owner."""
    seen = set()
    while binding is not None and id(binding) not in seen:
        seen.add(id(binding))
        if getattr(binding, "__self__", None) is manager and getattr(
            binding, "__func__", None
        ) is getattr(manager.ensure, "__func__", None):
            return True
        binding = getattr(binding, "__wrapped__", None)
    return False


def _native_ffn_back(block, cache, inputs, banks):
    norm_x, weights, slots, shared, post, comb, residual = inputs
    layer = block.ffn.experts
    routed = layer._expert_gemm(
        cache, norm_x, weights, slots, views=banks, n=None, alphas=None, is_prefill=False
    )
    routed = layer._maybe_all_reduce(routed)
    combined = (routed + shared).view(*residual.shape[:2], block.dim)
    return block.hc_post(combined, residual, post, comb)


@dataclass
class _Entry:
    graph: object
    inputs: tuple
    banks: tuple
    output: object
    signature: tuple
    input_pointers: tuple


class FFNBackGraph:
    def __init__(self, manager, engine, *, shared_pool=True):
        from .runtime import TidewayOffload

        front = getattr(manager, "ffn_front_graph", None)
        if not isinstance(manager, TidewayOffload) or manager.geo.quant != "ds_fp4":
            raise TypeError("FFN back graph requires bounded DeepSeek native FP4 Tideway")
        if (
            front is None
            or front.manager is not manager
            or front.engine is not engine
            or front.state != "installed"
            or front.entries
            or front.sequence is not None
        ):
            raise ValueError("install FFN back after its front controller, before any warmup")
        if manager.writeback_ahead:
            raise ValueError("initial FFN back candidate requires the qualified non-ahead pager")
        if any(
            getattr(manager, name, None) is not None
            for name in ("pure_decode_graphs", "compressor_decode_graph", "decode_compute_graph")
        ):
            raise ValueError("FFN back cannot compose overlapping compute wrappers")
        if getattr(engine.graph_runner, "graph_bs_list", []):
            raise ValueError("bounded FFN back cannot compose a full-model Graph")
        if getattr(engine.config.tp_info, "size", 1) != 1:
            raise ValueError("FFN back currently requires a single GPU")
        if type(shared_pool) is not bool:
            raise ValueError("shared_pool must be boolean")
        cache = manager.cache
        if (
            cache is not engine.moe_offload_cache
            or cache.quant_format != "ds_fp4"
            or cache.decode_target != "gpu"
        ):
            raise ValueError("FFN back requires the exact GPU cache owned by the bounded pager")
        if not _owner_ensure(cache.ensure_experts, manager):
            raise ValueError("bounded ensure binding differs from the owner")
        if tuple(manager.bank_names) != (
            "gate_up_packed",
            "gate_up_scale",
            "down_packed",
            "down_scale",
        ):
            raise ValueError("FFN back requires the complete native four-bank FP4 layout")
        self.manager, self.engine, self.front, self.cache = manager, engine, front, cache
        self.device = engine.device
        self.blocks, self.names = dict(front.blocks), dict(front.names)
        self.owned_ids = frozenset(self.blocks)
        self.shared_pool, self.pool, self.stream = shared_pool, None, None
        self.entries, self.capture_records, self.output_ranges = {}, {}, []
        self.sequence = None
        self.step_keys, self.step_seen, self.step_batch = [], set(), None
        self.step_active = self.inside_back = False
        self.state, self.failure = "installed", None
        self.memory_begin = self.memory_sealed = None
        self.warmup_started = self.warmup_seconds = None
        self.stats = dict(
            captures=0,
            native_reference_calls=0,
            output_byte_checks=0,
            immutable_input_byte_checks=0,
            warmup_replays=0,
            measured_replays=0,
            warmup_steps=0,
            measured_steps=0,
            ensures_outside_graph=0,
            failures=0,
        )
        self.banks = tuple(cache.bank_views())
        self.static_signature = self._static_signature()
        self.reset_stats()

    def reset_stats(self):
        self.request_replays = self.request_steps = self.request_ensures = 0

    def _fail(self, error):
        if self.state != "failed":
            self.failure = f"{type(error).__name__}: {error}"
            self.stats["failures"] += 1
        self.state, self.step_active = "failed", False

    def _memory(self):
        return dict(
            allocated=int(torch.cuda.memory_allocated(self.device)),
            reserved=int(torch.cuda.memory_reserved(self.device)),
            free=int(torch.cuda.mem_get_info(self.device)[0]),
            process_peak_allocated=int(torch.cuda.max_memory_allocated(self.device)),
            process_peak_reserved=int(torch.cuda.max_memory_reserved(self.device)),
        )

    def _static_signature(self):
        """One check per Decode token, rather than repeated across all Blocks.

        This reads tensor metadata/Python scalars only: no device sync, free-
        memory query, routing readback or model data transfer is introduced.
        """
        cache = self.cache
        if (
            cache is not self.manager.cache
            or cache is not self.engine.moe_offload_cache
            or cache.quant_format != "ds_fp4"
            or cache.decode_target != "gpu"
            or self.manager.writeback_ahead
            or not _owner_ensure(cache.ensure_experts, self.manager)
        ):
            raise RuntimeError("FFN back cache ownership/decode path changed")
        if cache.gate_up_alpha is not None or cache.down_alpha is not None:
            raise ValueError(
                "native DS FP4 back Graph requires both per-expert alpha banks to be None"
            )
        banks = tuple(cache.bank_views())
        if len(banks) != 4 or any(t.device != self.device or not t.is_contiguous() for t in banks):
            raise RuntimeError("FFN back complete expert-bank geometry changed")
        scalars = []
        for key, block in self.blocks.items():
            layer = block.ffn.experts
            if (
                layer.offload_cache is not cache
                or layer.tp_size != 1
                or cache.is_cpu_layer(layer.layer_id)
            ):
                raise RuntimeError("FFN back layer changed cache/CPU/TP ownership")
            scalars.append(
                (
                    key,
                    id(layer),
                    block.dim,
                    block.hc_mult,
                    layer.layer_id,
                    layer.top_k,
                    layer.swiglu_limit,
                    layer.activation,
                    layer.apply_router_weight_on_input,
                )
            )
        return (
            tuple(_storage(t) for t in banks),
            cache.cache_size,
            tuple(scalars),
            bool(torch.backends.cuda.matmul.allow_tf32),
            str(torch.get_float32_matmul_precision()),
            bool(torch.is_autocast_enabled("cuda")),
            str(torch.get_autocast_dtype("cuda")),
        )

    def begin_warmup(self):
        if self.state != "installed" or self.front.state != "warming":
            raise RuntimeError("FFN back warmup must follow front begin_warmup exactly once")
        self.memory_begin = self._memory()
        self.warmup_started = time.perf_counter()
        self.state = "warming"

    def begin_decode(self, batch):
        try:
            if self.state not in ("warming", "sealed") or self.step_active:
                raise RuntimeError("FFN back Decode is outside its lifecycle")
            if not batch.is_decode or batch.size != 1 or batch.padded_size != 1:
                raise RuntimeError("FFN back requires explicit Decode BS=1")
            if not self.front.step_active or self.front.step_batch is not batch:
                raise RuntimeError("FFN back Decode has no matching front step")
            if torch.cuda.is_current_stream_capturing() or self.inside_back:
                raise RuntimeError("nested FFN back capture/invocation is unsupported")
            stream = torch.cuda.current_stream(self.device)
            if self.stream is None:
                if stream == torch.cuda.default_stream(self.device):
                    raise RuntimeError("FFN back requires the original nondefault engine stream")
                self.stream = stream
            elif stream != self.stream:
                raise RuntimeError("FFN back requires its original engine stream")
            if self._static_signature() != self.static_signature:
                raise RuntimeError("FFN back bank/scalar/math signature changed before Attention")
            self.step_active, self.step_batch = True, batch
            self.step_keys, self.step_seen = [], set()
        except BaseException as error:
            self._fail(error)
            raise

    def end_decode(self):
        try:
            if not self.step_active or set(self.step_keys) != self.owned_ids:
                raise RuntimeError("Decode did not execute every FFN back exactly once")
            actual = tuple(self.step_keys)
            if actual != self.front.sequence:
                raise RuntimeError("FFN back order differs from its front controller")
            if self.sequence is None:
                if self.state != "warming":
                    raise RuntimeError("measurement cannot discover FFN back order")
                self.sequence = actual
            elif actual != self.sequence:
                raise RuntimeError("FFN back replay order changed")
            self.step_active, self.step_batch = False, None
            if self.state == "warming":
                self.stats["warmup_steps"] += 1
            else:
                self.stats["measured_steps"] += 1
                self.request_steps += 1
        except BaseException as error:
            self._fail(error)
            raise

    def seal(self):
        try:
            if (
                self.state != "warming"
                or self.step_active
                or self.front.state != "sealed"
                or self.sequence != self.front.sequence
                or set(self.entries) != self.owned_ids
            ):
                raise RuntimeError("FFN back seal requires its complete original Block sequence")
            if self._static_signature() != self.static_signature:
                raise RuntimeError("FFN back static ownership changed before sealing")
            self.stream.synchronize()
            self.memory_sealed = self._memory()
            self.warmup_seconds = time.perf_counter() - self.warmup_started
            self.state = "sealed"
        except BaseException as error:
            self._fail(error)
            raise

    def _bindings(self, block, front_entry):
        key = id(block)
        if (
            self.state not in ("warming", "sealed")
            or not self.step_active
            or self.front.step_batch is not self.step_batch
            or key not in self.owned_ids
            or key in self.step_seen
            or self.front.entries.get(key) is not front_entry
            or not self.front.step_keys
            or self.front.step_keys[-1] != key
        ):
            raise RuntimeError("FFN back input owner or step lifecycle changed")
        if self.sequence is not None and self.sequence[len(self.step_keys)] != key:
            raise RuntimeError("FFN back repeated or reordered a Block")
        if self.stream != self.front.stream:
            raise RuntimeError("FFN front, ensure and back must use the same original stream")
        layer = block.ffn.experts
        inputs = tuple(front_entry.outputs) + (front_entry.x,)
        if len(inputs) != 7:
            raise RuntimeError("FFN back requires all six front outputs and its residual input")
        entry = self.entries.get(key)
        if entry is not None:
            # Front entries own these private tensors for their entire lifetime;
            # only contents change. The pager overwrites slot contents, never
            # geometry. Full layout was checked at capture; reject rebinding or
            # storage replacement here without rebuilding 11 layout signatures.
            if any(
                current is not bound or int(current.data_ptr()) != pointer
                for current, bound, pointer in zip(inputs, entry.inputs, entry.input_pointers)
            ):
                raise RuntimeError("FFN back retained front tensor/storage changed")
            return inputs, self.banks, entry.signature
        norm_x, weights, slots, shared, post, comb, residual = inputs
        if (
            norm_x.shape != (1, block.dim)
            or norm_x.dtype != torch.bfloat16
            or shared.shape != norm_x.shape
            or shared.dtype != norm_x.dtype
            or weights.shape != (1, layer.top_k)
            or weights.dtype != torch.float32
            or slots.shape != weights.shape
            or slots.dtype != torch.int32
            or residual.shape != (1, 1, block.hc_mult, block.dim)
            or residual.dtype != norm_x.dtype
            or post.shape != (1, block.hc_mult)
            or comb.shape != (1, block.hc_mult, block.hc_mult)
        ):
            raise RuntimeError("FFN back native input geometry/dtype changed")
        if any(t.device != self.device or t.device.type != "cuda" for t in inputs) or any(
            not t.is_contiguous() for t in (norm_x, weights, slots, shared, residual)
        ):
            raise RuntimeError("FFN back requires native input layout on the owner CUDA device")
        signature = (tuple(_storage(t) for t in inputs), self.static_signature)
        return inputs, self.banks, signature

    def _protect_output(self, key, output):
        storage = output.untyped_storage()
        lo = int(storage.data_ptr())
        hi = lo + int(storage.nbytes())
        if any(old_key != key and max(lo, a) < min(hi, b) for a, b, old_key in self.output_ranges):
            raise RuntimeError("FFN back shared pool aliases another retained Block output")
        self.output_ranges.append((lo, hi, key))

    @torch.inference_mode()
    def run(self, block, front_entry):
        try:
            inputs, banks, signature = self._bindings(block, front_entry)
            key = id(block)
            entry = self.entries.get(key)
            if entry is not None and signature != entry.signature:
                raise RuntimeError(
                    "FFN back stable input/bank address/scalar/math signature changed"
                )
            if entry is None and (self.state != "warming" or self.sequence is not None):
                raise RuntimeError("measurement cannot capture new FFN back geometry")
            # This is exactly the native GPU _decode_routed ownership sequence.
            # The owner performs routing readback/CPU policy/SSD and all expert
            # DMA, then writes current slots on THIS stream. Nothing is replayed.
            self.cache.ensure_experts(block.ffn.experts.layer_id, inputs[2])
            self.cache.copy_missing()
            self.stats["ensures_outside_graph"] += 1
            if self.state == "sealed":
                self.request_ensures += 1
            self.step_keys.append(key)
            self.step_seen.add(key)
            if entry is None:
                expected_inputs = tuple(t.clone() for t in inputs)

                def invoke():
                    if self.inside_back:
                        raise RuntimeError("nested FFN back pure compute")
                    self.inside_back = True
                    try:
                        return _native_ffn_back(block, self.cache, inputs, banks)
                    finally:
                        self.inside_back = False

                reference = invoke()
                expected = reference.clone()
                self.stats["native_reference_calls"] += 1
                started = time.perf_counter()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, pool=self.pool if self.shared_pool else None, stream=self.stream
                ):
                    output = invoke()
                graph.replay()
                self.stream.synchronize()
                if _layout(output) != _layout(reference) or not _byte_equal(output, expected):
                    raise RuntimeError(
                        "native routed FP4 plus HC post differs bytewise under Graph"
                    )
                if not all(_byte_equal(t, wanted) for t, wanted in zip(inputs, expected_inputs)):
                    raise RuntimeError("FFN back compute mutated a retained front input")
                self._protect_output(key, output)
                entry = _Entry(
                    graph,
                    inputs,
                    banks,
                    output,
                    signature,
                    tuple(int(t.data_ptr()) for t in inputs),
                )
                self.entries[key] = entry
                if self.shared_pool and self.pool is None:
                    self.pool = graph.pool()
                self.capture_records[key] = dict(
                    name=self.names[key],
                    layer=int(block.layer_id),
                    capture_seconds=time.perf_counter() - started,
                    input_copy_bytes=0,
                    referenced_front_input_bytes=sum(_nbytes(t) for t in inputs),
                    retained_output_bytes=_nbytes(output),
                    output_byte_equal=True,
                    front_inputs_unchanged=True,
                    ensure_calls_for_capture_step=1,
                )
                self.stats["captures"] += 1
                self.stats["output_byte_checks"] += 1
                self.stats["immutable_input_byte_checks"] += len(inputs)
                self.stats["warmup_replays"] += 1
            else:
                entry.graph.replay()
                if self.state == "warming":
                    self.stats["warmup_replays"] += 1
                else:
                    self.stats["measured_replays"] += 1
                    self.request_replays += 1
            return entry.output
        except BaseException as error:
            self._fail(error)
            raise

    def snapshot(self):
        return dict(
            enabled=True,
            state=self.state,
            failure=self.failure,
            scope="native routed FP4 math, unchanged TP1 reduction, shared add and HC post; ensure/SSD/DMA remain eager",
            dynamic_inputs=(
                "normalized_x",
                "route_weights",
                "current_slot_ids",
                "shared_output",
                "HC_post_weights",
                "HC_combine_weights",
                "residual",
            ),
            input_bindings="retained front Graph outputs plus front static residual; no extra input staging",
            expected_blocks=len(self.owned_ids),
            captured_blocks=len(self.entries),
            shared_pool=self.shared_pool,
            strict_replay_order=True,
            counters=dict(self.stats),
            request_steps=self.request_steps,
            request_replays=self.request_replays,
            request_ensures=self.request_ensures,
            request_input_copy_bytes=0,
            extra_expert_payload_bytes=0,
            retained_output_bytes=sum(_nbytes(entry.output) for entry in self.entries.values()),
            memory_begin=self.memory_begin,
            memory_sealed=self.memory_sealed,
            capture_records=list(self.capture_records.values()),
            warmup_seconds=self.warmup_seconds,
            ensures_captured=False,
            expert_transfers_captured=False,
            captures_during_measurement=False,
            native_alpha_banks_none=True,
            static_guards="install, each Decode begin and seal",
            per_block_guards="front owner, replay order and retained input object/storage identities",
            failed_attention_steps_retried=False,
            validation="warmup native-vs-Graph output and input byte checks; independent full-model gate still required",
        )

    def close(self):
        if self.state == "closed":
            return
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except BaseException as error:
            self._fail(error)
            # Retain inputs, expert-bank owners, outputs, Graph and pool until
            # the original stream can be drained; close may be retried.
            raise
        self.entries.clear()
        self.blocks.clear()
        self.output_ranges.clear()
        self.pool = None
        self.state = "closed"


def install_ffn_back_graph(session, engine, *, shared_pool=True):
    manager = session.manager
    if getattr(manager, "ffn_back_graph", None) is not None:
        raise RuntimeError("FFN back Graph already installed")
    controller = FFNBackGraph(manager, engine, shared_pool=shared_pool)
    previous = vars(manager).get("ffn_back_graph")
    start = len(session.restore)
    try:
        session.restore.append((manager, "ffn_back_graph", previous))
        manager.ffn_back_graph = controller
    except BaseException:
        del session.restore[start:]
        if previous is None:
            if "ffn_back_graph" in vars(manager):
                delattr(manager, "ffn_back_graph")
        else:
            manager.ffn_back_graph = previous
        raise
    return controller
