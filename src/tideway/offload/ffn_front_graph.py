"""DSV4 FFN front graphs; attention, expert routing/IO and HC post stay eager.

The graph is a pure function of residual activation, current input token and
read-only front weights. All six outputs remain owned by the controller. Routed
expert execution may overwrite the IDs with cache slots; each graph replay must
write fresh logical IDs before the next ensure. No failed block is retried.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import time

import torch

from .attention_graph import _layout, _storage, _nbytes, _byte_equal

_ABSENT = object()


def _ffn_front(block, residual, input_ids):
    x, post, comb = block.hc_pre(residual, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    x = block.ffn_norm(x)
    x = x.view(-1, block.ffn.dim)
    weights, indices = block.ffn.gate(x, input_ids.flatten())
    # Preserve native MoE order: shared expert precedes routed CPU/SSD work.
    shared = block.ffn.shared_experts(x)
    return (
        x,
        weights.float().contiguous(),
        indices.to(torch.int32).contiguous(),
        shared,
        post,
        comb,
    )


@dataclass
class _Entry:
    graph: object
    x: object
    input_ids: object
    outputs: tuple
    readonly: tuple
    signature: tuple


class FFNFrontGraph:
    def __init__(self, manager, engine, *, shared_pool=True):
        from .runtime import TidewayOffload
        from freetoken.models.deepseek_v4.model import Block

        if not isinstance(manager, TidewayOffload) or manager.geo.quant != "ds_fp4":
            raise TypeError("FFN front graph requires bounded DeepSeek Tideway")
        if any(
            getattr(manager, name, None) is not None
            for name in ("pure_decode_graphs", "compressor_decode_graph", "decode_compute_graph")
        ):
            raise ValueError(
                "FFN front prototype cannot compose overlapping pure/MoE/compressor wrappers"
            )
        if getattr(getattr(engine, "graph_runner", None), "graph_bs_list", []):
            raise ValueError("FFN front graph cannot compose full-model graphs")
        if getattr(getattr(getattr(engine, "config", None), "tp_info", None), "size", 1) != 1:
            raise ValueError("FFN front graph requires a single GPU")
        if type(shared_pool) is not bool:
            raise ValueError("shared_pool must be boolean")
        tree = getattr(engine.model, "_transformer", engine.model)
        blocks = [
            (name, module) for name, module in tree.named_modules() if isinstance(module, Block)
        ]
        if not blocks:
            raise ValueError("model contains no native DSV4 Blocks")
        self.manager, self.engine, self.device = manager, engine, engine.device
        self.blocks = {id(block): block for _, block in blocks}
        self.names = {id(block): name for name, block in blocks}
        self.owned_ids = frozenset(self.blocks)
        self.shared_pool, self.pool, self.stream = shared_pool, None, None
        self.entries, self.capture_records, self.output_ranges = {}, {}, []
        self.sequence = None
        self.step_keys, self.step_seen, self.step_batch = [], set(), None
        self.step_active = self.inside_front = False
        self.state, self.failure = "installed", None
        self.memory_begin = self.memory_sealed = None
        self.warmup_started = self.warmup_seconds = None
        self.stats = dict(
            captures=0,
            native_reference_calls=0,
            output_byte_checks=0,
            warmup_replays=0,
            measured_replays=0,
            warmup_steps=0,
            measured_steps=0,
            prefill_bypasses=0,
            failures=0,
        )
        self.reset_stats()

    def reset_stats(self):
        self.request_replays = self.request_input_copy_bytes = self.request_steps = 0
        back = getattr(self.manager, "ffn_back_graph", None)
        if back is not None:
            back.reset_stats()

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

    def begin_warmup(self):
        if self.state != "installed":
            raise RuntimeError("FFN front warmup begins exactly once")
        self.memory_begin = self._memory()
        self.warmup_started = time.perf_counter()
        self.state = "warming"
        back = getattr(self.manager, "ffn_back_graph", None)
        if back is not None:
            back.begin_warmup()

    def seal(self):
        try:
            if (
                self.state != "warming"
                or self.step_active
                or self.sequence is None
                or set(self.sequence) != self.owned_ids
                or set(self.entries) != self.owned_ids
            ):
                raise RuntimeError("FFN front seal requires a complete captured Block sequence")
            self.stream.synchronize()
            self.memory_sealed = self._memory()
            self.warmup_seconds = time.perf_counter() - self.warmup_started
            self.state = "sealed"
            back = getattr(self.manager, "ffn_back_graph", None)
            if back is not None:
                back.seal()
        except BaseException as error:
            self._fail(error)
            raise

    @contextmanager
    def warmup(self):
        self.begin_warmup()
        try:
            yield self
            self.seal()
        except BaseException as error:
            self._fail(error)
            raise

    def begin_decode(self, batch):
        try:
            if self.state not in ("warming", "sealed") or self.step_active:
                raise RuntimeError("FFN front Decode is outside its lifecycle")
            if not batch.is_decode or batch.size != 1 or batch.padded_size != 1:
                raise RuntimeError("FFN front requires explicit Decode BS=1")
            self.step_active, self.step_batch = True, batch
            self.step_keys, self.step_seen = [], set()
            back = getattr(self.manager, "ffn_back_graph", None)
            if back is not None:
                back.begin_decode(batch)
        except BaseException as error:
            self._fail(error)
            raise

    def end_decode(self):
        try:
            if not self.step_active or set(self.step_keys) != self.owned_ids:
                raise RuntimeError("Decode did not execute every FFN front exactly once")
            actual = tuple(self.step_keys)
            if self.sequence is None:
                if self.state != "warming":
                    raise RuntimeError("measurement cannot discover FFN front order")
                self.sequence = actual
            elif actual != self.sequence:
                raise RuntimeError("FFN front Block order changed")
            back = getattr(self.manager, "ffn_back_graph", None)
            if back is not None:
                back.end_decode()
            self.step_active, self.step_batch = False, None
            if self.state == "warming":
                self.stats["warmup_steps"] += 1
            else:
                self.stats["measured_steps"] += 1
                self.request_steps += 1
        except BaseException as error:
            self._fail(error)
            raise

    def _is_decode(self):
        from freetoken.core import get_global_ctx

        return bool(get_global_ctx().batch.is_decode)

    def _validate_context(self, block, x, input_ids):
        from freetoken.core import get_global_ctx

        if get_global_ctx().batch is not self.step_batch:
            raise RuntimeError("FFN front global batch differs from active engine step")
        if x.shape != (1, 1, block.hc_mult, block.dim) or x.dtype != torch.bfloat16:
            raise RuntimeError("FFN front requires one BF16 Decode residual token")
        if input_ids.shape != (1, 1) or input_ids.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("FFN front requires dynamic int32/int64 input_ids [1,1]")
        if any(
            t.device != self.device or t.device.type != "cuda" or not t.is_contiguous()
            for t in (x, input_ids)
        ):
            raise RuntimeError("FFN front inputs must be contiguous on one CUDA device")
        if torch.cuda.is_current_stream_capturing() or self.inside_front:
            raise RuntimeError("nested FFN front/full-model capture is unsupported")
        stream = torch.cuda.current_stream(self.device)
        if self.stream is None:
            if stream == torch.cuda.default_stream(self.device):
                raise RuntimeError("FFN front requires the nondefault engine stream")
            self.stream = stream
        elif self.stream != stream:
            raise RuntimeError("FFN front requires serial replay on its original stream")

    def _signature(self, block, x, input_ids):
        readonly, constants = [], []
        for key in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
            readonly.append((key, getattr(block, key)))
        for prefix, root in (
            ("norm", block.ffn_norm),
            ("gate", block.ffn.gate),
            ("shared", block.ffn.shared_experts),
        ):
            for name, module in root.named_modules():
                for key, value in module._parameters.items():
                    if value is not None:
                        readonly.append((prefix + "." + name + "." + key, value))
                for key, value in vars(module).items():
                    if not key.startswith("_") and type(value) in (int, float, bool, str):
                        constants.append((prefix, name, key, value))
        scalar = (
            block.dim,
            block.hc_mult,
            block.norm_eps,
            block.hc_eps,
            block.hc_sinkhorn_iters,
            block.ffn.dim,
        )
        flags = (
            bool(torch.backends.cuda.matmul.allow_tf32),
            str(torch.get_float32_matmul_precision()),
            bool(torch.is_autocast_enabled("cuda")),
            str(torch.get_autocast_dtype("cuda")),
        )
        signature = (
            _layout(x),
            _layout(input_ids),
            tuple((name, _storage(t)) for name, t in readonly),
            tuple(constants),
            scalar,
            flags,
        )
        return signature, tuple(t for _, t in readonly)

    def _check_key(self, key):
        if not self.step_active or key not in self.owned_ids or key in self.step_seen:
            raise RuntimeError("FFN front needs one owned Block per active Decode step")
        index = len(self.step_keys)
        if self.sequence is not None and (
            index >= len(self.sequence) or self.sequence[index] != key
        ):
            raise RuntimeError("FFN front order changed or repeated")
        if self.state == "sealed" and key not in self.entries:
            raise RuntimeError("measurement cannot capture a new FFN front")
        self.step_keys.append(key)
        self.step_seen.add(key)

    def _front(self, block, x, input_ids):
        return _ffn_front(block, x, input_ids)

    def _capture_graph(self, invoke):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self.pool if self.shared_pool else None, stream=self.stream
        ):
            output = invoke()
        return graph, output

    def _protect_outputs(self, key, outputs):
        for tensor in outputs:
            storage = tensor.untyped_storage()
            start = int(storage.data_ptr())
            end = start + int(storage.nbytes())
            if any(
                old_key != key and max(start, lo) < min(end, hi)
                for lo, hi, old_key in self.output_ranges
            ):
                raise RuntimeError("shared FFN front pool aliases a retained earlier Block output")
            self.output_ranges.append((start, end, key))

    def _run_front(self, block, x, input_ids, signature, readonly):
        key = id(block)
        entry = self.entries.get(key)
        if entry is None:
            if self.state != "warming" or self.sequence is not None:
                raise RuntimeError("FFN front capture is limited to first complete warm Decode")
            static_x, static_ids = torch.empty_like(x), torch.empty_like(input_ids)
            static_x.copy_(x, non_blocking=True)
            static_ids.copy_(input_ids, non_blocking=True)

            def invoke():
                if self.inside_front:
                    raise RuntimeError("nested FFN front invocation")
                self.inside_front = True
                try:
                    return self._front(block, static_x, static_ids)
                finally:
                    self.inside_front = False

            reference = invoke()
            if (
                not isinstance(reference, tuple)
                or len(reference) != 6
                or not all(isinstance(t, torch.Tensor) for t in reference)
            ):
                raise RuntimeError("FFN front must return six tensor outputs")
            if (
                reference[1].dtype != torch.float32
                or reference[2].dtype != torch.int32
                or not reference[1].is_contiguous()
                or not reference[2].is_contiguous()
            ):
                raise RuntimeError("FFN front routing output contract changed")
            layouts = tuple(_layout(t) for t in reference)
            expected = tuple(t.clone() for t in reference)
            self.stats["native_reference_calls"] += 1
            started = time.perf_counter()
            graph, outputs = self._capture_graph(invoke)
            graph.replay()
            self.stream.synchronize()
            if (
                not isinstance(outputs, tuple)
                or len(outputs) != 6
                or tuple(_layout(t) for t in outputs) != layouts
                or not all(_byte_equal(t, wanted) for t, wanted in zip(outputs, expected))
            ):
                raise RuntimeError("FFN front Graph differs bytewise from native front")
            self._protect_outputs(key, outputs)
            entry = _Entry(graph, static_x, static_ids, outputs, readonly, signature)
            self.entries[key] = entry
            if self.shared_pool and self.pool is None:
                self.pool = graph.pool()
            self.capture_records[key] = dict(
                name=self.names[key],
                layer=int(block.layer_id),
                capture_seconds=time.perf_counter() - started,
                static_input_bytes=_nbytes(static_x) + _nbytes(static_ids),
                output_bytes=sum(_nbytes(t) for t in outputs),
            )
            self.stats["captures"] += 1
            self.stats["output_byte_checks"] += 6
            self.stats["warmup_replays"] += 1
        else:
            if signature != entry.signature:
                raise RuntimeError(
                    "FFN front input layout/weight address/scalar/math flags changed"
                )
            entry.x.copy_(x, non_blocking=True)
            entry.input_ids.copy_(input_ids, non_blocking=True)
            entry.graph.replay()
            if self.state == "warming":
                self.stats["warmup_replays"] += 1
            else:
                self.stats["measured_replays"] += 1
                self.request_replays += 1
                self.request_input_copy_bytes += _nbytes(x) + _nbytes(input_ids)
        return entry.outputs

    @torch.inference_mode()
    def run(self, native, block, x, pos, rows, cmp_stage_cap, input_ids, wctx=None):
        if self.state in ("failed", "closed"):
            raise RuntimeError(f"FFN front graph is {self.state}: {self.failure}")
        if not self._is_decode():
            self.stats["prefill_bypasses"] += 1
            return native(block, x, pos, rows, cmp_stage_cap, input_ids, wctx)
        try:
            if self.state not in ("warming", "sealed"):
                raise RuntimeError("explicit FFN front warmup/seal is required")
            # Reject obvious lifecycle/order/readonly changes before Attention
            # advances KV. Failures after Attention are fatal, never retried.
            self._validate_context(block, x, input_ids)
            self._check_key(id(block))
            signature, readonly = self._signature(block, x, input_ids)
            entry = self.entries.get(id(block))
            if entry is not None and signature != entry.signature:
                raise RuntimeError("FFN front signature changed before Attention")
            residual = x
            x, post, comb = block.hc_pre(
                x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
            )
            x = block.attn_norm(x)
            x = block.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx)
            x = block.hc_post(x, residual, post, comb)
            residual = x
            # HC post preserves the original residual shape and dtype.
            if _layout(x) != signature[0]:
                raise RuntimeError("FFN residual layout differs from validated Block input")
            norm_x, weights, logical_ids, shared, post, comb = self._run_front(
                block, x, input_ids, signature, readonly
            )
            back = getattr(self.manager, "ffn_back_graph", None)
            if back is not None:
                return back.run(block, self.entries[id(block)])
            routed = block.ffn.experts.routed_forward(norm_x, weights, logical_ids)
            x = (routed + shared).view(*residual.shape[:2], block.dim)
            return block.hc_post(x, residual, post, comb)
        except BaseException as error:
            self._fail(error)
            raise

    def snapshot(self):
        back = getattr(self.manager, "ffn_back_graph", None)
        return dict(
            enabled=True,
            state=self.state,
            failure=self.failure,
            scope="HC-FFN-pre, FFN norm, gate and shared expert; Attention and routed expert/IO and HC-post remain outside this FFN-front Graph",
            separate_ffn_back_graph=back.snapshot() if back is not None else None,
            dynamic_inputs=("residual_activation", "input_ids"),
            ids_rewritten_before_each_ensure=True,
            shared_pool=self.shared_pool,
            strict_replay_order=True,
            outputs_retained_for_controller_lifetime=True,
            expected_blocks=len(self.owned_ids),
            captured_blocks=len(self.entries),
            counters=dict(self.stats),
            request_steps=self.request_steps,
            request_replays=self.request_replays,
            request_input_copy_bytes=self.request_input_copy_bytes,
            static_input_bytes=sum(
                _nbytes(e.x) + _nbytes(e.input_ids) for e in self.entries.values()
            ),
            output_tensor_bytes=sum(_nbytes(t) for e in self.entries.values() for t in e.outputs),
            memory_begin=self.memory_begin,
            memory_sealed=self.memory_sealed,
            capture_records=list(self.capture_records.values()),
            warmup_seconds=self.warmup_seconds,
            validation="native front vs Graph six outputs bytewise; independent full-model eager gate required",
            extra_expert_payload_bytes=0,
            failed_attention_steps_retried=False,
        )

    def close(self):
        if self.state == "closed":
            return
        back = getattr(self.manager, "ffn_back_graph", None)
        if back is not None:
            back.close()
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except BaseException as error:
            # Retained routing IDs, post/comb and static inputs may still be in
            # use if the stream drain failed. Poison inference without dropping
            # their owners; a subsequent close can retry the drain safely.
            self._fail(error)
            raise
        self.entries.clear()
        self.blocks.clear()
        self.output_ranges.clear()
        self.pool = None
        self.state = "closed"


def install_ffn_front_graph(session, engine, *, shared_pool=True):
    from freetoken.models.deepseek_v4.model import Block

    manager = session.manager
    if getattr(manager, "ffn_front_graph", None) is not None:
        raise RuntimeError("FFN front graph already installed")
    controller = FFNFrontGraph(manager, engine, shared_pool=shared_pool)
    restore_start = len(session.restore)
    previous_controller = vars(manager).get("ffn_front_graph", _ABSENT)
    try:
        native = Block.decode_step

        def decode_step(block, x, pos, rows, cmp_stage_cap, input_ids, wctx=None):
            if id(block) not in controller.owned_ids:
                return native(block, x, pos, rows, cmp_stage_cap, input_ids, wctx)
            return controller.run(native, block, x, pos, rows, cmp_stage_cap, input_ids, wctx)

        session.restore.append((Block, "decode_step", vars(Block).get("decode_step")))
        Block.decode_step = decode_step
        original_forward = engine.forward_batch

        def forward_batch(batch, args):
            if not batch.is_decode:
                return original_forward(batch, args)
            controller.begin_decode(batch)
            try:
                output = original_forward(batch, args)
                controller.end_decode()
                return output
            except BaseException as error:
                controller._fail(error)
                raise

        session.restore.append((engine, "forward_batch", vars(engine).get("forward_batch")))
        engine.forward_batch = forward_batch
        manager.ffn_front_graph = controller
    except BaseException as error:
        try:
            controller.close()
        except BaseException as cleanup_error:
            error.add_note(f"FFN front cleanup failed: {cleanup_error!r}")
        failed = []
        for owner, name, previous in reversed(session.restore[restore_start:]):
            try:
                if previous is None:
                    if name in vars(owner):
                        delattr(owner, name)
                else:
                    setattr(owner, name, previous)
            except BaseException as rollback_error:
                failed.append((owner, name, previous))
                error.add_note(f"FFN front rollback failed: {rollback_error!r}")
        del session.restore[restore_start:]
        session.restore.extend(reversed(failed))
        try:
            if previous_controller is _ABSENT:
                if "ffn_front_graph" in vars(manager):
                    delattr(manager, "ffn_front_graph")
            else:
                manager.ffn_front_graph = previous_controller
        except BaseException as rollback_error:
            error.add_note(f"FFN front manager rollback failed: {rollback_error!r}")
        raise
    return controller
