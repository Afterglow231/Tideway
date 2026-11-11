"""Isolated DSV4 Attention prefix graphs; original-width selection stays eager.

Capture projections, RoPE, KV writes, compressors and indexer scores. The dynamic
top-k, sparse attention and output projection use the original incoming cap in
eager execution. In particular, BF16 top-k sees the original candidate width,
not padded max-context scores that can change tie selection. Every persistent
window/ring/pool write belongs to the prefix; warmup restores all state before
replay, and retains every prefix output until its eager consumer has completed.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import time

import torch

_ABSENT = object()


def _layout(t):
    return (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


def _storage(t):
    return (id(t), int(t.data_ptr()), _layout(t))


def _nbytes(t):
    return int(t.numel() * t.element_size())


def _byte_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _attention_prefix(attn, x, pos, rows, fixed_cap, wctx):
    """Exact native operation order through the indexer scores, excluding top-k.

    Only score storage/grid width differs from eager; the valid scoring lanes use
    the same kernel math. The finish slices scores before selection. This property
    still requires the independent unchanged-eager full-model numerical gate.
    """
    from freetoken.core import get_global_ctx
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace, fp4_act_quant_inplace
    from freetoken.kernel.triton.dsv4.hadamard import hadamard_transform
    from freetoken.kernel.triton.dsv4.norm import rms_norm
    from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode

    B, rd = x.size(0), attn.rope_head_dim
    if wctx is None:
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
    window_slots, prev_window_slots, _ = wctx
    freqs_t = attn.freqs_cis.index_select(0, pos)
    qr = q = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(q).unflatten(-1, (attn.n_heads, attn.head_dim))
    q = rms_norm(q, None, attn.eps)
    apply_rotary_emb_decode(q[..., -rd:], freqs_t)

    kv = attn.wkv(x)
    kv = attn.kv_norm(kv)
    apply_rotary_emb_decode(kv[..., -rd:], freqs_t)
    act_quant_fp8_inplace(kv[..., :-rd], 64)
    attn.attn.store_window(kv.view(B, -1), attn.layer_id, window_slots)

    if attn.compress_ratio:
        attn.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows)
    if attn.indexer is None:
        index_score = torch.empty((0,), dtype=x.dtype, device=x.device)
    else:
        indexer = attn.indexer
        ratio, idx_rd = indexer.compress_ratio, indexer.rope_head_dim
        n_stage = (fixed_cap + 1) // ratio
        iq = indexer.wq_b(qr).unflatten(-1, (indexer.n_heads, indexer.head_dim))
        apply_rotary_emb_decode(iq[..., -idx_rd:], indexer.freqs_cis.index_select(0, pos))
        iq = hadamard_transform(iq)
        fp4_act_quant_inplace(iq, 32)
        indexer.compressor.decode_step(x, pos, prev_window_slots, window_slots, rows)
        weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads**-0.5)
        valid = (pos + 1) // ratio
        index_score = indexer.attn.indexer_decode_scores(
            iq.reshape(B, indexer.n_heads, indexer.head_dim),
            weights.reshape(B, indexer.n_heads),
            valid,
            n_stage,
            ratio,
            indexer.layer_id,
        ).view(B, 1, n_stage)
    return q, freqs_t, index_score


def _attention_finish(attn, outputs, pos, rows, incoming_cap, wctx):
    """Original-width native top-k, sparse attention and output projection.

    This function never advances persistent KV/compressor state. It resolves block
    addresses from the current real metadata, outside the prefix snapshot proxy.
    """
    from freetoken.core import get_global_ctx
    from freetoken.models.deepseek_v4.ops import apply_rotary_emb_decode

    q, freqs_t, index_score = outputs
    B, ratio, rd = q.size(0), attn.compress_ratio, attn.rope_head_dim
    if wctx is None:
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
    _, _, window_slots_topk = wctx
    n_cmp_stage = (incoming_cap + 1) // ratio if ratio else 0
    cmp_counts = None
    if ratio:
        valid = (pos + 1) // ratio
        if attn.indexer is not None:
            if (
                index_score.ndim != 3
                or index_score.shape[:2] != (B, 1)
                or n_cmp_stage > index_score.shape[-1]
            ):
                raise RuntimeError(
                    "Attention prefix score capacity/layout does not cover original eager width"
                )
            # BS=M=1 makes this prefix view contiguous, matching native topk's
            # exact width/stride without a padded selection or extra copy.
            current_scores = index_score[..., :n_cmp_stage]
            if not current_scores.is_contiguous():
                raise RuntimeError(
                    "Attention prefix scores require contiguous original-width top-k input"
                )
            blocks = attn.indexer.attn.indexer_select_decode(
                current_scores, valid=valid, topk=attn.indexer.index_topk, offset=0
            )
        else:
            blk = torch.arange(n_cmp_stage, device=q.device)
            col_valid = blk[None, :] < valid[:, None]
            blocks = torch.where(col_valid, blk[None, :], -1).view(B, 1, n_cmp_stage)
        compress_idxs = attn.attn.blocks_to_global(blocks, ratio, rows=rows)
        topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1)
        cmp_counts = valid.clamp(max=compress_idxs.shape[-1]).to(torch.int32).view(B, 1)
    else:
        topk_idxs = window_slots_topk
    topk_idxs = topk_idxs.int()
    o = attn.attn.attend(
        q,
        attn.layer_id,
        topk_idxs,
        attn.window_size,
        attn.attn_sink,
        attn.softmax_scale,
        cmp_counts=cmp_counts,
        has_compression=bool(ratio),
    )
    apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)
    return attn._wo(o, B, 1)


@dataclass
class _Entry:
    graph: object
    x: object
    output: tuple
    states: tuple
    readonly: tuple
    signature: tuple


class AttentionDecodeGraph:
    def __init__(self, manager, engine, *, shared_pool=True):
        from .runtime import TidewayOffload
        from freetoken.models.deepseek_v4.attention import Attention

        if not isinstance(manager, TidewayOffload) or manager.geo.quant != "ds_fp4":
            raise TypeError("Attention prefix graph requires bounded DeepSeek Tideway")
        if any(
            getattr(manager, name, None) is not None
            for name in ("compressor_decode_graph", "decode_compute_graph", "pure_decode_graphs")
        ):
            raise ValueError("Attention prefix prototype must not compose other graph wrappers")
        if getattr(getattr(engine, "graph_runner", None), "graph_bs_list", []):
            raise ValueError("Attention prefix prototype cannot compose full-model graphs")
        if getattr(getattr(getattr(engine, "config", None), "tp_info", None), "size", 1) != 1:
            raise ValueError("Attention prefix prototype requires a single GPU")
        if type(shared_pool) is not bool:
            raise ValueError("shared_pool must be boolean")
        tree = getattr(engine.model, "_transformer", engine.model)
        found = [
            (name, module) for name, module in tree.named_modules() if isinstance(module, Attention)
        ]
        if not found:
            raise ValueError("model has no native Attention instances")
        self.manager, self.engine = manager, engine
        self.instances = {id(module): module for _, module in found}
        self.labels = {id(module): name for name, module in found}
        self.owned_ids = frozenset(self.instances)
        self.shared_pool, self.pool = shared_pool, None
        self.stream, self.device = None, engine.device
        self.entries, self.capture_records = {}, {}
        self.warmed, self.output_ranges = set(), []
        self.sequence = None
        self.step_keys, self.step_seen = [], set()
        self.step_active, self.step_ready = False, False
        self.step_batch, self.step_sources = None, None
        self.static_snapshot, self.static_common = None, None
        self.backend, self.fixed_cap = None, None
        self.state, self.failure = "installed", None
        self.memory_begin = self.memory_sealed = None
        self.warmup_started = self.warmup_seconds = None
        self.stats = dict(
            captures=0,
            eager_warmup_calls=0,
            native_reference_calls=0,
            state_byte_checks=0,
            output_byte_checks=0,
            state_restores=0,
            restore_on_failure=0,
            metadata_copies=0,
            snapshot_proxy_calls=0,
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
        self.request_metadata_copies = self.request_metadata_copy_bytes = 0

    def _fail(self, error):
        if self.state != "failed":
            self.failure = f"{type(error).__name__}: {error}"
            self.stats["failures"] += 1
        self.state = "failed"
        self.step_active = False

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
            raise RuntimeError("Attention graph warmup begins exactly once")
        self.memory_begin = self._memory()
        self.warmup_started = time.perf_counter()
        self.state = "warming"

    def seal(self):
        try:
            if self.state != "warming" or self.step_active:
                raise RuntimeError("seal requires successful completed warm Decode steps")
            missing = self.owned_ids - set(self.entries)
            if missing or self.sequence is None or set(self.sequence) != self.owned_ids:
                raise RuntimeError("Attention warmup lacks a complete captured layer sequence")
            self.stream.synchronize()
            self.memory_sealed = self._memory()
            self.warmup_seconds = time.perf_counter() - self.warmup_started
            self.state = "sealed"
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
                raise RuntimeError("Attention Decode step outside lifecycle boundary")
            if not batch.is_decode or batch.size != 1 or batch.padded_size != 1:
                raise RuntimeError("Attention graph prototype requires explicit Decode BS=1")
            self.step_active, self.step_ready = True, False
            self.step_batch, self.step_sources = batch, None
            self.step_keys, self.step_seen = [], set()
        except BaseException as error:
            self._fail(error)
            raise

    def end_decode(self):
        try:
            if not self.step_active or set(self.step_keys) != self.owned_ids:
                raise RuntimeError("Decode did not invoke each Attention exactly once")
            actual = tuple(self.step_keys)
            if self.sequence is None:
                if self.state != "warming":
                    raise RuntimeError("measurement cannot discover Attention order")
                self.sequence = actual
            elif actual != self.sequence:
                raise RuntimeError("Attention layer replay order changed")
            self.step_active = False
            self.step_batch = self.step_sources = None
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

    def _state_tensors(self, attn):
        # Every external write by decode_step: complete layer window, each full
        # carry ring (including scratch), and full cmp/idx pool (with scratch).
        states = [attn.attn.pool.window_pool[attn.layer_id]]
        if attn.compressor is not None:
            states.extend((attn.compressor.state_ring.buffer, attn.compressor.cmp_pool))
        if attn.indexer is not None:
            comp = attn.indexer.compressor
            states.extend((comp.state_ring.buffer, comp.cmp_pool))
        return tuple(states)

    @staticmethod
    def _math_flags():
        return (
            bool(torch.backends.cuda.matmul.allow_tf32),
            str(torch.get_float32_matmul_precision()),
            bool(torch.is_autocast_enabled("cuda")),
            str(torch.get_autocast_dtype("cuda")),
        )

    def _signature(self, attn, x, states):
        readonly = []
        constants = []
        # Parameters include FP8 scale, BF16 compressor weights, APE and norms.
        # RoPE tensors and scalar attributes are ordinary Python attributes.
        for name, module in attn.named_modules():
            for key, value in module._parameters.items():
                if value is not None:
                    readonly.append((name + "." + key, value))
            if isinstance(getattr(module, "freqs_cis", None), torch.Tensor):
                readonly.append((name + ".freqs_cis", module.freqs_cis))
            for key, value in vars(module).items():
                if not key.startswith("_") and type(value) in (int, float, bool, str):
                    constants.append((name, key, value))
        backend = attn.attn
        pool = backend.pool
        layer = int(attn.layer_id)
        rings = []
        for comp in (attn.compressor, None if attn.indexer is None else attn.indexer.compressor):
            if comp is not None:
                ring = comp.state_ring
                rings.append(
                    (
                        id(ring),
                        int(ring.ring_size),
                        int(ring.item_size),
                        int(ring.n_slots),
                        int(backend.compress_scratch_base(layer, comp.tier)),
                    )
                )
        signature = (
            _layout(x),
            tuple(_storage(t) for t in states),
            tuple((name, _storage(t)) for name, t in readonly),
            tuple(constants),
            tuple(rings),
            id(backend),
            id(pool),
            int(pool.P),
            str(pool.dtype),
            int(backend.window_size),
            bool(backend._RING_CHECK),
            self.fixed_cap,
            self._math_flags(),
        )
        return signature, tuple(t for _, t in readonly)

    def _validate_context(self, attn, x, pos, rows, wctx):
        if id(attn) not in self.owned_ids:
            raise ValueError("Attention instance is outside this controller")
        if tuple(x.shape[:2]) != (1, 1) or x.ndim != 3 or x.dtype != torch.bfloat16:
            raise RuntimeError("Attention graph requires BF16 x with BS=1 and one Decode token")
        if any(t.shape != (1,) or t.dtype != torch.int64 for t in (pos, rows)):
            raise RuntimeError("Attention graph pos/rows must be int64 vectors of length one")
        if wctx is not None and (not isinstance(wctx, tuple) or len(wctx) != 3):
            raise RuntimeError("native window context must contain three tensors")
        tensors = (x, pos, rows, *(() if wctx is None else wctx), *self._state_tensors(attn))
        if any(
            t.device.type != "cuda" or t.device != self.device or not t.is_contiguous()
            for t in tensors
        ):
            raise RuntimeError("Attention graph inputs/state must be contiguous on one CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("nested/full-model graph capture is unsupported")
        current = torch.cuda.current_stream(self.device)
        if self.stream is None:
            if current == torch.cuda.default_stream(self.device):
                raise RuntimeError("Attention graph requires the nondefault inference stream")
            self.stream = current
        elif current != self.stream:
            raise RuntimeError("Attention graph requires serial replay on its original stream")

    def _prepare_step(self, attn, pos, rows, wctx, incoming_cap):
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        if ctx.batch is not self.step_batch:
            raise RuntimeError("Attention graph global batch differs from active engine step")
        if self.step_ready:
            # The native Transformer supplies one hoisted context to all layers.
            # Do not silently reuse the first layer's context for different inputs.
            sources = (pos, rows, *(() if wctx is None else wctx))
            if tuple(_storage(t) for t in sources) != self.step_sources:
                raise RuntimeError("per-layer position/row/window context changed within a token")
            if attn.attn is not self.backend:
                raise RuntimeError("Attention backend changed within a token")
        else:
            md = ctx.batch.attn_metadata
            snapshot = md.full_snapshot()
            if snapshot.ndim != 2 or snapshot.shape[0] != 1 or snapshot.dtype != torch.int64:
                raise RuntimeError("Attention snapshot must be [1, context_cap] int64")
            if (
                snapshot.device != self.device
                or not snapshot.is_contiguous()
                or snapshot.shape[1] < 1
            ):
                raise RuntimeError("unsupported Attention snapshot layout/device")
            self.step_sources = tuple(
                _storage(t) for t in (pos, rows, *(() if wctx is None else wctx))
            )
            # Fresh every token. Never save a metadata-computed window context as
            # a Python cache across steps. The caller normally already computed it.
            actual_wctx = md.window_ctx(pos, rows) if wctx is None else wctx
            win = int(attn.window_size)
            if (
                len(actual_wctx) != 3
                or actual_wctx[0].shape != (1,)
                or actual_wctx[1].shape != (1,)
                or actual_wctx[2].shape != (1, 1, win)
                or any(
                    t.dtype != torch.int64 or t.device != self.device or not t.is_contiguous()
                    for t in actual_wctx
                )
            ):
                raise RuntimeError("native window context layout changed")
            common = (pos, rows, *actual_wctx)
            if self.static_snapshot is None:
                if self.state != "warming":
                    raise RuntimeError("measurement cannot allocate static Attention metadata")
                self.static_snapshot = torch.empty_like(snapshot)
                self.static_common = tuple(torch.empty_like(t) for t in common)
                self.fixed_cap = int(snapshot.shape[1] - 1)
                self.backend = attn.attn
            elif (
                _layout(snapshot) != _layout(self.static_snapshot)
                or attn.attn is not self.backend
                or tuple(_layout(t) for t in common)
                != tuple(_layout(t) for t in self.static_common)
            ):
                raise RuntimeError(
                    "Attention metadata capacity/backend/layout changed after capture"
                )
            self.static_snapshot.copy_(snapshot, non_blocking=True)
            for target, source in zip(self.static_common, common):
                target.copy_(source, non_blocking=True)
            self.step_ready = True
            self.stats["metadata_copies"] += 1
            if self.state == "sealed":
                self.request_metadata_copies += 1
                self.request_metadata_copy_bytes += _nbytes(snapshot) + sum(
                    _nbytes(t) for t in common
                )
        if type(incoming_cap) is not int or incoming_cap < 0 or incoming_cap > self.fixed_cap:
            raise RuntimeError("eager current position exceeds captured Attention context capacity")

    def _check_key(self, key):
        if not self.step_active or key in self.step_seen:
            raise RuntimeError("Attention must run exactly once in an active Decode step")
        index = len(self.step_keys)
        if self.sequence is not None and (
            index >= len(self.sequence) or self.sequence[index] != key
        ):
            raise RuntimeError("Attention layer order changed")
        self.step_keys.append(key)
        self.step_seen.add(key)

    @contextmanager
    def _snapshot_proxy(self):
        backend = self.backend
        previous = vars(backend).get("snapshot", _ABSENT)

        def snapshot():
            self.stats["snapshot_proxy_calls"] += 1
            return self.static_snapshot

        backend.snapshot = snapshot
        try:
            yield
        finally:
            if previous is _ABSENT:
                delattr(backend, "snapshot")
            else:
                backend.snapshot = previous

    def _capture_graph(self, invoke):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self.pool if self.shared_pool else None, stream=self.stream
        ):
            output = invoke()
        return graph, output

    def _prefix(self, attn, x, pos, rows, fixed_cap, wctx):
        return _attention_prefix(attn, x, pos, rows, fixed_cap, wctx)

    def _finish(self, attn, output, pos, rows, incoming_cap, wctx):
        return _attention_finish(attn, output, pos, rows, incoming_cap, wctx)

    def _restore(self, states, values):
        for state, saved in zip(states, values):
            state.copy_(saved, non_blocking=True)
        self.stream.synchronize()
        self.stats["state_restores"] += 1

    def _protect_output(self, key, output):
        for tensor in output:
            storage = tensor.untyped_storage()
            if storage.nbytes() == 0:
                continue
            start, end = int(storage.data_ptr()), int(storage.data_ptr()) + int(storage.nbytes())
            if any(
                old_key != key and max(start, lo) < min(end, hi)
                for lo, hi, old_key in self.output_ranges
            ):
                raise RuntimeError(
                    "shared Attention pool aliases a retained earlier layer prefix output"
                )
            self.output_ranges.append((start, end, key))

    def _capture_stateful(self, attn, x, states, signature, readonly):
        started = time.perf_counter()
        static_x = torch.empty_like(x)
        static_x.copy_(x, non_blocking=True)
        initial = tuple(t.clone() for t in states)
        expected = None
        pos, rows, *wctx = self.static_common

        def invoke():
            with self._snapshot_proxy():
                return self._prefix(attn, static_x, pos, rows, self.fixed_cap, tuple(wctx))

        try:
            self.stream.synchronize()
            # Also warms the fixed-cap score kernel shapes before capture.
            # This reference's persistent advance is undone before replay.
            reference = invoke()
            if (
                not isinstance(reference, tuple)
                or len(reference) != 3
                or not all(isinstance(t, torch.Tensor) for t in reference)
            ):
                raise RuntimeError("Attention prefix must return q, freqs and index scores")
            reference_layout = tuple(_layout(t) for t in reference)
            expected_output = tuple(t.clone() for t in reference)
            expected = tuple(t.clone() for t in states)
            self.stats["native_reference_calls"] += 1
            self._restore(states, initial)
            graph, output = self._capture_graph(invoke)
            graph.replay()
            self.stream.synchronize()
            if not all(_byte_equal(t, value) for t, value in zip(states, expected)):
                raise RuntimeError(
                    "Attention graph differs from native complete window/ring/pool state"
                )
            if (
                not isinstance(output, tuple)
                or len(output) != len(reference)
                or tuple(_layout(t) for t in output) != reference_layout
                or not all(_byte_equal(t, value) for t, value in zip(output, expected_output))
            ):
                raise RuntimeError(
                    "Attention prefix graph outputs differ bytewise from native prefix"
                )
            self._protect_output(id(attn), output)
            self.stats["state_byte_checks"] += len(states)
            self.stats["output_byte_checks"] += len(output)
        except BaseException as error:
            try:
                self._restore(states, initial)
                self.stats["restore_on_failure"] += 1
            except BaseException as restore_error:
                error.add_note(f"Attention initial-state restoration failed: {restore_error!r}")
            raise
        self.capture_records[id(attn)] = dict(
            name=self.labels[id(attn)],
            layer=int(attn.layer_id),
            ratio=int(attn.compress_ratio),
            fixed_cap=self.fixed_cap,
            static_input_bytes=_nbytes(static_x),
            temporary_validation_state_bytes=2 * sum(_nbytes(t) for t in states),
            capture_seconds=time.perf_counter() - started,
        )
        if self.shared_pool and self.pool is None:
            self.pool = graph.pool()
        return _Entry(graph, static_x, output, states, readonly, signature)

    @torch.inference_mode()
    def run(self, native, attn, x, pos, rows, cmp_stage_cap, wctx=None):
        if self.state in ("failed", "closed"):
            raise RuntimeError(f"Attention graph is {self.state}: {self.failure}")
        if not self._is_decode():
            self.stats["prefill_bypasses"] += 1
            return native(attn, x, pos, rows, cmp_stage_cap, wctx)
        try:
            if self.state not in ("warming", "sealed"):
                raise RuntimeError("explicit Attention warmup and seal required")
            self._validate_context(attn, x, pos, rows, wctx)
            key = id(attn)
            self._check_key(key)
            self._prepare_step(attn, pos, rows, wctx, cmp_stage_cap)
            if key not in self.warmed:
                if self.state != "warming":
                    raise RuntimeError("measurement reached an uncaptured Attention")
                # First token advances state once via eager prefix. Selection
                # and attention still use the incoming native width exactly.
                with self._snapshot_proxy():
                    output = self._prefix(attn, x, pos, rows, self.fixed_cap, wctx)
                self.warmed.add(key)
                self.stats["eager_warmup_calls"] += 1
                return self._finish(attn, output, pos, rows, cmp_stage_cap, wctx)
            states = self._state_tensors(attn)
            signature, readonly = self._signature(attn, x, states)
            entry = self.entries.get(key)
            if entry is None:
                if self.state != "warming":
                    raise RuntimeError("new Attention capture during measurement")
                entry = self._capture_stateful(attn, x, states, signature, readonly)
                self.entries[key] = entry
                self.stats["captures"] += 1
                self.stats["warmup_replays"] += 1
                return self._finish(attn, entry.output, pos, rows, cmp_stage_cap, wctx)
            if signature != entry.signature:
                raise RuntimeError(
                    "Attention weight/state address, layout, scalar or math flags changed"
                )
            entry.x.copy_(x, non_blocking=True)
            entry.graph.replay()
            if self.state == "warming":
                self.stats["warmup_replays"] += 1
            else:
                self.stats["measured_replays"] += 1
                self.request_replays += 1
                self.request_input_copy_bytes += _nbytes(x)
            return self._finish(attn, entry.output, pos, rows, cmp_stage_cap, wctx)
        except BaseException as error:
            self._fail(error)
            raise

    def snapshot(self):
        records = list(self.capture_records.values())
        net = None
        if self.memory_begin is not None and self.memory_sealed is not None:
            net = {
                key: self.memory_sealed[key] - self.memory_begin[key]
                for key in ("allocated", "reserved", "free")
            }
        return dict(
            enabled=True,
            scope="DSV4 Attention projections/KV/compressors/indexer-scores prefix; original-width top-k/sparse attention/output eager; CPU expert IO outside",
            state=self.state,
            failure=self.failure,
            fixed_cap=self.fixed_cap,
            expected_instances=len(self.owned_ids),
            captured_instances=len(self.entries),
            warmed_instances=len(self.warmed),
            shared_pool=self.shared_pool,
            strict_replay_order=True,
            outputs_retained_for_controller_lifetime=True,
            counters=dict(self.stats),
            request_replays=self.request_replays,
            request_steps=self.request_steps,
            request_input_copy_bytes=self.request_input_copy_bytes,
            request_metadata_copies=self.request_metadata_copies,
            request_metadata_copy_bytes=self.request_metadata_copy_bytes,
            static_metadata_bytes=0
            if self.static_snapshot is None
            else _nbytes(self.static_snapshot) + sum(_nbytes(t) for t in self.static_common),
            static_input_bytes=sum(_nbytes(e.x) for e in self.entries.values()),
            output_tensor_bytes=sum(_nbytes(t) for e in self.entries.values() for t in e.output),
            max_temporary_validation_state_bytes=max(
                (r["temporary_validation_state_bytes"] for r in records), default=0
            ),
            memory_begin=self.memory_begin,
            memory_sealed=self.memory_sealed,
            observed_warmup_interval_net_bytes=net,
            peak_counters_reset_by_controller=False,
            memory_note="Signed whole-warmup deltas include normal warmup; process peaks are read without reset, no positive per-capture sum.",
            selection_width="incoming native cmp_stage_cap; wide scores sliced before eager top-k; native sparse split shape",
            validation="native prefix vs Graph tuple outputs and complete state bytewise; independent unchanged-eager full-model numerical gate still required",
            capture_records=records,
            warmup_seconds=self.warmup_seconds,
            extra_expert_payload_bytes=0,
        )

    def close(self):
        if self.state == "closed":
            return
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except BaseException as error:
            # A failed drain does not prove that captured kernels have stopped
            # reading their inputs/state or writing retained outputs. Keep all
            # owners alive, reject inference, and allow a later close retry.
            self._fail(error)
            raise
        self.entries.clear()
        self.instances.clear()
        self.output_ranges.clear()
        self.pool = self.static_snapshot = self.static_common = None
        self.state = "closed"


def install_attention_decode_graph(session, engine, *, shared_pool=True):
    from freetoken.models.deepseek_v4.attention import Attention

    manager = session.manager
    if getattr(manager, "attention_decode_graph", None) is not None:
        raise RuntimeError("Attention Decode graph already installed")
    controller = AttentionDecodeGraph(manager, engine, shared_pool=shared_pool)
    restore_start = len(session.restore)
    previous_controller = vars(manager).get("attention_decode_graph", _ABSENT)
    try:
        native = Attention.decode_step

        def decode_step(attn, x, pos, rows, cmp_stage_cap, wctx=None):
            if id(attn) not in controller.owned_ids:
                return native(attn, x, pos, rows, cmp_stage_cap, wctx)
            return controller.run(native, attn, x, pos, rows, cmp_stage_cap, wctx)

        session.restore.append((Attention, "decode_step", vars(Attention).get("decode_step")))
        Attention.decode_step = decode_step
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
        manager.attention_decode_graph = controller
    except BaseException as installation_error:
        try:
            controller.close()
        except BaseException as cleanup_error:
            installation_error.add_note(f"Attention graph cleanup failed: {cleanup_error!r}")
        failed = []
        for owner, name, previous in reversed(session.restore[restore_start:]):
            try:
                current = vars(owner).get(name, _ABSENT)
                if previous is None:
                    if current is not _ABSENT:
                        delattr(owner, name)
                elif current is not previous:
                    setattr(owner, name, previous)
            except BaseException as rollback_error:
                failed.append((owner, name, previous))
                installation_error.add_note(
                    f"Attention binding rollback failed ({name}): {rollback_error!r}"
                )
        del session.restore[restore_start:]
        session.restore.extend(reversed(failed))
        try:
            if previous_controller is _ABSENT:
                if "attention_decode_graph" in vars(manager):
                    delattr(manager, "attention_decode_graph")
            else:
                manager.attention_decode_graph = previous_controller
        except BaseException as rollback_error:
            installation_error.add_note(f"Attention manager rollback failed: {rollback_error!r}")
        raise
    return controller
