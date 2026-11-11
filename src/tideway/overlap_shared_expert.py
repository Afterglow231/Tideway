"""Opt-in overlap of native Gemma shared and routed expert branches.

The two branches read the already-computed pre-FFN inputs and only meet at the
original dual-normalization merge. Their operators, BF16 intermediates, route
decisions, expert copies and merge arithmetic are unchanged. Prefill is native.

Install on an idle engine, recapture its native BS1 graph, then call seal(). Old
graphs are invalidated during installation. close() drains/destroys affected
graphs before restoring methods; recapture is required to resume graph serving.
This initial adapter supports the native full-host Gemma BF16 GPU path only.
"""

from dataclasses import dataclass

_ABSENT = object()


def _storage(t):
    return (int(t.data_ptr()), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


@dataclass
class _Layer:
    module: object
    native: object
    shared: object
    shared_forward: object
    router: object
    experts: object
    ready: object
    done: object
    hidden: int
    intermediate: int
    installed: object = None
    captures: int = 0
    eager_calls: int = 0


class SharedExpertOverlap:
    def __init__(self, engine, *, max_extra_cuda_bytes=64 << 20):
        import torch
        from freetoken.core import get_global_ctx
        from freetoken.engine.graph import GraphRunner
        from freetoken.models.gemma4.model import Gemma4ForCausalLM
        from freetoken.models.gemma4.moe import Gemma4MLP
        from freetoken.models.blocks import GatedMLP
        from freetoken.kernel.triton.gemma4_fused import gemma_dual_rmsnorm_residual_scalar

        if type(max_extra_cuda_bytes) is not int or max_extra_cuda_bytes <= 0:
            raise ValueError("positive extra-CUDA-memory limit required")
        if not isinstance(engine.model, Gemma4ForCausalLM) or engine.device.type != "cuda":
            raise TypeError("shared overlap initially supports native CUDA Gemma4 only")
        if engine.config.dtype != torch.bfloat16 or getattr(engine.config.tp_info, "size", 1) != 1:
            raise ValueError("shared overlap requires single-GPU BF16 inference")
        if getattr(engine, "tideway_shared_expert_overlap", None) is not None:
            raise RuntimeError("shared overlap already installed")
        cache = engine.moe_offload_cache
        if (
            cache is None
            or cache.quant_format != "bf16"
            or cache.decode_target != "gpu"
            or getattr(cache, "cpu_executor", None) is not None
            or "ensure_experts" in vars(cache)
        ):
            raise ValueError("shared overlap requires native GPU paging from the full host pool")
        runner = engine.graph_runner
        if type(runner) is not GraphRunner or runner.graph_bs_list != [1]:
            raise ValueError("shared overlap requires original native BS1 whole-model graphs")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("install shared overlap only outside capture")
        if engine.stream == torch.cuda.default_stream(engine.device):
            raise ValueError("shared overlap requires the original nondefault engine stream")
        modules = [layer.feed_forward for layer in engine.model.model.layers.op_list]
        if not modules or len({id(m) for m in modules}) != len(modules):
            raise RuntimeError("invalid shared-branch layer ownership")
        for m in modules:
            if (
                type(m) is not Gemma4MLP
                or type(m.shared_mlp) is not GatedMLP
                or "forward" in vars(m)
                or m.experts.offload_cache is not cache
                or "forward" in vars(m.shared_mlp)
            ):
                raise ValueError("unsupported or already-patched native Gemma MLP")
        self.torch, self.engine, self.cache, self.model = torch, engine, cache, engine.model
        self.context, self.merge = get_global_ctx, gemma_dual_rmsnorm_residual_scalar
        self.stream = engine.stream
        self.device = engine.device
        self.original_runner = runner
        self.sealed_runner = None
        self.limit = max_extra_cuda_bytes
        self.state = "installing"
        self.failure = None
        self.layers = []
        self.inside = False
        self.warmup_calls = 0
        self.prefill_bypasses = 0
        self.capture_calls = 0
        self.eager_calls = 0
        self.memory_before = self._memory()
        self.memory_warmed = None
        self.memory_sealed = None
        self.side = torch.cuda.Stream(device=self.device)
        self.bank_signature = tuple(_storage(t) for t in cache.bank_views())
        self.weights = tuple((n, _storage(t)) for n, t in self.model.state_dict().items())
        self.previous_owner = vars(engine).get("tideway_shared_expert_overlap", _ABSENT)
        try:
            # A stale old graph must never silently bypass the new hooks. This
            # mutates only this engine's graph objects, not its backend/source.
            self.stream.synchronize()
            self.side.synchronize()
            runner.destroy_cuda_graphs()
            for m in modules:
                up = m.shared_mlp.gate_up_proj.weight
                down = m.shared_mlp.down_proj.weight
                h = int(up.shape[1])
                i = int(down.shape[1])
                if tuple(up.shape) != (2 * i, h) or tuple(down.shape) != (h, i):
                    raise RuntimeError("unexpected native shared-MLP matrix shapes")
                layer = _Layer(
                    m,
                    m.forward,
                    m.shared_mlp,
                    m.shared_mlp.forward,
                    m.router,
                    m.experts,
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                    h,
                    i,
                )
                self.layers.append(layer)
            # Pure stateless initialization: zeros, native shared MLP, no router,
            # KV, expert-cache operation or actual request/input trajectory.
            self._warm_side()
            self.memory_warmed = self._memory()
            self._check_memory(self.memory_warmed)
            for layer in self.layers:

                def forward(pre_ff, x, _layer=layer):
                    return self._forward(_layer, pre_ff, x)

                layer.installed = forward
                layer.module.forward = forward
            engine.tideway_shared_expert_overlap = self
            self.state = "installed"
        except BaseException as error:
            self._rollback_install(error)
            raise

    def _rollback_install(self, error):
        self.failure = f"{type(error).__name__}: {error}"
        self.state = "failed"
        try:
            self._drain()
        except BaseException as drain_error:
            # Keep the events, stream, output owners and any installed wrappers
            # reachable if completion cannot be established. Do not expose a
            # native retry while side work may still reference these resources.
            self.engine.tideway_shared_expert_overlap = self
            error.add_note(
                f"shared-overlap installation drain failed; owners retained: {drain_error!r}"
            )
            return False
        self._restore_methods()
        return True

    def _memory(self):
        t = self.torch
        return dict(
            allocated=int(t.cuda.memory_allocated(self.device)),
            reserved=int(t.cuda.memory_reserved(self.device)),
            free=int(t.cuda.mem_get_info(self.device)[0]),
        )

    def _check_memory(self, now):
        for field in ("allocated", "reserved"):
            if now[field] - self.memory_before[field] > self.limit:
                raise RuntimeError(f"shared overlap exceeds extra CUDA {field} budget")

    def _drain(self):
        self.stream.synchronize()
        self.side.synchronize()

    def _warm_side(self):
        t = self.torch
        with t.inference_mode(), t.cuda.stream(self.stream):
            for layer in self.layers:
                x = t.zeros((1, layer.hidden), dtype=t.bfloat16, device=self.device)
                layer.ready.record(self.stream)
                self.side.wait_event(layer.ready)
                with t.cuda.stream(self.side):
                    x.record_stream(self.side)
                    y = layer.shared_forward(x)
                    layer.done.record(self.side)
                self.stream.wait_event(layer.done)
                y.record_stream(self.stream)
                self.warmup_calls += 1
        self._drain()

    def _check_layer(self, layer, pre_ff, x):
        t = self.torch
        if self.state not in ("installed", "sealed"):
            raise RuntimeError("shared overlap is " + self.state + ": " + str(self.failure))
        if (
            self.engine.model is not self.model
            or self.engine.moe_offload_cache is not self.cache
            or self.engine.stream != self.stream
            or layer.module.forward is not layer.installed
            or layer.module.shared_mlp is not layer.shared
            or layer.module.router is not layer.router
            or layer.module.experts is not layer.experts
            or layer.experts.offload_cache is not self.cache
        ):
            raise RuntimeError("shared overlap ownership changed")
        if t.cuda.current_stream(self.device) != self.stream or self.inside:
            raise RuntimeError("shared overlap requires one serial engine stream")
        for value in (pre_ff, x):
            if (
                value.device != self.device
                or value.dtype != t.bfloat16
                or tuple(value.shape) != (1, layer.hidden)
                or value.requires_grad
            ):
                raise ValueError("shared overlap requires BF16 text decode BS1 geometry")
        if self.state == "sealed" and t.cuda.is_current_stream_capturing():
            raise RuntimeError("unexpected recapture after shared-overlap seal")

    def _forward(self, layer, pre_ff, x):
        if self.state not in ("installed", "sealed"):
            raise RuntimeError("shared overlap is " + self.state + ": " + str(self.failure))
        batch = self.context().batch
        if batch.is_prefill:
            self.prefill_bypasses += 1
            return layer.native(pre_ff, x)
        if not batch.is_decode or batch.size != 1 or batch.padded_size != 1:
            raise ValueError("shared overlap requires explicit native decode BS1")
        if getattr(batch, "mm_embeds", None) is not None:
            raise ValueError("shared overlap supports text decode only")
        self._check_layer(layer, pre_ff, x)
        capturing = self.torch.cuda.is_current_stream_capturing()
        self.inside = True
        forked = False
        joined = False
        try:
            layer.ready.record(self.stream)
            self.side.wait_event(layer.ready)
            with self.torch.cuda.stream(self.side):
                # The producer is main; this input remains owned until the side
                # read completes. The output below is similarly recorded on its
                # consuming main stream before the native merge.
                pre_ff.record_stream(self.side)
                try:
                    shared = layer.shared_forward(pre_ff)
                finally:
                    layer.done.record(self.side)
                    forked = True
            topk_weights, topk_ids = layer.router.forward(x)
            routed_in = layer.module.pre_feedforward_layernorm_2.forward(x)
            routed = layer.experts.routed_forward(routed_in, topk_weights, topk_ids.clone())
            self.stream.wait_event(layer.done)
            joined = True
            shared.record_stream(self.stream)
            m = layer.module
            result = self.merge(
                shared,
                m.post_feedforward_layernorm_1.weight,
                routed,
                m.post_feedforward_layernorm_2.weight,
                m.post_feedforward_layernorm.weight,
                x,
                m.layer_scalar,
                m.post_feedforward_layernorm_1.eps,
                m.post_feedforward_layernorm_2.eps,
                m.post_feedforward_layernorm.eps,
            )
            if capturing:
                layer.captures += 1
                self.capture_calls += 1
            else:
                layer.eager_calls += 1
                self.eager_calls += 1
            return result
        except BaseException as error:
            if forked and not joined:
                try:
                    self.stream.wait_event(layer.done)
                except BaseException as join_error:
                    error.add_note(f"shared stream join failed: {join_error!r}")
            self.failure = f"{type(error).__name__}: {error}"
            self.state = "failed"
            raise
        finally:
            self.inside = False

    def seal(self):
        if self.state != "installed" or self.inside:
            raise RuntimeError("shared overlap seals once after native recapture")
        runner = self.engine.graph_runner
        try:
            if (
                runner is self.original_runner
                or runner.graph_bs_list != [1]
                or set(runner.graph_map) != {1}
                or any(layer.captures != 1 for layer in self.layers)
            ):
                raise RuntimeError(
                    "native BS1 graph did not capture every shared-branch fork exactly once"
                )
            if tuple(_storage(t) for t in self.cache.bank_views()) != self.bank_signature:
                raise RuntimeError("expert banks changed during shared-overlap installation")
            if tuple((n, _storage(t)) for n, t in self.model.state_dict().items()) != self.weights:
                raise RuntimeError("model weights changed during shared-overlap installation")
            self._drain()
            self.memory_sealed = self._memory()
            self._check_memory(self.memory_sealed)
            self.sealed_runner = runner
            self.state = "sealed"
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.state = "failed"
            raise
        return self.snapshot()

    def snapshot(self):
        return dict(
            enabled=self.state == "sealed",
            state=self.state,
            failure=self.failure,
            scope="native full-host Gemma4 BF16 text decode BS1; native Prefill unchanged",
            layer_count=len(self.layers),
            capture_forks=self.capture_calls,
            layer_capture_forks=[layer.captures for layer in self.layers],
            eager_decode_branch_calls=self.eager_calls,
            pure_shared_warmup_calls=self.warmup_calls,
            prefill_bypasses=self.prefill_bypasses,
            extra_cuda_budget_bytes=self.limit,
            memory_before=self.memory_before,
            memory_after_side_warmup=self.memory_warmed,
            memory_after_capture=self.memory_sealed,
            nonreused_shared_activation_upper_bound_bytes=sum(
                2 * (3 * layer.intermediate + layer.hidden) for layer in self.layers
            ),
            cuBLAS_workspace="Opaque implementation allocation; actual CUDA allocated/reserved deltas are guarded.",
            expert_payload_or_policy_changed=False,
            arithmetic="Original shared MLP, router, routed MLP and dual-normalization merge; unchanged BF16 intermediates.",
            dependencies="main-ready event -> shared side stream; shared-done event -> native merge on main",
            lifetime="pre_ff.record_stream(side); shared.record_stream(main); both streams drained before teardown",
            actual_overlap="Requires profiler verification; fork/join graph topology alone does not prove concurrent execution.",
            warmup="Pure native shared MLP on synthetic zeros before capture; no router, KV or expert-cache state updates.",
        )

    def _restore_methods(self):
        for layer in self.layers:
            if layer.installed is not None and vars(layer.module).get("forward") is layer.installed:
                delattr(layer.module, "forward")
        if vars(self.engine).get("tideway_shared_expert_overlap") is self:
            if self.previous_owner is _ABSENT:
                delattr(self.engine, "tideway_shared_expert_overlap")
            else:
                self.engine.tideway_shared_expert_overlap = self.previous_owner

    def close(self):
        if self.state == "closed":
            return
        if self.inside or self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot close shared overlap during a forward/capture")
        self._drain()
        # Graph nodes refer to side-stream allocations/events. Destroy them
        # before dropping any owner or allowing a plain native recapture.
        self.engine.graph_runner.destroy_cuda_graphs()
        self._restore_methods()
        self.state = "closed"


def install_shared_expert_overlap(engine, *, max_extra_cuda_bytes=64 << 20):
    """Install before final native recapture; call controller.seal() afterwards."""
    return SharedExpertOverlap(engine, max_extra_cuda_bytes=max_extra_cuda_bytes)
