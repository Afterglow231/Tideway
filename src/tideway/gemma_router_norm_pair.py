"""Paired Gemma router and routed RMSNorm for single-token decode.

Install on an idle full-host engine before graph recapture, then call seal().
Install shared-expert overlap first when combining the adapters, and close in
reverse order. Activation owners remain outside the model's state_dict graph.
"""

from dataclasses import dataclass

_ABSENT = object()


def _storage(t):
    return (int(t.data_ptr()), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


@dataclass
class _Pair:
    mlp: object
    router: object
    norm: object
    router_native: object
    norm_native: object
    ones: object
    hidden: int
    eps: float
    root: float
    router_hook: object = None
    norm_hook: object = None
    pending: object = None
    capture_outputs: object = None
    captures: int = 0
    eager_calls: int = 0


class GemmaRouterNormPair:
    def __init__(self, engine, *, max_extra_cuda_bytes=64 << 20):
        import torch
        from sgl_kernel import rmsnorm
        from sgl_kernel.elementwise import _has_flashinfer
        from flashinfer.norm import _use_cuda_norm
        from freetoken.core import get_global_ctx
        from freetoken.engine.graph import GraphRunner
        from freetoken.layers.norm import GemmaRMSNorm
        from freetoken.models.gemma4.model import Gemma4ForCausalLM
        from freetoken.models.gemma4.moe import Gemma4MLP, Gemma4Router
        from .cute_router_norm_pair import check_upstream, router_norm_pair

        if type(max_extra_cuda_bytes) is not int or max_extra_cuda_bytes <= 0:
            raise ValueError("positive extra-CUDA-memory limit required")
        if not isinstance(engine.model, Gemma4ForCausalLM) or engine.device.type != "cuda":
            raise TypeError("norm pair requires native CUDA Gemma4")
        if engine.config.dtype != torch.bfloat16 or getattr(engine.config.tp_info, "size", 1) != 1:
            raise ValueError("norm pair requires single-GPU BF16 inference")
        if getattr(engine, "tideway_gemma_router_norm_pair", None) is not None:
            raise RuntimeError("norm pair already installed")
        cache, runner = engine.moe_offload_cache, engine.graph_runner
        if (
            cache is None
            or cache.quant_format != "bf16"
            or cache.decode_target != "gpu"
            or getattr(cache, "cpu_executor", None) is not None
            or "ensure_experts" in vars(cache)
        ):
            raise ValueError("norm pair requires native GPU paging from the full host pool")
        if type(runner) is not GraphRunner or runner.graph_bs_list != [1]:
            raise ValueError("norm pair requires original native BS1 whole-model graphs")
        if (
            torch.cuda.is_current_stream_capturing()
            or not _has_flashinfer
            or _use_cuda_norm(engine.device)
            or torch.compiler.is_dynamo_compiling()
        ):
            raise RuntimeError("install outside capture using the original CuTe norm dispatch")
        overlap = getattr(engine, "tideway_shared_expert_overlap", None)
        if overlap is not None and overlap.state != "installed":
            raise RuntimeError("install norm pair after overlap installation, before either seal")
        modules = [layer.feed_forward for layer in engine.model.model.layers.op_list]
        if not modules or len({id(m) for m in modules}) != len(modules):
            raise RuntimeError("invalid native MLP ownership")
        for m in modules:
            r, n = m.router, m.pre_feedforward_layernorm_2
            if (
                type(m) is not Gemma4MLP
                or type(r) is not Gemma4Router
                or type(r.norm) is not GemmaRMSNorm
                or type(n) is not GemmaRMSNorm
                or r.norm.with_scale
                or not n.with_scale
                or r.norm.eps != n.eps
                or r.norm.size != n.size
                or r.norm.rmsnorm is not rmsnorm
                or n.rmsnorm is not rmsnorm
                or "forward" in vars(r)
                or "forward" in vars(n)
                or "forward" in vars(r.norm)
                or m.experts.offload_cache is not cache
            ):
                raise ValueError(
                    "norm pair requires unpatched native router and paired raw-weight norm"
                )
            if "forward" in vars(m) and (
                overlap is None
                or not any(
                    layer_entry.module is m and layer_entry.installed is m.forward
                    for layer_entry in overlap.layers
                )
            ):
                raise ValueError("unknown MLP wrapper cannot guarantee paired norm consumption")
        self.torch, self.engine, self.cache, self.model = torch, engine, cache, engine.model
        self.context, self.kernel = get_global_ctx, router_norm_pair
        self.stream, self.device, self.overlap = engine.stream, engine.device, overlap
        self.original_runner, self.sealed_runner = runner, None
        self.state, self.failure, self.inside = "installing", None, False
        self.layers = []
        self.prefill_bypasses = 0
        self.capture_calls = 0
        self.eager_calls = 0
        self.limit = max_extra_cuda_bytes
        self.upstream = check_upstream()
        self.memory_before = self._memory()
        self.memory_warmed = None
        self.memory_sealed = None
        self.weights = tuple((n, _storage(t)) for n, t in self.model.state_dict().items())
        self.bank_signature = tuple(_storage(t) for t in cache.bank_views())
        self.previous_owner = vars(engine).get("tideway_gemma_router_norm_pair", _ABSENT)
        try:
            self._drain()
            runner.destroy_cuda_graphs()
            # Pure stateless precompile/warmup only. No routes, cache accesses,
            # KV writes, sampling, or real/future request inputs are involved.
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                for m in modules:
                    r, n = m.router, m.pre_feedforward_layernorm_2
                    x = torch.zeros((1, n.size), device=self.device, dtype=torch.bfloat16)
                    ones = r.norm._kernel_weight(x)
                    p = _Pair(m, r, n, r.forward, n.forward, ones, n.size, n.eps, r._scalar_root)
                    self.layers.append(p)
                    self.kernel(x, ones, n.weight, r.scale, p.eps, p.root)
            self._drain()
            self.memory_warmed = self._memory()
            self._check_memory(self.memory_warmed)
            for p in self.layers:

                def route(x, _p=p):
                    return self._route(_p, x)

                def norm(x, _p=p):
                    return self._norm(_p, x)

                p.router_hook, p.norm_hook = route, norm
                p.router.forward, p.norm.forward = route, norm
            engine.tideway_gemma_router_norm_pair = self
            self.state = "installed"
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.state = "failed"
            try:
                self._drain()
                self._restore()
            except BaseException as drain_error:
                engine.tideway_gemma_router_norm_pair = self
                error.add_note(f"norm-pair rollback failed; owners retained: {drain_error!r}")
            raise

    def _memory(self):
        t = self.torch
        return dict(
            allocated=int(t.cuda.memory_allocated(self.device)),
            reserved=int(t.cuda.memory_reserved(self.device)),
        )

    def _check_memory(self, now):
        if any(now[k] - self.memory_before[k] > self.limit for k in ("allocated", "reserved")):
            raise RuntimeError("norm pair exceeds extra CUDA allocation budget")

    def _drain(self):
        self.stream.synchronize()
        if self.overlap is not None:
            self.overlap._drain()

    def _check_owner(self, p):
        if self.state not in ("installed", "sealed"):
            raise RuntimeError("norm pair is " + self.state + ": " + str(self.failure))
        if (
            self.engine.model is not self.model
            or self.engine.moe_offload_cache is not self.cache
            or self.engine.stream != self.stream
            or p.mlp.router is not p.router
            or p.mlp.pre_feedforward_layernorm_2 is not p.norm
            or p.router.forward is not p.router_hook
            or p.norm.forward is not p.norm_hook
            or p.router.norm._ones_weight is not p.ones
            or p.router._scalar_root != p.root
            or p.norm.eps != p.eps
            or p.router.norm.eps != p.eps
            or getattr(self.engine, "tideway_shared_expert_overlap", None) is not self.overlap
        ):
            raise RuntimeError("norm pair ownership or arithmetic parameters changed")

    def _check_decode(self, p, x):
        t = self.torch
        batch = self.context().batch
        if (
            not batch.is_decode
            or batch.size != 1
            or batch.padded_size != 1
            or getattr(batch, "mm_embeds", None) is not None
        ):
            raise ValueError("norm pair requires native text decode BS1")
        if (
            t.cuda.current_stream(self.device) != self.stream
            or self.inside
            or x.device != self.device
            or x.dtype != t.bfloat16
            or tuple(x.shape) != (1, p.hidden)
            or not x.is_contiguous()
            or x.requires_grad
        ):
            raise ValueError("invalid norm-pair stream or geometry")
        if self.state == "sealed":
            raise RuntimeError(
                "sealed norm pair requires native graph replay; eager/recapture is unsupported"
            )

    def _route(self, p, x):
        self._check_owner(p)
        if self.context().batch.is_prefill:
            if p.pending is not None:
                raise RuntimeError("unfinished decode norm pair before prefill")
            self.prefill_bypasses += 1
            return p.router_native(x)
        self._check_decode(p, x)
        if p.pending is not None:
            raise RuntimeError("paired routed norm was not consumed")
        self.inside = True
        try:
            h, routed = self.kernel(x, p.ones, p.norm.weight, p.router.scale, p.eps, p.root)
            p.pending = (x, routed)
            # Original projection and fused top-k route arithmetic are unchanged.
            logits = p.router.proj.forward(h)
            result = p.router.gemma4_fused_routing(
                logits, p.router.per_expert_scale, p.router._top_k
            )
            if self.torch.cuda.is_current_stream_capturing():
                if p.capture_outputs is not None:
                    raise RuntimeError("multiple captures need a fresh norm-pair controller")
                p.capture_outputs = (h, routed)
                p.captures += 1
                self.capture_calls += 1
            else:
                p.eager_calls += 1
                self.eager_calls += 1
            return result
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.state = "failed"
            raise
        finally:
            self.inside = False

    def _norm(self, p, x):
        self._check_owner(p)
        if self.context().batch.is_prefill:
            if p.pending is not None:
                raise RuntimeError("unfinished decode norm pair before prefill")
            return p.norm_native(x)
        self._check_decode(p, x)
        if p.pending is None or p.pending[0] is not x:
            self.failure = "paired norm must immediately consume the same router input"
            self.state = "failed"
            raise RuntimeError(self.failure)
        result = p.pending[1]
        p.pending = None
        return result

    def seal(self):
        if self.state != "installed" or self.inside:
            raise RuntimeError("norm pair seals once after final native graph capture")
        try:
            runner = self.engine.graph_runner
            if (
                runner is self.original_runner
                or runner.graph_bs_list != [1]
                or set(runner.graph_map) != {1}
                or any(p.captures != 1 or p.pending is not None for p in self.layers)
            ):
                raise RuntimeError("native recapture did not execute every norm pair exactly once")
            for p in self.layers:
                self._check_owner(p)
            if tuple((n, _storage(t)) for n, t in self.model.state_dict().items()) != self.weights:
                raise RuntimeError("norm pair changed model weights/state_dict")
            if tuple(_storage(t) for t in self.cache.bank_views()) != self.bank_signature:
                raise RuntimeError("norm pair changed expert banks")
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
            state=self.state,
            enabled=self.state == "sealed",
            failure=self.failure,
            layer_count=len(self.layers),
            capture_pairs=self.capture_calls,
            layer_capture_pairs=[p.captures for p in self.layers],
            eager_decode_pairs=self.eager_calls,
            prefill_router_bypasses=self.prefill_bypasses,
            replay_count=None,
            replay_count_note="Native GraphRunner does not expose execution counters.",
            scope="Native full-host Gemma BF16 text decode BS1; native prefill unchanged",
            upstream_sources=self.upstream,
            arithmetic="Original CuTe F32 RMS reduction/rsqrt; "
            "router norm BF16 -> scale multiply BF16 -> F32-root multiply BF16; "
            "raw routed-weight norm BF16; original router projection and top-k.",
            composition="shared overlap first, norm pair second, one final native recapture, seal both",
            activation_ownership="Controller only; no extra BaseOP tensor attributes",
            extra_cuda_budget_bytes=self.limit,
            memory_before=self.memory_before,
            memory_after_operator_warmup=self.memory_warmed,
            memory_after_capture=self.memory_sealed,
            expert_payload_or_policy_changed=False,
            numeric_qualification="Separate raw-bit and full-model gates required",
        )

    def _restore(self):
        for p in self.layers:
            if p.router_hook is not None and vars(p.router).get("forward") is p.router_hook:
                delattr(p.router, "forward")
            if p.norm_hook is not None and vars(p.norm).get("forward") is p.norm_hook:
                delattr(p.norm, "forward")
            p.pending = None
            p.capture_outputs = None
        if vars(self.engine).get("tideway_gemma_router_norm_pair") is self:
            if self.previous_owner is _ABSENT:
                delattr(self.engine, "tideway_gemma_router_norm_pair")
            else:
                self.engine.tideway_gemma_router_norm_pair = self.previous_owner

    def close(self):
        if self.state == "closed":
            return
        if self.inside or self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot close norm pair during forward/capture")
        # Fail closed if somebody has wrapped our closures; dropping their
        # activation owners before that wrapper is removed would be unsafe.
        for p in self.layers:
            if (p.router_hook is not None and p.router.forward is not p.router_hook) or (
                p.norm_hook is not None and p.norm.forward is not p.norm_hook
            ):
                raise RuntimeError("remove foreign norm-pair wrappers before close")
        self._drain()
        self.engine.graph_runner.destroy_cuda_graphs()
        self._restore()
        self.state = "closed"


def install_gemma_router_norm_pair(engine, *, max_extra_cuda_bytes=64 << 20):
    """Install before final native graph recapture, then call controller.seal()."""
    return GemmaRouterNormPair(engine, max_extra_cuda_bytes=max_extra_cuda_bytes)
