"""Opt-in single-launch Q/K/V head normalization for native BS1 Gemma graphs."""

from dataclasses import dataclass

_ABSENT = object()


@dataclass
class _Attention:
    module: object
    native: object
    q_weight: object
    k_weight: object
    v_weight: object
    hook: object = None
    captures: int = 0
    output_owner: object = None


class GemmaGroupedQKVNorm:
    def __init__(self, engine, *, max_extra_cuda_bytes=64 << 20):
        import torch
        from sgl_kernel import rmsnorm
        from freetoken.core import get_global_ctx
        from freetoken.models.gemma4.attention import Gemma4Attention
        from freetoken.models.gemma4.model import Gemma4ForCausalLM
        from freetoken.engine.graph import GraphRunner
        from .cute_grouped_qkv_norm import grouped_qkv_norm

        if not isinstance(engine.model, Gemma4ForCausalLM) or engine.device.type != "cuda":
            raise TypeError("Native CUDA Gemma required")
        if getattr(engine, "tideway_grouped_qkv_norm", None) is not None:
            raise RuntimeError("Grouped head norm is already installed")
        if type(engine.graph_runner) is not GraphRunner or engine.graph_runner.graph_bs_list != [1]:
            raise ValueError("Original native BS1 graphs required")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Install outside capture")
        self.engine, self.model, self.cache, self.stream = (
            engine,
            engine.model,
            engine.moe_offload_cache,
            engine.stream,
        )
        self.torch, self.context, self.kernel = torch, get_global_ctx, grouped_qkv_norm
        self.state = "installing"
        self.failure = None
        self.layers = []
        self.prefill_bypasses = 0
        self.inside = False
        self.previous = vars(engine).get("tideway_grouped_qkv_norm", _ABSENT)
        self.runner = engine.graph_runner
        self.limit = max_extra_cuda_bytes
        self.memory_before = self._memory()
        self.weights = tuple(
            (n, int(t.data_ptr()), tuple(t.shape), tuple(t.stride()), str(t.dtype))
            for n, t in self.model.state_dict().items()
        )
        try:
            self.stream.synchronize()
            self.runner.destroy_cuda_graphs()
            modules = [layer.self_attn for layer in engine.model.model.layers.op_list]
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                for a in modules:
                    if (
                        type(a) is not Gemma4Attention
                        or "forward" in vars(a)
                        or a.head_dim not in (256, 512)
                        or a.q_norm.rmsnorm is not rmsnorm
                        or a.k_norm.rmsnorm is not rmsnorm
                        or a.v_norm.rmsnorm is not rmsnorm
                        or not a.q_norm.with_scale
                        or not a.k_norm.with_scale
                        or a.v_norm.with_scale
                        or not (a.q_norm.eps == a.k_norm.eps == a.v_norm.eps)
                    ):
                        raise ValueError("Unmodified native Gemma head norms required")
                    x = torch.zeros(
                        (a.num_qo_heads + 2 * a.num_kv_heads, a.head_dim),
                        device=engine.device,
                        dtype=torch.bfloat16,
                    )
                    p = _Attention(
                        a,
                        a.forward,
                        a.q_norm._kernel_weight(x),
                        a.k_norm._kernel_weight(x),
                        a.v_norm._kernel_weight(x),
                    )
                    self.layers.append(p)
                    self.kernel(
                        x,
                        p.q_weight,
                        p.k_weight,
                        p.v_weight,
                        a.num_qo_heads,
                        a.num_kv_heads,
                        a.q_norm.eps,
                    )
            self.stream.synchronize()
            self._check_memory()
            for p in self.layers:

                def forward(x, _p=p):
                    return self._forward(_p, x)

                p.hook = forward
                p.module.forward = forward
            engine.tideway_grouped_qkv_norm = self
            self.state = "installed"
        except BaseException as error:
            self.state = "failed"
            self.failure = f"{type(error).__name__}: {error}"
            self.stream.synchronize()
            self._restore()
            raise

    def _memory(self):
        return dict(
            allocated=int(self.torch.cuda.memory_allocated(self.engine.device)),
            reserved=int(self.torch.cuda.memory_reserved(self.engine.device)),
        )

    def _check_memory(self):
        now = self._memory()
        if any(now[k] - self.memory_before[k] > self.limit for k in now):
            raise RuntimeError("Grouped head norm exceeds extra allocation budget")

    def _forward(self, p, x):
        if self.state not in ("installed", "sealed") or p.module.forward is not p.hook:
            raise RuntimeError("Grouped norm owner/state changed")
        if (
            self.engine.model is not self.model
            or self.engine.moe_offload_cache is not self.cache
            or self.engine.stream != self.stream
        ):
            raise RuntimeError("Grouped norm engine ownership changed")
        ctx = self.context()
        batch = ctx.batch
        if batch.is_prefill:
            self.prefill_bypasses += 1
            return p.native(x)
        if (
            self.state == "sealed"
            or self.inside
            or not batch.is_decode
            or batch.size != 1
            or batch.padded_size != 1
            or getattr(batch, "mm_embeds", None) is not None
            or self.torch.cuda.current_stream(self.engine.device) != self.stream
            or x.shape[0] != 1
            or x.dtype != self.torch.bfloat16
        ):
            raise RuntimeError("Grouped head norm requires original text decode capture/dry-runs")
        self.inside = True
        try:
            a = p.module
            positions = batch.positions
            T = x.shape[0]
            qkv = a.qkv_proj.forward(x)
            packed = self.kernel(
                qkv.view(a.num_qo_heads + 2 * a.num_kv_heads, a.head_dim),
                p.q_weight,
                p.k_weight,
                p.v_weight,
                a.num_qo_heads,
                a.num_kv_heads,
                a.q_norm.eps,
            )
            q_lin, k_lin, v_lin = packed.view(T, -1).split((a.q_dim, a.kv_dim, a.kv_dim), dim=-1)
            q = q_lin.view(T, a.num_qo_heads, a.head_dim)
            k = k_lin.view(T, a.num_kv_heads, a.head_dim)
            v = v_lin.view(T, a.num_kv_heads, a.head_dim)
            q, k = a._apply_rope(positions, q, k)
            k = k.reshape(T, a.num_kv_heads * a.head_dim)
            v = v.reshape(T, a.num_kv_heads * a.head_dim)
            o = ctx.attn_backend.forward(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                a.layer_id,
                batch,
                attn_spec=a.attn_spec,
            )
            o = o.reshape(T, a.num_qo_heads * a.head_dim)
            result = a.o_proj.forward(o)
            if self.torch.cuda.is_current_stream_capturing():
                if p.output_owner is not None:
                    raise RuntimeError("Grouped head norm requires one final graph capture")
                p.output_owner = packed
                p.captures += 1
            return result
        except BaseException as error:
            self.state = "failed"
            self.failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.inside = False

    def seal(self):
        if self.state != "installed" or any(p.captures != 1 for p in self.layers):
            raise RuntimeError("All head norms must be captured exactly once")
        if self.engine.graph_runner is self.runner or self.engine.graph_runner.graph_bs_list != [1]:
            raise RuntimeError("Final native recapture required")
        current = tuple(
            (n, int(t.data_ptr()), tuple(t.shape), tuple(t.stride()), str(t.dtype))
            for n, t in self.model.state_dict().items()
        )
        if current != self.weights:
            raise RuntimeError("Grouped norms changed model weight/state ownership")
        self.stream.synchronize()
        self._check_memory()
        self.state = "sealed"
        return self.snapshot()

    def snapshot(self):
        return dict(
            state=self.state,
            enabled=self.state == "sealed",
            failure=self.failure,
            layer_count=len(self.layers),
            capture_counts=[p.captures for p in self.layers],
            norm_nodes_removed_per_step=2 * len(self.layers),
            prefill_native=True,
            prefill_bypasses=self.prefill_bypasses,
            arithmetic="Original FlashInfer per-head BF16 RMSNorm row arithmetic and layout; Q/K/V groups share a launch only.",
        )

    def _restore(self):
        for p in reversed(self.layers):
            if p.hook is not None and vars(p.module).get("forward") is p.hook:
                delattr(p.module, "forward")
            p.output_owner = None
        if vars(self.engine).get("tideway_grouped_qkv_norm") is self:
            if self.previous is _ABSENT:
                delattr(self.engine, "tideway_grouped_qkv_norm")
            else:
                self.engine.tideway_grouped_qkv_norm = self.previous

    def close(self):
        if self.state == "closed":
            return
        if self.inside:
            raise RuntimeError("Cannot close grouped norms during forward")
        for p in self.layers:
            if p.module.forward is not p.hook:
                raise RuntimeError("Remove foreign attention wrappers before close")
        self.stream.synchronize()
        self.engine.graph_runner.destroy_cuda_graphs()
        self._restore()
        self.state = "closed"


def install_gemma_grouped_qkv_norm(engine, **kwargs):
    return GemmaGroupedQKVNorm(engine, **kwargs)
