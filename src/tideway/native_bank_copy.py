"""Opt-in byte-identical native two-bank copies; original prefill stays native."""


class NativeBankCopy:
    def __init__(self, engine):
        import torch
        from freetoken.core import get_global_ctx
        from .balanced_bank_copy_kernel import balanced_copy

        cache = engine.moe_offload_cache
        if (
            cache is None
            or cache.quant_format != "bf16"
            or cache.decode_target != "gpu"
            or getattr(cache, "cpu_executor", None) is not None
            or not cache._copy_fused_ok
            or len(cache.banks) != 2
            or "ensure_experts" in vars(cache)
        ):
            raise ValueError("Original native two-bank pinned BF16 cache required")
        if getattr(engine, "tideway_native_bank_copy", None) is not None:
            raise RuntimeError("Native copy already installed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Install outside capture")
        self.engine, self.cache, self.stream = engine, cache, engine.stream
        self.torch, self.context, self.kernel = torch, get_global_ctx, balanced_copy
        self.native = cache.copy_missing
        self.state = "installed"
        self.failure = None
        self.captures = [0] * len(cache._copy_src_ptrs)
        self.prefill_bypasses = 0
        self.eager_calls = 0
        self.geometry = dict(blocks=(8, 8), block=4096, warps=8)
        self.bank_signature = tuple(
            (int(v.data_ptr()), tuple(v.shape), tuple(v.stride()), str(v.dtype))
            for v in cache.bank_views()
        )
        self.old = vars(cache).get("copy_missing", None)
        self.hook = lambda: self._copy()
        # An independent zero-count scalar compiles only the copy launch. It
        # never writes expert payloads or changes the model's native counters.
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            zero = torch.zeros_like(cache.num_indices)
            self.kernel(
                cache._copy_dst_ptrs,
                cache._copy_src_ptrs[0],
                cache._copy_feat_bytes,
                cache.evict_slots,
                cache.src_indices,
                zero,
                **self.geometry,
            )
        self.stream.synchronize()
        cache.copy_missing = self.hook
        engine.tideway_native_bank_copy = self

    def _copy(self):
        if self.state not in ("installed", "sealed"):
            raise RuntimeError("Native copy is " + self.state)
        if self.engine.moe_offload_cache is not self.cache or self.engine.stream != self.stream:
            raise RuntimeError("Native copy owner changed")
        if self.context().batch.is_prefill:
            self.prefill_bypasses += 1
            return self.native()
        if self.state == "sealed":
            raise RuntimeError("Native copy requires sealed graph replay, not eager decode")
        layer = self.cache._pending_src_layer
        if type(layer) is not int or not 0 <= layer < len(self.captures):
            raise RuntimeError("Original pending source layer required")
        if self.torch.cuda.current_stream(self.engine.device) != self.stream:
            raise RuntimeError("Native engine stream required")
        self.kernel(
            self.cache._copy_dst_ptrs,
            self.cache._copy_src_ptrs[layer],
            self.cache._copy_feat_bytes,
            self.cache.evict_slots,
            self.cache.src_indices,
            self.cache.num_indices,
            **self.geometry,
        )
        if self.torch.cuda.is_current_stream_capturing():
            self.captures[layer] += 1
        else:
            self.eager_calls += 1

    def seal(self):
        if self.state != "installed" or self.captures != [1] * len(self.captures):
            raise RuntimeError("Each native bank copy must be captured exactly once")
        current = tuple(
            (int(v.data_ptr()), tuple(v.shape), tuple(v.stride()), str(v.dtype))
            for v in self.cache.bank_views()
        )
        if current != self.bank_signature:
            raise RuntimeError("Copy optimization changed expert banks")
        self.stream.synchronize()
        self.state = "sealed"
        return self.snapshot()

    def snapshot(self):
        return dict(
            state=self.state,
            enabled=self.state == "sealed",
            failure=self.failure,
            layer_count=len(self.captures),
            capture_counts=list(self.captures),
            prefill_native=True,
            prefill_bypasses=self.prefill_bypasses,
            eager_calls=self.eager_calls,
            geometry=self.geometry,
            expert_policy_changed=False,
            expert_payload_precision_changed=False,
            extra_payload_bytes=0,
        )

    def close(self):
        if self.state == "closed":
            return
        if self.cache.copy_missing is not self.hook:
            raise RuntimeError("Remove foreign copy wrappers before close")
        self.stream.synchronize()
        self.engine.graph_runner.destroy_cuda_graphs()
        if self.old is None:
            delattr(self.cache, "copy_missing")
        else:
            self.cache.copy_missing = self.old
        if getattr(self.engine, "tideway_native_bank_copy", None) is self:
            delattr(self.engine, "tideway_native_bank_copy")
        self.state = "closed"


def install_native_bank_copy(engine):
    return NativeBankCopy(engine)
