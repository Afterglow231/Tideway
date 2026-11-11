"""Overlap Gemma expert branches with independently owned routing IDs.

Decode preserves public router IDs and gives the in-place slot mapper a
private output. Prefill uses the original backend implementation.
"""

from dataclasses import dataclass

from .overlap_shared_expert import SharedExpertOverlap


@dataclass
class _Route:
    router: object
    native: object
    hook: object = None
    pending: object = None
    capture_outputs: object = None
    captures: int = 0
    eager_calls: int = 0


class SharedOverlapRouteCopies(SharedExpertOverlap):
    def __init__(self, engine, **kwargs):
        self.routes = []
        self.route_by_owner = {}
        self.route_prefill_bypasses = 0
        super().__init__(engine, **kwargs)
        try:
            from .gemma_route_copy_kernel import check_upstream, routing_with_private_ids
            from freetoken.kernel.triton.gemma4_fused import gemma4_fused_routing

            self.routing_upstream = check_upstream()
            self.routing_kernel = routing_with_private_ids
            for layer in self.layers:
                router = layer.router
                if router.gemma4_fused_routing is not gemma4_fused_routing:
                    raise RuntimeError("Routing function is already patched")
                record = _Route(router, router.gemma4_fused_routing)
                self.routes.append(record)
                self.route_by_owner[id(router)] = record

                def route(logits, scale, topk, _record=record):
                    return self._route(_record, logits, scale, topk)

                record.hook = route
                router.gemma4_fused_routing = route
            # Stateless precompile: no router projection, model request, cache
            # access, KV write or actual input is used here.
            t = self.torch
            with t.inference_mode(), t.cuda.stream(self.stream):
                for record in self.routes:
                    router = record.router
                    x = t.zeros(
                        (1, router.per_expert_scale.numel()), device=self.device, dtype=t.bfloat16
                    )
                    self.routing_kernel(x, router.per_expert_scale, router._top_k)
            self._drain()
            self._check_memory(self._memory())
        except BaseException as error:
            self._rollback_install(error)
            raise

    def _route(self, p, logits, scale, topk):
        if p.router.gemma4_fused_routing is not p.hook:
            raise RuntimeError("Routing output owner changed")
        if self.state not in ("installed", "sealed"):
            raise RuntimeError("Routing copies are " + self.state)
        batch = self.context().batch
        if batch.is_prefill:
            if p.pending is not None:
                raise RuntimeError("Unconsumed routing IDs before prefill")
            self.route_prefill_bypasses += 1
            return p.native(logits, scale, topk)
        if (
            not batch.is_decode
            or batch.size != 1
            or batch.padded_size != 1
            or not self.inside
            or self.state == "sealed"
            or scale is not p.router.per_expert_scale
            or topk != p.router._top_k
        ):
            raise RuntimeError("Private IDs require original BS1 capture/dry-run routing")
        if p.pending is not None:
            raise RuntimeError("Private routing IDs were not consumed")
        weights, ids, private = self.routing_kernel(logits, scale, topk)
        p.pending = (ids, private)
        if self.torch.cuda.is_current_stream_capturing():
            if p.capture_outputs is not None:
                raise RuntimeError("Routing copy recapture requires a fresh owner")
            p.capture_outputs = (weights, ids, private)
            p.captures += 1
        else:
            p.eager_calls += 1
        return weights, ids

    def _private_ids(self, router, ids):
        record = self.route_by_owner[id(router)]
        if record.pending is None or record.pending[0] is not ids:
            raise RuntimeError("Private ID consumer does not match its router")
        private = record.pending[1]
        record.pending = None
        return private

    def _forward(self, layer, pre_ff, x):
        if self.state not in ("installed", "sealed"):
            raise RuntimeError("Shared overlap is " + self.state)
        batch = self.context().batch
        if batch.is_prefill:
            return super()._forward(layer, pre_ff, x)
        if (
            not batch.is_decode
            or batch.size != 1
            or batch.padded_size != 1
            or getattr(batch, "mm_embeds", None) is not None
        ):
            raise ValueError("Private routing IDs require native text decode BS1")
        self._check_layer(layer, pre_ff, x)
        capturing = self.torch.cuda.is_current_stream_capturing()
        self.inside = True
        forked = False
        joined = False
        try:
            layer.ready.record(self.stream)
            self.side.wait_event(layer.ready)
            with self.torch.cuda.stream(self.side):
                pre_ff.record_stream(self.side)
                try:
                    shared = layer.shared_forward(pre_ff)
                finally:
                    layer.done.record(self.side)
                    forked = True
            weights, ids = layer.router.forward(x)
            routed_in = layer.module.pre_feedforward_layernorm_2.forward(x)
            routed = layer.experts.routed_forward(
                routed_in, weights, self._private_ids(layer.router, ids)
            )
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
                self.stream.wait_event(layer.done)
            self.failure = f"{type(error).__name__}: {error}"
            self.state = "failed"
            raise
        finally:
            self.inside = False

    def seal(self):
        if not self.routes or any(p.captures != 1 or p.pending is not None for p in self.routes):
            raise RuntimeError(
                "Every private routing buffer must be captured and consumed exactly once"
            )
        return super().seal()

    def snapshot(self):
        result = super().snapshot()
        result["routing_copies"] = dict(
            layer_count=len(self.routes),
            captured_layers=sum(p.captures for p in self.routes),
            layer_capture_counts=[p.captures for p in self.routes],
            unconsumed_buffers=sum(p.pending is not None for p in self.routes),
            public_ids_preserved=True,
            prefill_native=True,
            decode_clone_nodes_removed_per_step=len(self.routes),
            arithmetic="Original routing kernel with one additional int32 store; original float operations and launch geometry.",
        )
        return result

    def _restore_methods(self):
        for record in reversed(self.routes):
            if (
                record.hook is not None
                and vars(record.router).get("gemma4_fused_routing") is record.hook
            ):
                record.router.gemma4_fused_routing = record.native
            record.pending = None
            record.capture_outputs = None
        super()._restore_methods()

    def close(self):
        if self.state == "closed":
            return
        for record in self.routes:
            if record.router.gemma4_fused_routing is not record.hook:
                raise RuntimeError("Remove foreign routing wrappers before close")
        super().close()


def install_shared_overlap_route_copies(engine, **kwargs):
    return SharedOverlapRouteCopies(engine, **kwargs)
