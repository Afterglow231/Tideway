"""Install Tideway expert managers with reversible cache and MoE-layer bindings.

Layer ownership, grouped-prefill dispatch, phase-policy compatibility and cache
entry points are checked before returning a Session."""

from dataclasses import dataclass, field
from typing import Any

from .runtime import TidewayOffload

from .hybrid import HybridOffload
from .phase_policy import configure_phase_policy, normalize_phase_policy_config

ARMS = {"tideway": TidewayOffload, "tideway_hybrid": HybridOffload}
_ABSENT = object()


def moe_layer_classes(geo):
    """Find MoE classes that define their own prefill routing implementation."""
    from freetoken.layers.moe import OffloadMoELayer

    classes = [OffloadMoELayer]
    if geo.quant == "ds_fp4":
        from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer

        classes.append(DSV4OffloadMoELayer)
    owning = [k for k in classes if "_prefill_routed" in vars(k)]
    return owning or classes


def _grouped_min_routes():
    """Read the grouped-GEMM crossover threshold from the backend."""
    try:
        from freetoken.moe.fused_ds_fp4 import _GROUPED_MIN_ROUTES

        return int(_GROUPED_MIN_ROUTES)
    except Exception:
        return 768  # Conservative fallback when the backend threshold is unavailable.


def runs_decode_gemv(is_prefill, n, routes, floor):
    """Return whether this dispatch would bypass grouped prefill GEMM.

    DS-FP4 requires a token count to select grouped GEMM. Below the backend's
    route threshold, falling back to per-route GEMV is intentional.
    """
    return bool(is_prefill) and n is None and routes >= floor


def gemv_message(routes, floor):
    return (
        f"ds_fp4 prefill with n=None and {routes} routes would run the decode per-route "
        f"GEMV instead of the grouped prefill GEMM (engine crossover "
        f"_GROUPED_MIN_ROUTES={floor}). Pass n=cache.cache_size. "
        f"See tideway/offload/install.py G2."
    )


@dataclass
class Session:
    manager: Any
    classes: list
    restore: list = field(default_factory=list)
    cache_restore: list = field(default_factory=list)

    def uninstall(self):
        for klass, name, original in reversed(self.restore):
            if original is None:
                delattr(klass, name)
            else:
                setattr(klass, name, original)
        self.restore.clear()
        for cache, name, original in reversed(self.cache_restore):
            if original is _ABSENT:
                if name in vars(cache):
                    delattr(cache, name)
            else:
                setattr(cache, name, original)
        self.cache_restore.clear()


def install(
    cache,
    *,
    geo,
    host_rows,
    reader,
    arm="tideway",
    writeback=True,
    seed_tail=False,
    prefill_guard=False,
    writeback_ahead="auto",
    dma_decode=True,
    kernel_guard=True,
    prior=None,
    core_rows=None,
    batch_prefill=True,
    ssd_prefetch_rows=0,
    volatile_mirror_rows=0,
    phase_policy=None,
):
    """Install a bounded or hybrid expert manager and return its lifecycle session.

    ``arm='tideway'`` selects the bounded pager; ``'tideway_hybrid'`` selects
    the static GPU-core layout. Phase policies require the bounded pager and
    non-ahead writeback. The reader remains owned by the caller.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    if reader.geo is not geo and reader.geo.name != geo.name:
        raise ValueError("reader was built from a different checkpoint geometry than `geo`")

    cls = ARMS[arm]
    policy_config = normalize_phase_policy_config(phase_policy)
    policy_enabled = policy_config.mode not in ("off", "exclusive")
    if policy_enabled:
        # Validate before constructors rebind cache methods or allocate payload.
        if cls is not TidewayOffload:
            raise ValueError(
                "phase_policy requires the bounded TidewayOffload arm; "
                f"{arm!r} is incompatible and will not be downgraded"
            )
        if writeback_ahead == "auto":
            effective_ahead = bool(writeback) and host_rows + cache.cache_size >= geo.total_experts
        elif type(writeback_ahead) is bool:
            effective_ahead = bool(writeback) and writeback_ahead
        else:
            raise ValueError('active phase_policy requires writeback_ahead to be bool or "auto"')
        if effective_ahead:
            raise ValueError(
                "active phase_policy requires non-ahead writeback; "
                "set writeback_ahead=False explicitly"
            )
    if ssd_prefetch_rows and cls is not TidewayOffload:
        raise ValueError("SSD prefetch leases are only implemented in the bounded Tideway runtime")
    if volatile_mirror_rows and cls is not HybridOffload:
        raise ValueError("volatile-core mirrors require the hybrid layout")
    cache_restore = [
        (cache, name, vars(cache).get(name, _ABSENT))
        for name in (
            "bank_sources",
            "banks",
            "bank_views",
            "ensure_experts",
            "copy_missing",
            "reset",
        )
    ]
    if cls is TidewayOffload:
        manager = cls(
            cache,
            host_rows,
            reader,
            geo=geo,
            writeback=writeback,
            seed_tail=seed_tail,
            prefill_guard=prefill_guard,
            writeback_ahead=writeback_ahead,
            dma_decode=dma_decode,
            batch_prefill=batch_prefill,
            ssd_prefetch_rows=ssd_prefetch_rows,
        )
    elif cls is HybridOffload:
        manager = cls(
            cache,
            host_rows,
            reader,
            geo=geo,
            prior_hist=prior,
            core_rows=core_rows,
            batch_prefill=batch_prefill,
            volatile_mirror_rows=volatile_mirror_rows,
        )

    # Patch each defining class and retain its original implementation for teardown.
    from freetoken.layers.moe import OffloadMoELayer

    classes = moe_layer_classes(geo)
    session = Session(manager=manager, classes=classes, cache_restore=cache_restore)
    try:
        if policy_enabled:
            # Configuration errors after construction use the same restoration
            # boundary as layer patching errors. This precedes any caller warmup.
            configure_phase_policy(manager, policy_config)
        replacements = {}
        for klass in classes:
            original = vars(klass).get("_prefill_routed")
            session.restore.append((klass, "_prefill_routed", original))

            def replacement(
                layer,
                hidden,
                weights,
                ids,
                *,
                _native=original,
                _ds_small=(geo.quant == "ds_fp4" and klass is not OffloadMoELayer),
            ):
                # Preserve DSV4's original small-prefill branch. The native
                # implementation calls the installed cache.ensure_experts and
                # copy_missing methods, so the bounded RAM/SSD owner remains in
                # control. Full chunks continue through the streaming manager.
                # Preserve the native small-prefill path for both layouts.
                if _ds_small and hidden.shape[0] * layer.top_k < geo.experts:
                    if cls is HybridOffload and hidden.shape[0] * layer.top_k > len(
                        manager.lru_slots
                    ):
                        # GPU-only lookup can pin at most the non-core LRU
                        # capacity. A multi-token active set larger than that
                        # would evict sole-owner core slots despite their high
                        # LRU timestamps. Stream this whole tiny chunk instead;
                        # the shape-based choice is identical across layers.
                        manager.stats["small_prefill_streamed_layers"] = (
                            manager.stats.get("small_prefill_streamed_layers", 0) + 1
                        )
                        return manager.prefill(
                            layer, hidden, weights, ids, allow_small_streaming=True
                        )
                    if cls is HybridOffload and layer.layer_id == 0:
                        manager.prepare_small_prefill()
                    manager.stats["small_prefill_calls"] = (
                        manager.stats.get("small_prefill_calls", 0) + 1
                    )
                    policy = getattr(manager, "phase_policy", None)
                    if policy is not None:
                        with policy.small_prefill():
                            return _native(layer, hidden, weights, ids)
                    return _native(layer, hidden, weights, ids)
                return manager.prefill(layer, hidden, weights, ids)

            replacements[klass] = replacement
            klass._prefill_routed = replacement
        for klass in classes:
            if vars(klass).get("_prefill_routed") is not replacements[klass]:
                raise RuntimeError(f"patch did not land on {klass.__name__}")

        # The base class has no separate small-prefill branch to guard.
        manager.crossover_guard = any(k is not OffloadMoELayer for k in classes)

        # Guard the DS-FP4 grouped-GEMM dispatch.
        if kernel_guard and geo.quant == "ds_fp4":
            floor = _grouped_min_routes()
            for klass in classes:
                native_gemm = klass._expert_gemm
                session.restore.append((klass, "_expert_gemm", vars(klass).get("_expert_gemm")))

                def guarded(
                    layer,
                    cache_,
                    hidden,
                    weights,
                    ids,
                    *,
                    views,
                    n,
                    alphas,
                    is_prefill,
                    _native=native_gemm,
                    _floor=floor,
                ):
                    routes = int(hidden.shape[0]) * int(getattr(layer, "top_k", 0) or 0)
                    if runs_decode_gemv(is_prefill, n, routes, _floor):
                        raise RuntimeError(gemv_message(routes, _floor))
                    return _native(
                        layer,
                        cache_,
                        hidden,
                        weights,
                        ids,
                        views=views,
                        n=n,
                        alphas=alphas,
                        is_prefill=is_prefill,
                    )

                klass._expert_gemm = guarded

        # Verify ownership of all three cache entry points.
        for attr in ("ensure_experts", "copy_missing", "bank_views"):
            if attr not in vars(cache) or not callable(vars(cache)[attr]):
                raise RuntimeError(f"cache.{attr} was not rebound by the {arm} arm")
        if getattr(cache.ensure_experts, "__self__", None) is not manager:
            raise RuntimeError(f"cache.ensure_experts is not bound to the installed {arm} manager")
    except BaseException as installation_error:
        if not policy_enabled:
            # Restore bindings when the phase policy is disabled.
            session.uninstall()
            raise
        try:
            # Close only manager-owned prefetch/lease resources. The supplied
            # reader remains caller-owned and is never closed here.
            manager.close()
        except BaseException as cleanup_error:
            installation_error.add_note(f"Phase-policy manager cleanup failed: {cleanup_error!r}")
        try:
            # Binding restoration must still run if resource cleanup failed.
            session.uninstall()
        except BaseException as rollback_error:
            installation_error.add_note(f"Phase-policy binding rollback failed: {rollback_error!r}")
        # Preserve the original installation failure, not a cleanup exception.
        raise
    return session
