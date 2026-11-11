"""Lifecycle management for native and paged decode execution.

Select execution from the installed pager, quantization and graph capabilities.
Install instrumentation before calling after_collector_installed(), then run
the application's warmup inside warmup(). Existing graph controllers retain
their ownership until explicitly closed.
"""

from contextlib import contextmanager
from dataclasses import dataclass

_ABSENT = object()
_DS_ATTRIBUTES = (
    "route_map_uploader",
    "attention_decode_graph",
    "ffn_front_graph",
    "ffn_back_graph",
    "segmented_decode",
)


def _dependencies():
    import torch
    from freetoken.engine.graph import GraphRunner
    from .offload.runtime import TidewayOffload
    from .offload.hybrid import HybridOffload, recapture_decode_graphs
    from .offload.paged_decode_graph import enable_paged_decode_graph
    from .offload.segmented_decode import enable_segmented_decode

    return dict(
        bounded_type=TidewayOffload,
        hybrid_type=HybridOffload,
        native_type=GraphRunner,
        paged=enable_paged_decode_graph,
        segmented=enable_segmented_decode,
        recapture=recapture_decode_graphs,
        stream_context=torch.cuda.stream,
    )


def _bound_owner(method):
    seen = set()
    while method is not None and id(method) not in seen:
        seen.add(id(method))
        owner = getattr(method, "__self__", None)
        if owner is not None:
            return owner
        method = getattr(method, "__wrapped__", None)
    return None


@dataclass(frozen=True)
class DecodeSelection:
    path: str
    reason: str
    quantization: str | None
    native_batch_sizes: tuple = ()


def _select(engine, manager, session, deps):
    if session is not None:
        supplied = getattr(session, "manager", None)
        if supplied is None:
            raise ValueError("Session has no pager owner")
        if manager is None:
            manager = supplied
        elif supplied is not manager:
            raise ValueError("Session and manager owners differ")
    cache = getattr(engine, "moe_offload_cache", None)
    if cache is None:
        raise TypeError("Decode execution requires an expert cache")
    if manager is None:
        owner = _bound_owner(getattr(cache, "ensure_experts", None))
        if isinstance(owner, (deps["bounded_type"], deps["hybrid_type"])):
            manager = owner
    if manager is not None and getattr(manager, "cache", None) is not cache:
        raise ValueError("Engine and pager expert-cache owners differ")
    if (
        getattr(cache, "decode_target", "gpu") != "gpu"
        or getattr(engine, "cpu_moe_executor", None) is not None
    ):
        raise TypeError("CPU-executed expert paths are not supported by this facade")
    if isinstance(manager, deps["bounded_type"]):
        quant = getattr(manager.geo, "quant", None)
        if quant == "bf16":
            choice = DecodeSelection(
                "paged_segments",
                "Installed bounded BF16 host pager requires explicit ensure boundaries",
                quant,
            )
        elif quant == "ds_fp4":
            if session is None:
                raise ValueError("Bounded FP4 segments require the original installation Session")
            choice = DecodeSelection(
                "deepseek_segments",
                "Installed bounded FP4 pager uses the existing attention/front/back adapters",
                quant,
            )
        else:
            raise TypeError("No graph-preserving bounded adapter for quantization " + str(quant))
    elif manager is None or isinstance(manager, deps["hybrid_type"]):
        runner = getattr(engine, "graph_runner", None)
        if not isinstance(runner, deps["native_type"]):
            raise TypeError("Device-controlled execution requires the original native GraphRunner")
        sizes = tuple(getattr(runner, "graph_bs_list", ()))
        if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError(
                "Native graph capability was not declared; eager fallback is not selected"
            )
        if isinstance(manager, deps["hybrid_type"]):
            # Hybrid intentionally clears native bank_sources/banks and binds
            # its own fixed RAM-row table plus GPU payload banks instead.
            if (
                getattr(manager, "ram_row_dev", None) is None
                or not getattr(cache, "bank_caches", None)
                or not callable(getattr(cache, "bank_views", None))
                or not tuple(cache.bank_views())
            ):
                raise ValueError(
                    "Hybrid graph execution requires its fixed RAM-row map and GPU banks"
                )
            if _bound_owner(getattr(cache, "ensure_experts", None)) is not manager:
                raise ValueError("Hybrid graph cache is not owned by the installed device pager")
        elif not getattr(cache, "banks", None):
            raise ValueError(
                "Native full-host execution requires existing fixed expert-bank sources"
            )
        choice = DecodeSelection(
            "native_graph",
            "Device-controlled expert residency supports the existing whole-step graph",
            getattr(cache, "quant_format", None),
            sizes,
        )
    else:
        raise TypeError("Unknown pager ownership; no implicit eager fallback")
    return choice, manager


class DecodeExecution:
    def __init__(self, engine, manager=None, session=None):
        if getattr(engine, "tideway_decode_execution", None) is not None:
            raise RuntimeError("Unified decode execution already installed")
        self.deps = _dependencies()
        self.selection, self.manager = _select(engine, manager, session, self.deps)
        self.engine, self.session = engine, session
        self.cache = engine.moe_offload_cache
        self.state, self.failure = "selected", None
        self.controller = None
        self.native_runner = None
        self.native_guard = None
        self.native_previous = _ABSENT
        self.pending_cleanup = ()
        self.session_start = None
        self.session_tail = ()
        self.session_installed = {}
        self.manager_before = {}
        self.collector = None
        self.native_recapture_record = None
        self.wrapper_counts = dict(
            after_collector_calls=0, warmup_scopes=0, native_recapture_calls=0
        )
        self.engine_previous = vars(engine).get("tideway_decode_execution", _ABSENT)
        engine.tideway_decode_execution = self

    def _owner(self):
        if (
            getattr(self.engine, "tideway_decode_execution", None) is not self
            or self.engine.moe_offload_cache is not self.cache
        ):
            raise RuntimeError("Unified decode execution ownership changed")

    def after_collector_installed(self, collector=None):
        """Explicit setup fence; collector=None means tracing is intentionally off."""
        self._owner()
        if self.state != "selected":
            raise RuntimeError("Decode setup runs exactly once after collector installation")
        if collector is not None and getattr(collector, "cache", None) is not self.cache:
            raise ValueError("Collector observes a different expert cache")
        self.collector = collector
        self.wrapper_counts["after_collector_calls"] += 1
        try:
            if self.selection.path == "paged_segments":
                self.controller = self.deps["paged"](self.engine, self.manager)
            elif self.selection.path == "deepseek_segments":
                present = [
                    name for name in _DS_ATTRIBUTES if getattr(self.manager, name, None) is not None
                ]
                if present:
                    raise RuntimeError(
                        "Leave the existing FP4 graph/uploader composition intact; unified setup cannot adopt or replace "
                        + ", ".join(present)
                    )
                self.session_start = len(self.session.restore)
                self.manager_before = {
                    name: vars(self.manager).get(name, _ABSENT) for name in _DS_ATTRIBUTES
                }
                self.controller = self.deps["segmented"](
                    self.session, self.engine, shared_pool=True, ffn_back=True
                )
                self.session_tail = tuple(self.session.restore[self.session_start :])
                for owner, name, _ in self.session_tail:
                    self.session_installed[(id(owner), name)] = (
                        owner,
                        vars(owner).get(name, _ABSENT),
                    )
            else:
                # Important for stream-sensitive branch adapters: the native
                # helper's eager dummy initialization must use the engine stream.
                with self.deps["stream_context"](self.engine.stream):
                    self.native_recapture_record = self.deps["recapture"](
                        self.engine, bs=self.selection.native_batch_sizes
                    )
                self.wrapper_counts["native_recapture_calls"] += 1
                self.native_runner = self.engine.graph_runner
                self._check_native_graphs()
                native = self.native_runner.can_use_cuda_graph
                self.native_previous = vars(self.native_runner).get("can_use_cuda_graph", _ABSENT)

                def checked(batch):
                    self._owner()
                    if self.state in ("failed", "closed"):
                        raise RuntimeError("Unified decode execution is " + self.state)
                    result = native(batch)
                    if batch.is_decode and not result:
                        raise RuntimeError(
                            "Decode batch cannot use the declared native graph; eager fallback rejected"
                        )
                    return result

                self.native_guard = checked
                self.native_runner.can_use_cuda_graph = checked
            self.state = "prepared"
            return self.snapshot()
        except BaseException as error:
            self.pending_cleanup = getattr(error, "segmented_decode_cleanup_pending", ())
            self.state, self.failure = "failed", repr(error)
            raise

    def _check_native_graphs(self):
        runner = self.engine.graph_runner
        if not isinstance(runner, self.deps["native_type"]):
            raise RuntimeError("Native GraphRunner was replaced")
        if tuple(runner.graph_bs_list) != self.selection.native_batch_sizes:
            raise RuntimeError("Native graph batch-size capability changed")
        if set(runner.graph_map) != set(self.selection.native_batch_sizes):
            raise RuntimeError("Native recapture did not produce every declared graph")
        if self.native_runner is not None and runner is not self.native_runner:
            raise RuntimeError("Native GraphRunner owner changed after setup")

    @contextmanager
    def warmup(self):
        self._owner()
        if self.state != "prepared":
            raise RuntimeError("Call after_collector_installed before real warmup, exactly once")
        self.state = "warming"
        self.wrapper_counts["warmup_scopes"] += 1
        try:
            if self.controller is not None:
                with self.controller.warmup():
                    yield self
                if self.controller.snapshot().get("state") != "sealed":
                    raise RuntimeError("Underlying graph controller did not seal")
            else:
                yield self
                self._check_native_graphs()
            self.state = "sealed"
        except BaseException as error:
            self.state, self.failure = "failed", repr(error)
            raise

    def reset_stats(self):
        self._owner()
        if self.state not in ("prepared", "sealed"):
            raise RuntimeError("Reset observation only outside active warmup/failure")
        if self.selection.path == "paged_segments" and self.controller is not None:
            self.controller.reset_stats()
        elif self.selection.path == "deepseek_segments" and self.controller is not None:
            self.controller.attention.reset_stats()
            self.controller.front.reset_stats()  # Existing front also resets its back controller.
            self.controller.uploader.reset_stats()
        # Native GraphRunner exposes no execution counts; invent no counters and
        # never reset expert cache state just to satisfy a uniform observation API.

    def snapshot(self):
        result = dict(
            enabled=self.state == "sealed",
            state=self.state,
            failure=self.failure,
            path=self.selection.path,
            selection_reason=self.selection.reason,
            quantization=self.selection.quantization,
            selection_inputs="installed pager, quantization, native graph capability",
            wrapper_lifecycle_counts=dict(self.wrapper_counts),
            collector_attached=self.collector is not None,
            underlying_controller=None,
            actual_captured_graph_objects=None,
            execution_counters=None,
            no_implicit_eager_fallback=True,
            qualification="Facade CPU contracts do not replace per-adapter GPU numerical/performance qualification",
        )
        if self.controller is not None:
            snap = self.controller.snapshot()
            result["underlying_controller"] = snap
            if self.selection.path == "paged_segments":
                result["actual_captured_graph_objects"] = snap.get("segment_count")
                result["execution_counters"] = snap.get("counters")
                result["actual_pager_boundaries"] = snap.get("boundary_count")
            else:
                components = {
                    name: snap.get(name)
                    for name in ("attention_decode_graph", "ffn_front_graph", "ffn_back_graph")
                }
                counts = [
                    s.get("captured_instances", s.get("captured_blocks"))
                    for s in components.values()
                    if s is not None
                ]
                result["actual_captured_graph_objects"] = (
                    sum(counts) if len(counts) == 3 and all(c is not None for c in counts) else None
                )
                result["execution_counters"] = {
                    name: s.get("counters") for name, s in components.items() if s is not None
                }
                result["composition_note"] = (
                    "Attention, FFN and pinned route-map components share one explicit lifecycle"
                )
        elif self.selection.path == "native_graph":
            runner = self.engine.graph_runner
            result["actual_captured_graph_objects"] = len(getattr(runner, "graph_map", {}))
            result["native_graph_batch_sizes"] = list(getattr(runner, "graph_bs_list", ()))
            result["native_recapture"] = self.native_recapture_record
            result["execution_counter_availability"] = (
                "Native GraphRunner has no replay/capture-operation counters; null is intentional"
            )
            result["graph_ownership"] = (
                "Native graph objects remain engine-owned after facade close"
            )
        return result

    def _check_session_tail(self):
        actual = self.session.restore[self.session_start :]
        if len(actual) != len(self.session_tail) or any(
            a is not b for a, b in zip(actual, self.session_tail)
        ):
            raise RuntimeError("Session restoration journal changed; close later owners first")
        for (_, name), (owner, installed) in self.session_installed.items():
            if vars(owner).get(name, _ABSENT) is not installed:
                raise RuntimeError(
                    "Unknown outer binding on "
                    + name
                    + "; remove caller-owned observers before close"
                )

    def close(self):
        if self.state == "closed":
            return
        self._owner()
        if self.state == "warming":
            raise RuntimeError("Cannot close an active warmup")
        try:
            if self.selection.path == "deepseek_segments" and self.controller is not None:
                self._check_session_tail()
                self.controller.close()  # Failed drain keeps all hooks/resources for retry.
                for owner, name, previous in reversed(self.session_tail):
                    if previous is None:
                        if name in vars(owner):
                            delattr(owner, name)
                    else:
                        setattr(owner, name, previous)
                del self.session.restore[self.session_start :]
                for name, previous in self.manager_before.items():
                    if previous is _ABSENT:
                        if name in vars(self.manager):
                            delattr(self.manager, name)
                    else:
                        setattr(self.manager, name, previous)
            elif self.controller is not None:
                self.controller.close()
            elif self.native_runner is not None:
                if self.engine.graph_runner is not self.native_runner:
                    raise RuntimeError("Native runner changed; refusing foreign teardown")
                if (
                    vars(self.native_runner).get("can_use_cuda_graph", _ABSENT)
                    is not self.native_guard
                ):
                    raise RuntimeError("Native graph eligibility observer changed")
                self.engine.stream.synchronize()
                if self.native_previous is _ABSENT:
                    delattr(self.native_runner, "can_use_cuda_graph")
                else:
                    self.native_runner.can_use_cuda_graph = self.native_previous
            for resource in self.pending_cleanup:
                resource.close()
            self.pending_cleanup = ()
        except BaseException as error:
            self.state, self.failure = "failed", repr(error)
            raise
        if self.engine_previous is _ABSENT:
            delattr(self.engine, "tideway_decode_execution")
        else:
            self.engine.tideway_decode_execution = self.engine_previous
        self.state = "closed"


def enable_decode_execution(engine, manager=None, session=None):
    """Select only; call after_collector_installed(), then warmup(), explicitly."""
    return DecodeExecution(engine, manager, session)
