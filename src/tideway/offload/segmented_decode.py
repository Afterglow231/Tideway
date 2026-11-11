"""Explicit, transactional setup of the bounded DeepSeek Decode combination."""

from contextlib import contextmanager

_ABSENT = object()
_ATTRS = (
    "route_map_uploader",
    "attention_decode_graph",
    "ffn_front_graph",
    "ffn_back_graph",
    "segmented_decode",
)


class SegmentedDecode:
    """Own the newly installed resources; Session retains binding restoration.

    Use ``warmup()`` around real, untimed engine warmup. No synthetic request is
    created here and no capture is allowed after all enabled controllers have sealed.
    ``close()`` drains only this combination; the caller then uninstalls its Session.
    """

    def __init__(self, session, uploader, attention, front, back=None):
        self.session = session
        self.uploader, self.attention, self.front = uploader, attention, front
        self.back = back
        self.state, self.failure = "installed", None

    def _fail(self, error):
        self.state, self.failure = "failed", f"{type(error).__name__}: {error}"
        # One controller may already have sealed when the other fails. Neither
        # may continue the request or silently fall back to eager afterwards.
        self.attention._fail(error)
        self.front._fail(error)
        if self.back is not None:
            self.back._fail(error)

    @contextmanager
    def warmup(self):
        if self.state != "installed":
            raise RuntimeError("segmented Decode warmup begins exactly once")
        try:
            self.attention.begin_warmup()
            self.front.begin_warmup()
            self.state = "warming"
            yield self
            self.attention.seal()
            self.front.seal()
            self.state = "sealed"
        except BaseException as error:
            self._fail(error)
            raise

    def snapshot(self):
        return dict(
            enabled=True,
            state=self.state,
            failure=self.failure,
            scope="bounded DeepSeek, explicit Decode BS=1; original Prefill bypass",
            route_map_upload=self.uploader.snapshot(),
            attention_decode_graph=self.attention.snapshot(),
            ffn_front_graph=self.front.snapshot(),
            ffn_back_enabled=self.back is not None,
            ffn_back_graph=self.back.snapshot() if self.back is not None else None,
        )

    def close(self):
        if self.state == "closed":
            return
        first = None
        # Front.close also owns the back dependency. Closing the explicit back
        # reference first additionally handles cleanup after Session restoration;
        # an already closed back is a no-op. Failed drains retain tensor owners.
        resources = (self.back, self.front, self.attention, self.uploader)
        for resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    first.add_note(f"Additional segmented Decode cleanup failed: {error!r}")
        # If a DMA drain failed, preserve a retryable close operation.
        if first is not None:
            self.state, self.failure = "failed", f"{type(first).__name__}: {first}"
            raise first
        self.state = "closed"


def _rollback(session, restore_start, attributes, resources, error):
    # Drain newly created resources even if one drain fails. Do not close the
    # existing manager, reader, pager, prefetcher or previous Session hooks.
    resources = list(resources)
    for name, previous in attributes.items():
        current = vars(session.manager).get(name, _ABSENT)
        if (
            current is not _ABSENT
            and current is not previous
            and hasattr(current, "close")
            and all(current is not resource for resource in resources)
        ):
            resources.append(current)
    pending = []
    for resource in reversed(resources):
        try:
            resource.close()
        except BaseException as cleanup_error:
            pending.append(resource)
            error.add_note(f"Segmented Decode cleanup failed: {cleanup_error!r}")
    if pending:
        # Keep any source buffers whose DMA drain failed alive and available
        # for an explicit cleanup retry without changing original manager attrs.
        error.segmented_decode_cleanup_pending = tuple(pending)
        error.add_note(
            "Retry close() on segmented_decode_cleanup_pending resources after resolving the cleanup failure."
        )
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
            error.add_note(f"Segmented Decode binding rollback failed ({name}): {rollback_error!r}")
    del session.restore[restore_start:]
    # A failed binding restoration remains available to Session.uninstall().
    session.restore.extend(reversed(failed))
    for name, previous in attributes.items():
        try:
            if previous is _ABSENT:
                if name in vars(session.manager):
                    delattr(session.manager, name)
            else:
                setattr(session.manager, name, previous)
        except BaseException as rollback_error:
            error.add_note(f"Segmented Decode manager rollback failed ({name}): {rollback_error!r}")


def enable_segmented_decode(session, engine, *, shared_pool=True, ffn_back=False):
    """Enable async slot maps, Attention prefix, then FFN front, explicitly.

    Only the bounded DeepSeek Tideway manager is supported. This does not alter
    cache capacities, phase-policy settings or serving configuration.
    Install before warmup, then use the returned object's ``warmup()`` context.
    ``ffn_back=True`` additionally captures native routed FP4 math and HC post after
    ordinary CPU ensure. It is opt-in and requires native alpha banks to be None.
    """
    from .runtime import TidewayOffload
    from .async_route_map import configure_async_route_map
    from .attention_graph import install_attention_decode_graph
    from .ffn_front_graph import install_ffn_front_graph

    manager = getattr(session, "manager", None)
    if not isinstance(manager, TidewayOffload) or manager.geo.quant != "ds_fp4":
        raise TypeError("segmented Decode requires the bounded DeepSeek Tideway manager")
    if not isinstance(getattr(session, "restore", None), list):
        raise TypeError("segmented Decode requires a Session with binding restoration")
    if type(shared_pool) is not bool:
        raise ValueError("shared_pool must be boolean")
    if type(ffn_back) is not bool:
        raise ValueError("ffn_back must be boolean")
    if getattr(getattr(engine, "device", None), "type", None) != "cuda":
        raise ValueError("segmented Decode requires a CUDA engine")
    if getattr(manager, "writeback_ahead", False):
        raise ValueError("segmented Decode currently requires writeback_ahead=False")
    if getattr(getattr(getattr(engine, "config", None), "tp_info", None), "size", 1) != 1:
        raise ValueError("segmented Decode requires one GPU")
    if getattr(getattr(engine, "graph_runner", None), "graph_bs_list", []):
        raise ValueError("segmented Decode cannot compose native full-model graphs")
    for name in (*_ATTRS, "pure_decode_graphs", "compressor_decode_graph", "decode_compute_graph"):
        if getattr(manager, name, None) is not None:
            raise ValueError(f"segmented Decode conflicts with existing {name}")

    restore_start = len(session.restore)
    attributes = {name: vars(manager).get(name, _ABSENT) for name in _ATTRS}
    resources = []
    try:
        uploader = configure_async_route_map(manager)
        resources.append(uploader)
        attention = install_attention_decode_graph(session, engine, shared_pool=shared_pool)
        resources.append(attention)
        front = install_ffn_front_graph(session, engine, shared_pool=shared_pool)
        resources.append(front)
        back = None
        if ffn_back:
            from .ffn_back_graph import install_ffn_back_graph

            back = install_ffn_back_graph(session, engine, shared_pool=shared_pool)
            resources.append(back)
        combined = SegmentedDecode(session, uploader, attention, front, back)
        manager.segmented_decode = combined
        return combined
    except BaseException as error:
        _rollback(session, restore_start, attributes, resources, error)
        raise
