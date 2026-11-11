"""Upload the already-decided slot map without a pageable temporary GPU tensor.

Only the final CPU-to-GPU index upload changes. Expert policy, weight DMA,
router readback and arithmetic remain unchanged. The receiving compute stream
orders its subsequent expert kernels after this upload and all earlier weight
copies. A completion event protects each pinned source buffer from CPU reuse.
"""

import time
import numpy as np
import torch


class PinnedRouteMapUploader:
    def __init__(self, capacity_hint=256, *, torch_api=None):
        if type(capacity_hint) is not int or capacity_hint < 1:
            raise ValueError("capacity_hint must be a positive integer")
        self.capacity_hint = capacity_hint
        self.torch = torch if torch_api is None else torch_api
        self.buffers = {}
        self.closed = False
        self.failure = None
        self.reset_stats()

    def reset_stats(self):
        # Completion events and buffers survive request-metric resets.
        self.stats = dict(
            uploads=0, index_bytes=0, buffer_allocations=0, reuse_waits=0, reuse_wait_s=0.0
        )

    def _drain(self, entry):
        try:
            needs_stream = entry.get("untracked", False)
            if needs_stream or (entry["pending"] and not entry["event"].query()):
                before = time.perf_counter()
                if needs_stream:
                    entry["stream"].synchronize()
                else:
                    entry["event"].synchronize()
                self.stats["reuse_waits"] += 1
                self.stats["reuse_wait_s"] += time.perf_counter() - before
        except BaseException as error:
            self.failure = f"{type(error).__name__}: {error}"
            raise
        entry["pending"] = False
        entry["untracked"] = False

    def copy_to(self, destination, mapped):
        if self.closed:
            raise RuntimeError("routing-map uploader is closed")
        if self.failure is not None:
            raise RuntimeError("routing-map uploader failed: " + self.failure)
        if destination.device.type != "cuda":
            raise ValueError("asynchronous routing maps require a CUDA destination")
        if destination.dtype not in (self.torch.int32, self.torch.int64):
            raise TypeError("routing-map destination must be int32 or int64")
        values = np.asarray(mapped)
        if tuple(values.shape) != tuple(destination.shape):
            raise ValueError("routing-map shape differs from destination")
        if values.dtype.kind not in "iu":
            raise TypeError("slot maps must contain integers")
        count = int(values.size)
        if not count:
            return
        key = (str(destination.device), destination.dtype)
        with self.torch.cuda.device(destination.device):
            entry = self.buffers.get(key)
            if entry is not None:
                self._drain(entry)
            if entry is None or entry["host"].numel() < count:
                host = self.torch.empty(
                    max(self.capacity_hint, count), dtype=destination.dtype, pin_memory=True
                )
                entry = dict(
                    host=host,
                    array=host.numpy(),
                    pending=False,
                    untracked=False,
                    event=self.torch.cuda.Event(enable_timing=False),
                )
                self.buffers[key] = entry
                self.stats["buffer_allocations"] += 1
            np.copyto(entry["array"][:count], values.reshape(-1), casting="unsafe")
            view = entry["host"][:count].view(tuple(destination.shape))
            stream = self.torch.cuda.current_stream(destination.device)
            entry["stream"] = stream
            entry["untracked"] = True
            # Record immediately after the copy, before CPU may overwrite its
            # pinned source. Normally the next router readback has already
            # completed it; the explicit event also protects stream changes.
            try:
                destination.copy_(view, non_blocking=True)
                entry["event"].record(stream)
            except BaseException as error:
                # A failed record cannot establish that the source is safe.
                # Block all later uploads even if a caller catches the error.
                self.failure = f"{type(error).__name__}: {error}"
                try:
                    stream.synchronize()
                    entry["untracked"] = False
                except BaseException as cleanup_error:
                    error.add_note(f"Routing-map copy drain failed: {cleanup_error!r}")
                raise
            entry["pending"] = True
            entry["untracked"] = False
            self.stats["uploads"] += 1
            self.stats["index_bytes"] += count * destination.element_size()

    def snapshot(self):
        return dict(
            mode="pinned_async",
            counters=dict(self.stats),
            pinned_bytes=sum(
                e["host"].numel() * e["host"].element_size() for e in self.buffers.values()
            ),
            extra_expert_payload_bytes=0,
            closed=self.closed,
            failure=self.failure,
            protects_host_reuse_with_completion_events=True,
        )

    def close(self):
        if self.closed:
            return
        for entry in self.buffers.values():
            self._drain(entry)
        self.closed = True


def configure_async_route_map(manager):
    """Enable asynchronous route maps before warmup on a bounded pager."""
    from .runtime import TidewayOffload

    if not isinstance(manager, TidewayOffload):
        raise TypeError("pinned routing-map upload requires bounded TidewayOffload")
    if getattr(manager, "route_map_uploader", None) is not None:
        raise RuntimeError("routing-map uploader already configured")
    manager.route_map_uploader = PinnedRouteMapUploader(int(manager.E))
    return manager.route_map_uploader
