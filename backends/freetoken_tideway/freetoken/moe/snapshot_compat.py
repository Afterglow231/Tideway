"""Correct the borrowed-reference bug in PyTorch 2.11.0 CUDA snapshots.

THCPModule_memorySnapshot wraps PyTuple_GetItem's borrowed references in
THPObjectPtr without incrementing them first. Its local destructors therefore
consume one reference to each argument. Pool IDs above Python's immortal small
integer range can then be freed while the argument tuple still refers to them.

This Tideway-process compatibility hook supplies exactly the references that
the affected native binding consumes. It retains the native snapshot, its pool
filter, trace flag, and errors. Allocators and inference are unchanged.
"""

from __future__ import annotations

import ctypes
from functools import wraps
import sys
import threading


_LOCK = threading.Lock()
_INCREF = ctypes.pythonapi.Py_IncRef
_INCREF.argtypes = [ctypes.py_object]
_INCREF.restype = None


def install_snapshot_refcount_guard():
    import torch

    with _LOCK:
        native = torch._C._cuda_memorySnapshot
        existing = getattr(native, "_tideway_snapshot_refcount_guard", None)
        if existing is not None:
            return dict(existing)
        report = {
            "enabled": False,
            "torch_version": torch.__version__,
            "allocator_changed": False,
            "native_snapshot_retained": True,
        }
        if torch.__version__.split("+")[0] != "2.11.0":
            return report

        # Missing positive pool IDs return no segments and create no pool or
        # device allocation. Two Python owners keep these dynamic integers
        # alive while we inspect and restore any erroneous reference loss.
        values = (int("1048577"), int("1048579"))
        arguments = values + (False,)
        before = [sys.getrefcount(value) for value in arguments]
        try:
            native(arguments)
        finally:
            after = [sys.getrefcount(value) for value in arguments]
            loss = [a - b for a, b in zip(before, after)]
            for value, count in zip(arguments, loss):
                for _ in range(max(0, count)):
                    _INCREF(value)
        report["observed_borrowed_reference_loss"] = loss[:2]
        if loss == [0, 0, 0]:
            return report
        if loss[:2] != [1, 1] or loss[2] not in (0, 1):
            raise RuntimeError(f"Unexpected CUDA snapshot reference contract: {loss}")

        @wraps(native)
        def snapshot(arguments):
            if isinstance(arguments, tuple) and len(arguments) in (2, 3):
                # Both valid tuple branches construct all their THPObjectPtrs
                # before validating the elements. They consume these references
                # on both success and exception paths.
                for value in arguments:
                    _INCREF(value)
            return native(arguments)

        report["enabled"] = True
        snapshot._tideway_snapshot_refcount_guard = dict(report)
        torch._C._cuda_memorySnapshot = snapshot
        return dict(report)
