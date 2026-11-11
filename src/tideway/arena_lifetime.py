"""Idle-only lifetime barrier before replacing Shift's borrowed expert banks.

PyTorch tensors allocated through the custom arena do not own the bank tensor's
storage. Drop the explicitly known transient caches and reject remaining users
before the original bank allocations or allocator registrations can disappear.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings


_FLA_INDEX_NAMES = ("prepare_lens", "prepare_chunk_indices", "prepare_chunk_offsets")
_FI_PLAN_FIELDS = (
    "_qo_indptr_buf",
    "_paged_kv_indptr_buf",
    "_paged_kv_indices_buf",
    "_paged_kv_last_page_len_buf",
)


def _attention_backends(backend, path="attention"):
    """Only documented backend composition fields; no arbitrary object mutation."""
    if backend is None:
        return
    yield path, backend
    for name in ("prefill_backend", "decode_backend", "full_backend", "swa_backend"):
        child = getattr(backend, name, None)
        if child is not None and child is not backend:
            yield from _attention_backends(child, path + "/" + name)


def clear_flashinfer_prefill_plans(backend, bank_ranges):
    """Release retiring-arena metadata from a non-graph FlashInfer prefill plan.

    FlashInfer's non-graph plan replaces these four buffers on every call. Its
    first plan executes inside the activation arena, so three CPU->GPU indptr /
    last-page conversions can outlive that arena through the wrapper fields.
    Workspaces, KV tensors, graph wrappers and all other state are untouched.
    """
    import torch

    targets, seen = [], set()
    for path, node in _attention_backends(backend):
        wrapper = getattr(node, "prefill_wrapper", None)
        if wrapper is None or id(wrapper) in seen:
            continue
        seen.add(id(wrapper))
        for field in _FI_PLAN_FIELDS:
            value = getattr(wrapper, field, None)
            if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                continue
            ptr = int(value.data_ptr())
            if not any(r["start"] <= ptr < r["end"] for r in bank_ranges):
                continue
            if (
                type(wrapper).__module__ != "flashinfer.prefill"
                or type(wrapper).__name__ != "BatchPrefillWithPagedKVCacheWrapper"
                or getattr(wrapper, "_use_cuda_graph", True)
            ):
                raise RuntimeError("refusing to clear unknown or graph-owned FlashInfer plan")
            targets.append(
                (
                    wrapper,
                    field,
                    dict(
                        path=path + "/prefill_wrapper/" + field,
                        pointer=ptr,
                        bytes=value.numel() * value.element_size(),
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                    ),
                )
            )
    for wrapper, field, _ in targets:
        setattr(wrapper, field, None)
    return [description for _, _, description in targets]


def _owner_paths(llm, pointers):
    """Describe Python owners without retaining tensors after this call."""
    import torch

    paths = {ptr: [] for ptr in pointers}
    seen = set()
    cache = llm.engine.moe_offload_cache

    def walk(value, path, depth=0):
        if id(value) in seen or depth > 12 or value is cache:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type == "cuda" and int(value.data_ptr()) in paths:
                paths[int(value.data_ptr())].append(path)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + "/" + str(key), depth + 1)
        elif isinstance(value, (list, tuple)):
            for key, item in enumerate(value):
                walk(item, path + "/" + str(key), depth + 1)
        elif hasattr(value, "__dict__") and type(value).__module__.startswith(
            ("freetoken", "tideway", "flashinfer")
        ):
            for key, item in vars(value).items():
                walk(item, path + "/" + key, depth + 1)

    for name in ("model", "attn_backend", "graph_runner"):
        walk(getattr(llm.engine, name, None), "engine/" + name)
    walk(getattr(llm, "_last_data", None), "scheduler/_last_data")
    module = sys.modules.get("freetoken.kernel.fla.index")
    if module is not None:
        for name in _FLA_INDEX_NAMES:
            wrapper = getattr(module, name)
            for key, cell in zip(wrapper.__code__.co_freevars, wrapper.__closure__ or ()):
                if key == "cache_entries":
                    walk(cell.cell_contents, "FLA/" + name)
    return paths


def _write_report(report):
    default = Path(tempfile.gettempdir()) / f"tideway-arena-lifetime-{os.getpid()}.json"
    destination = Path(os.environ.get("TIDEWAY_ARENA_LIFETIME_REPORT", str(default))).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(destination)
    destination.write_text(json.dumps(report, indent=2) + "\n")


def clear_fla_index_caches():
    """Clear only the three known identity-keyed FLA index caches, without import.

    The vendored decorator has no public cache_clear. Validate its named closure
    and wrapped function identity rather than walking or clearing arbitrary
    global state. Existing references to the wrappers themselves stay valid.
    """
    module = sys.modules.get("freetoken.kernel.fla.index")
    if module is None:
        return {}
    cells = []
    for name in _FLA_INDEX_NAMES:
        wrapper = getattr(module, name)
        wrapped = getattr(wrapper, "__wrapped__", None)
        if wrapped is None or wrapped.__module__ != module.__name__ or wrapped.__name__ != name:
            raise RuntimeError(f"unexpected FLA index wrapper identity: {name}")
        closure = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or ()))
        cell = closure.get("cache_entries")
        entries = None if cell is None else cell.cell_contents
        if not isinstance(entries, list) or any(
            not isinstance(entry, tuple)
            or len(entry) != 3
            or not isinstance(entry[0], tuple)
            or not isinstance(entry[1], dict)
            for entry in entries
        ):
            raise RuntimeError(f"unexpected FLA index cache structure: {name}")
        cells.append((name, entries))
    result = {name: len(entries) for name, entries in cells}
    for _, entries in cells:
        entries.clear()
    return result


def guard_arena_bank_rebuild(llm):
    """Call after graph teardown, before dropping old pool or old bank tensors.

    Fails closed if any borrowed tensor or allocator-active block remains. This
    may clear recomputable index caches even when it rejects; live KV, weights,
    GDN snapshots and requests are never mutated by this function.
    """
    import torch

    cache = llm.engine.moe_offload_cache
    st = cache.shift
    if st is None or st.arena <= 0:
        return dict(checked=False, cleared_fla_entries={}, active_blocks=[], borrowed_tensors=[])
    if llm.prefill_manager.runnable or llm.decode_manager.runnable:
        raise RuntimeError("arena bank rebuild requires an idle scheduler")
    ranges = [
        dict(
            name=name,
            start=int(bank.data_ptr()),
            end=int(bank.data_ptr()) + bank.numel() * bank.element_size(),
            storage_id=int(bank.untyped_storage()._cdata),
        )
        for name, bank in cache.bank_caches.items()
    ]

    def active_blocks():
        result = []
        if st.pool is not None:
            for segment in st.pool.snapshot(include_traces=False):
                address = int(segment["address"])
                for block in segment.get("blocks", ()):
                    if block.get("state", "").startswith("active"):
                        result.append(
                            dict(
                                address=address,
                                bytes=int(block.get("size", 0)),
                                state=block.get("state"),
                            )
                        )
                    address += int(block.get("size", 0))
        return result

    def borrowed_tensors():
        result = []
        with warnings.catch_warnings():
            # isinstance on a deprecated torch.distributed enum encountered by
            # GC can emit an irrelevant warning; the audit still examines it.
            warnings.simplefilter("ignore", FutureWarning)
            for value in gc.get_objects():
                if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                    continue
                ptr = int(value.data_ptr())
                for r in ranges:
                    if r["start"] <= ptr < r["end"]:
                        storage = int(value.untyped_storage()._cdata)
                        if storage != r["storage_id"]:
                            result.append(
                                dict(
                                    bank=r["name"],
                                    pointer=ptr,
                                    bytes=value.numel() * value.element_size(),
                                    shape=list(value.shape),
                                    dtype=str(value.dtype),
                                    storage_id=storage,
                                )
                            )
        owners = _owner_paths(llm, {r["pointer"] for r in result})
        for item in result:
            item["owner_paths"] = owners[item["pointer"]]
        return result

    torch.cuda.synchronize(llm.engine.device)
    before_active, before_borrowed = active_blocks(), borrowed_tensors()
    report = dict(
        checked=True,
        old_bank_ranges=ranges,
        before_active_blocks=before_active,
        before_borrowed_tensors=before_borrowed,
        status="before_clearing",
    )
    _write_report(report)
    try:
        cleared = clear_fla_index_caches()
        cleared_plans = clear_flashinfer_prefill_plans(
            getattr(llm.engine, "attn_backend", None), ranges
        )
    except Exception as exc:
        report.update(status="rejected", error=str(exc))
        _write_report(report)
        raise
    gc.collect()
    torch.cuda.synchronize(llm.engine.device)
    # Drain allocator pending frees while the OLD arena registration still owns
    # their addresses. A later callback must not free a newly registered block.
    torch.cuda.empty_cache()
    active, borrowed = active_blocks(), borrowed_tensors()
    report.update(
        cleared_fla_entries=cleared,
        cleared_flashinfer_plans=cleared_plans,
        active_blocks=active,
        borrowed_tensors=borrowed,
        status="rejected" if active or borrowed else "passed",
    )
    _write_report(report)
    if active or borrowed:
        raise RuntimeError(f"unsafe arena bank rebuild: surviving old-bank users: {report}")
    return report
