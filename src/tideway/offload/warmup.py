"""Low-tier warmup/cold-reset protocol; no CUDA work at import time.

This module does not reinstall managers, allocate new payload banks, or recreate CUDA graphs.
It is deliberately separate from full-host native warmup: mandatory native RAM
backing is model storage, whereas bounded RAM residency is an invalidatable cache.
"""

from __future__ import annotations

import time


def _kind(manager):
    kind = type(manager).__name__
    if kind not in ("TidewayOffload", "HybridOffload"):
        raise TypeError(f"No audited cold-reset protocol for {kind}")
    return kind


def _reset_numpy_cache(cache):
    cache.slots.fill(-1)
    cache.where.fill(-1)
    cache.usage.fill(0)
    cache.step = 0
    if hasattr(cache, "last_evicted"):
        cache.last_evicted = cache.last_evicted[:0]


def _reset_ram(ram):
    # Never erase lease evidence to manufacture a cold state. Async owners must
    # have completed and released every row first.
    if ram.leases:
        raise RuntimeError(f"cannot cold-reset leased RAM rows: {dict(ram.leases)}")
    for key in ("where", "lru", "duplicates", "anchors"):
        getattr(ram, key).clear()
    ram.free = list(range(ram.capacity - 1, -1, -1))
    ram.evictions = ram.duplicate_evictions = 0


def reset_expert_state(manager, synchronize):
    """Metadata-only invalidation after idle inference; CPU-testable with fakes.

    Hybrid keeps fixed slot assignments but marks EVERY core/RAM/mirror payload
    invalid. It must restart with a real prefill, as a freshly installed hybrid
    manager does. Clearing only cache.reset()/reset_lru() would retain warm data.
    """
    kind = _kind(manager)
    if kind == "TidewayOffload":
        manager._finish_prefetch()
        manager._release_upload_leases()
    synchronize()
    if kind in ("TidewayOffload", "HybridOffload"):
        if manager.ram.leases:
            raise RuntimeError("live RAM leases survived warmup drain")
    if kind == "TidewayOffload":
        _reset_ram(manager.ram)
        _reset_numpy_cache(manager.gpu)
        if manager.arena_lo is not None:
            manager.gpu.usage[manager.arena_lo : manager.arena_lo + manager.arena_head] = 1 << 62
        manager.ahead_ema.fill(2.0)
        manager.empty_tail = []
        manager._prefetch_layer, manager._prefetch_experts = None, ()
        manager._upload_leases = [(), ()]
        if manager._prefetch is not None and manager._prefetch.future is not None:
            raise RuntimeError("unconsumed SSD prefetch future survived warmup drain")
    else:
        _reset_ram(manager.ram)
        manager.core_loaded.fill(False)
        manager.ram_loaded.fill(False)
        manager.mirror_ready.clear()
        # A prepared fast path skips cold-load checks; retaining it would silently
        # reuse warmed payload even after loaded flags were cleared.
        manager.static_plan = [None] * manager.L
        manager.static_plan_used = 0
        manager.volatile_dirty = bool(len(manager.volatile_core))
        manager._pending = 0
        manager._rows.zero_()
        manager._init_tables()  # mutate graph-captured arrays in place
        manager.slot_cpu = None  # rebuilt by the first begin_prefill
        manager._prof_ev.clear()
        for key in manager.prof:
            manager.prof[key] = 0
    manager.in_prefill = False
    manager.ready, manager.released = [None, None], [False, False]
    mover = getattr(manager, "mover", None)
    if mover is not None:
        # Keep loaded batch-copy functions and all pointer/index buffers. They
        # are runtime/compiled state, not expert residency or learned routing.
        for key in ("launches", "dma_copies", "batch_calls", "batch_entries", "_ring"):
            if hasattr(mover, key):
                setattr(mover, key, 0)
    for key in ("trace", "trace_marks"):
        value = getattr(manager, key, None)
        if value is not None:
            value.clear()
    hist = getattr(manager, "prefill_hist", None)
    if hist is not None:
        hist.fill(0)
    manager.reset_stats()
    for key in ("bytes", "seconds", "thread_seconds", "batches", "batch_rows"):
        if hasattr(manager.reader, key):
            setattr(manager.reader, key, 0)
    synchronize()
    return expert_cold_proof(manager)


def expert_cold_proof(manager):
    kind = _kind(manager)
    if kind == "HybridOffload":
        proof = dict(
            manager=kind,
            valid_core_payload_rows=int(manager.core_loaded.sum()),
            valid_host_payload_rows=int(manager.ram_loaded.sum()),
            valid_mirror_payload_rows=len(manager.mirror_ready),
            prepared_prefill_plans=sum(p is not None for p in manager.static_plan),
            static_core_slot_assignments_retained=True,
            graph_tensor_addresses_rebound=False,
            ram_leases=len(manager.ram.leases),
        )
        cold = not any(
            proof[key]
            for key in (
                "valid_core_payload_rows",
                "valid_host_payload_rows",
                "valid_mirror_payload_rows",
                "prepared_prefill_plans",
                "ram_leases",
            )
        )
    else:
        where = manager.ram.where
        free = manager.ram.free
        proof = dict(
            manager=kind,
            valid_host_payload_rows=len(where),
            valid_gpu_payload_rows=int((manager.gpu.slots >= 0).sum()),
            valid_gpu_reverse_entries=int((manager.gpu.where >= 0).sum()),
            free_host_rows=len(free),
            host_capacity_rows=manager.capacity,
            ram_leases=len(manager.ram.leases),
            prefetch_inflight=manager._prefetch_layer is not None,
        )
        cold = (
            not proof["valid_host_payload_rows"]
            and not proof["valid_gpu_payload_rows"]
            and not proof["valid_gpu_reverse_entries"]
            and not proof["ram_leases"]
            and not proof["prefetch_inflight"]
            and len(free) == manager.capacity
            and set(free) == set(range(manager.capacity))
        )
    proof["logical_expert_cold"] = bool(cold)
    proof["expert_payload_bytes_zeroed"] = False
    if not cold:
        raise RuntimeError(f"incomplete expert cold reset: {proof}")
    return proof


def _addresses(llm, manager):
    cache, engine = manager.cache, llm.engine
    tensors = {"bank_" + name: tensor for name, tensor in cache.bank_caches.items()}
    for name in (
        "slot_for_id",
        "id_of_slot",
        "usage",
        "step",
        "src_indices",
        "evict_slots",
        "num_indices",
    ):
        value = getattr(cache, name, None)
        if value is not None:
            tensors[name] = value
    tensors["page_table"] = llm.cache_manager.page_table
    pool = engine.kv_cache
    for name in ("_kv_buffer", "full_to_swa_index_mapping"):
        value = getattr(pool, name, None)
        if value is not None:
            tensors["kv_" + name] = value
    for name in ("full_kv_pool", "swa_kv_pool"):
        value = getattr(pool, name, None)
        if value is not None:
            tensors["kv_" + name] = value.buffer
    if hasattr(manager, "storage"):
        tensors["host_storage"] = manager.storage
    if hasattr(manager, "ram"):
        tensors.update({"ram_" + name: bank.tensor for name, bank in manager.ram.banks.items()})
    # Hybrid graphs also capture manager-owned routing rows/counters and Mover
    # pointer arrays. Protect all direct tensor attributes on these owners,
    # including host index rings, without treating allocator free lists as graph
    # addresses (the latter are intentionally rebuilt).
    for prefix, owner in (
        ("cache", cache),
        ("manager", manager),
        ("mover", getattr(manager, "mover", None)),
        ("shift", getattr(cache, "shift", None)),
    ):
        if owner is not None:
            for name, value in vars(owner).items():
                if callable(getattr(value, "data_ptr", None)):
                    tensors[prefix + "_" + name] = value
    linear = getattr(engine, "linear_state_pool", None)
    if linear is not None:
        for name in ("conv_states", "recurrent_states"):
            value = getattr(linear, name, None)
            if value is not None:
                tensors["linear_" + name] = value
    token_pool = getattr(llm.table_manager, "token_pool", None)
    if token_pool is not None:
        tensors["token_pool"] = token_pool
    return {name: int(tensor.data_ptr()) for name, tensor in tensors.items()}


def validate_arena_after(record, *, arena_enabled):
    """Reject unsafe surviving workspace before measured expert reuse."""
    if not arena_enabled:
        return dict(checked=False)
    arena = record.get("arena", {})
    errors = [
        key
        for key in ("arena_active_outside_head", "arena_fallback_in_use_MiB")
        if arena.get(key) != 0
    ]
    if errors:
        raise RuntimeError(f"unsafe/unknown live arena workspace after warmup: {errors}; {arena}")
    return dict(
        checked=True, active_outside_head=0, fallback_live_MiB=0, sizing_high_water_retained=True
    )


def _idle(llm):
    if llm.prefill_manager.runnable or llm.decode_manager.runnable:
        raise RuntimeError("warmup cold reset requires an idle scheduler")
    table = llm.table_manager
    if sorted(table._free_slots) != list(range(table._max_running_reqs)):
        raise RuntimeError("request tables still owned or duplicated at warmup reset")


def reset_all_cold(llm, manager, *, carry_stats=None):
    """Reset radix/full/SWA/GDN AND all offload payload-validity metadata."""
    import torch
    from tideway.shift_plan import clear_caches

    _kind(manager)
    _idle(llm)
    pointers = _addresses(llm, manager)
    graph_runner = getattr(llm.engine, "graph_runner", None)
    graphs = {key: id(value) for key, value in getattr(graph_runner, "graph_map", {}).items()}
    torch.cuda.synchronize(llm.engine.device)
    # Drain I/O before native cache/reset hooks can invalidate copy sources.
    if hasattr(manager, "_finish_prefetch"):
        manager._finish_prefetch()
    if hasattr(manager, "_release_upload_leases"):
        manager._release_upload_leases()
    torch.cuda.synchronize(llm.engine.device)
    kv = clear_caches(llm, require_success=True)
    if kv.get("kv_cleared") is not True:
        raise RuntimeError(f"KV warmup reset did not succeed: {kv}")
    expert = reset_expert_state(manager, lambda: torch.cuda.synchronize(llm.engine.device))
    cm = llm.cache_manager
    cm.check_integrity()
    if len(cm.free_slots) != cm.num_pages:
        raise RuntimeError("FULL KV pages retained after warmup reset")
    swa_free = swa_capacity = gdn_free = gdn_capacity = None
    if getattr(cm, "swa_paged", False):
        swa = cm.swa_pool
        if swa.swa_available_size() != swa.swa_num_tokens - 1:
            raise RuntimeError("SWA tokens retained after warmup reset")
        swa_free, swa_capacity = swa.swa_available_size(), swa.swa_num_tokens - 1
    if getattr(cm, "is_hybrid", False):
        linear = cm.linear_state_pool
        if linear.num_free_slots != linear.num_slots - 1:
            raise RuntimeError("GDN snapshots retained after warmup reset")
        gdn_free, gdn_capacity = linear.num_free_slots, linear.num_slots - 1
    if carry_stats is not None:
        if hasattr(carry_stats, "pending_payloads"):
            carry_stats.pending_payloads.clear()
        for value in carry_stats.values():
            if isinstance(value, list):
                value.clear()
    # These are release/bookkeeping flags, not allocation objects or payload.
    cache = manager.cache
    if hasattr(cache, "_prefill_buffer_layer"):
        cache._prefill_buffer_layer = [None, None]
    if hasattr(cache, "_prefill_buffer_released"):
        cache._prefill_buffer_released = [True, True]
    if hasattr(cache, "_prefill_buffer_has_release_event"):
        cache._prefill_buffer_has_release_event = [False, False]
    cache.reset_stats()
    if hasattr(cache, "expert_recency"):
        cache.expert_recency.fill_(-1)
    if getattr(cache, "trace_pos", None) is not None:
        cache.trace_pos.zero_()
    st = getattr(cache, "shift", None)
    if st is not None:
        for name in ("hist", "score", "p_calls", "p_misses", "p_num"):
            value = getattr(st, name, None)
            if value is not None:
                value.zero_()
        for name in ("p_layers", "seeded_layers", "seeded_rows", "prefill_chunks"):
            if hasattr(st, name):
                setattr(st, name, 0)
        if getattr(st, "empty_seed", None) is not None:
            raise RuntimeError(
                "low-tier reset does not support native empty-seed state; use its native protocol"
            )
    torch.cuda.synchronize(llm.engine.device)
    if _addresses(llm, manager) != pointers:
        raise RuntimeError("warmup reset rebound graph/model storage addresses")
    if getattr(llm.engine, "graph_runner", None) is not graph_runner:
        raise RuntimeError("warmup reset replaced the CUDA graph runner")
    if {key: id(value) for key, value in getattr(graph_runner, "graph_map", {}).items()} != graphs:
        raise RuntimeError("warmup reset replaced captured CUDA graph objects")
    _idle(llm)
    return dict(
        kv=kv,
        expert=expert,
        full_pages_free=len(cm.free_slots),
        swa_slots_free=swa_free,
        swa_slots_capacity=swa_capacity,
        gdn_slots_free=gdn_free,
        gdn_slots_capacity=gdn_capacity,
        storage_addresses_stable=True,
        graphs_recaptured=0,
        physical_payload_zeroed=False,
        SSD_device_cache_flushed=False,
        retained="compiled kernels, graph objects, tensor allocations, arena sizing and read-only routing prior",
    )


def warmup_offload(llm, manager, *, chunk, seed=7, carry_stats=None):
    """Uniform prescribed low-tier startup, before Recorder attaches.

    Prelude512/4 with arena disabled is identical for every low-tier arm. Then
    configured-chunk/8 on the actual serving path. Every stage ends fully cold.
    No future workload, route values or output length enters this schedule.
    """
    import torch
    from tideway.shift_plan import calibration_prefill

    _kind(manager)
    _idle(llm)
    if chunk < 512:
        raise ValueError("this protocol requires configured chunk >= 512")
    if getattr(llm.engine, "elastic_kv", None) is not None:
        raise ValueError("low-tier warmup is separate from full-host native elastic installation")
    began = time.perf_counter()
    reports = []
    # Offline generate enters the scheduler stream; its loop owns engine-stream handoffs.
    with torch.inference_mode(), torch.cuda.stream(llm.stream):
        initial = reset_all_cold(llm, manager, carry_stats=carry_stats)
        for index, (tokens, decode, arena_disabled) in enumerate(
            ((512, 4, True), (chunk, 8, False))
        ):
            st = getattr(manager.cache, "shift", None)
            old_library = getattr(st, "arena_lib", None) if st is not None else None
            old_forward = llm.engine.forward_batch
            actual = []

            def observed(batch, sampling):
                if batch.is_prefill:
                    actual.append(int(batch.input_ids.numel()))
                return old_forward(batch, sampling)

            try:
                if arena_disabled and st is not None:
                    st.arena_lib = None
                llm.engine.forward_batch = observed
                record = calibration_prefill(llm, tokens, seed=seed + index, decode_tokens=decode)
            finally:
                llm.engine.forward_batch = old_forward
                if st is not None:
                    st.arena_lib = old_library
            record.update(
                stage=index,
                prescribed_prompt_tokens=tokens,
                prescribed_decode_tokens=decode,
                actual_prefill_chunks=actual,
                arena_disabled=arena_disabled,
            )
            if sum(actual) != tokens:
                raise RuntimeError(f"warmup prompt reused or actual prefill incomplete: {record}")
            record["arena_lifetime_check"] = validate_arena_after(
                record,
                arena_enabled=bool(
                    not arena_disabled and st is not None and getattr(st, "arena", 0) > 0
                ),
            )
            record["reset"] = reset_all_cold(llm, manager, carry_stats=carry_stats)
            reports.append(record)
    return dict(
        protocol="bounded_hybrid_prescribed_warmup_v1",
        seconds=time.perf_counter() - began,
        initial_reset=initial,
        stages=reports,
        model_forward_warmups=2,
        all_measured_expert_tiers_logically_cold=True,
        future_workload_consulted=False,
        claim="prescribed kernel/runtime warmup; not proof that every future shape has no JIT",
    )
