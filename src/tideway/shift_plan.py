"""Plan and calibrate the activation arena and GPU expert capacity.

Derive the prefill chunk from KV and sliding-window limits, estimate initial
activation storage from model geometry, and fit expert slots to measured VRAM.
Calibrate the final layout with a synthetic request, shrink it on allocator
fallback, then clear request and expert caches before serving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import math
import os
import time

CHUNK_CANDIDATES = (8192, 16384, 32768)


@dataclass
class ShiftPlan:
    model: str
    chunk: int
    swa_pool_tokens: int | None
    swa_floor_tokens: int | None
    act_bytes_per_token: int
    row_bytes: int
    provisional_slots: int
    headroom_bytes: int
    free_before_fit: int | None = None
    slots: int | None = None
    arena_rows: int | None = None
    arena_bytes: int | None = None
    rebuild_s: float | None = None
    calibration: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    cold_cache_reset: dict | None = None

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ model-aware
def activation_bytes_per_token(mc) -> int:
    """Estimate BF16 prefill activation storage per token before calibration.

    Include aliased fused-MoE buffers, routing data, attention projections,
    recurrent-attention workspace, dense MLP intermediates and residuals.
    """
    H = int(mc.hidden_size)
    intermediate_size = int(getattr(mc, "moe_intermediate_size", 0) or 0)
    k = int(getattr(mc, "num_experts_per_tok", 0) or 0)
    E = int(getattr(mc, "num_experts", 0) or 0)
    moe = (
        k * (max(2 * intermediate_size, H) + intermediate_size) * 2 + E * 4 + k * 8
        if (intermediate_size and k)
        else 0
    )
    attn = 0
    for g in getattr(mc, "attention_groups", None) or ():
        kind = getattr(g, "kind", "full")
        if kind in ("full", "swa", "dsv4"):
            d = int(g.head_dim)
            attn = max(
                attn,
                (int(mc.num_qo_heads) * d + 2 * int(g.num_kv_heads) * d) * 2
                + int(mc.num_qo_heads) * d * 2,
            )
        elif kind == "linear_gated_delta":
            proj = (
                2 * int(g.num_key_heads) * int(g.key_head_dim)
                + 2 * int(g.num_value_heads) * int(g.value_head_dim)
            ) * 2
            state_ws = (
                int(g.num_value_heads) * int(g.key_head_dim) * int(g.value_head_dim) * 4 // 64
            )
            attn = max(attn, proj + state_ws)
    if attn == 0:
        d = int(mc.head_dim)
        attn = (int(mc.num_qo_heads) * d + 2 * int(mc.num_kv_heads) * d) * 2 + int(
            mc.num_qo_heads
        ) * d * 2
    dense = 0
    di = int(getattr(mc, "intermediate_size", 0) or 0)
    if di:
        dense += (2 * di + di) * 2
    si = int(getattr(mc, "shared_expert_intermediate_size", 0) or 0)
    if si:
        dense += (2 * si + si) * 2
    hidden = 6 * H * 2
    return int(moe + attn + dense + hidden)


def swa_constraints(config, kv_tokens: int):
    """Return sliding-window pool capacity and per-request floor, or (None, None)."""
    mc = config.model_config
    if getattr(mc, "swa_attention_group", None) is None or mc.swa_attention_group() is None:
        return None, None
    from freetoken.kvcache.hybrid_swa_pool import _swa_paged_num_tokens, _swa_per_req_swa_floor

    pages = max(1, kv_tokens // config.page_size)
    return int(_swa_paged_num_tokens(config, pages)), int(_swa_per_req_swa_floor(config))


def chunk_for(config, kv_tokens: int, cap: int = 32768):
    """Choose a prefill chunk that fits KV capacity and the sliding-window floor."""
    pool, floor = swa_constraints(config, kv_tokens)
    ok = []
    for c in CHUNK_CANDIDATES:
        if c > cap or c > kv_tokens:
            continue
        if pool is not None and c + floor > pool:
            continue
        ok.append(c)
    if not ok:
        raise ValueError(
            f"no prefill chunk in {CHUNK_CANDIDATES} fits (kv={kv_tokens}, swa pool={pool}, floor={floor})"
        )
    return max(ok), pool, floor


def planning_config(
    model_path: str,
    *,
    kv_tokens: int,
    max_context: int,
    swa_ratio: float | None,
    cache_type: str = "radix",
):
    """Build scheduler geometry without initializing an engine."""
    import torch
    from freetoken.distributed import DistributedInfo
    from freetoken.scheduler.config import SchedulerConfig

    kw = dict(
        model_path=str(model_path),
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        offline_mode=True,
        max_running_req=1,
        cache_type=cache_type,
        max_seq_len_override=max_context,
        num_token_override=kv_tokens,
        max_extend_tokens=8192,
    )
    if swa_ratio is not None:
        kw["swa_full_tokens_ratio"] = swa_ratio
    return SchedulerConfig(**kw)


# ------------------------------------------------------------------ hardware-aware
def arena_bytes_for(act_bytes_per_token: int, chunk: int, floor_bytes: int = 3 * 2**30) -> int:
    return max(floor_bytes, 2 * act_bytes_per_token * chunk + 512 * 2**20)


def fit_slots(
    *,
    free_bytes: int,
    headroom_bytes: int,
    row_bytes: int,
    provisional: int,
    total_experts: int,
    floor_slots: int,
) -> int:
    grow = (free_bytes - headroom_bytes) // row_bytes
    return int(max(floor_slots, min(total_experts, provisional + grow)))


def budget_slots(llm, headroom_bytes: int, floor_slots: int) -> tuple[int, dict]:
    """Compute the expert-slot limit accepted by the backend rebuild budget."""
    from freetoken.engine.cache_budget import expert_bytes_per_slot, net_cache_budget_bytes
    from freetoken.kvcache.linear_state_pool import state_pool_bytes

    eng = llm.engine
    cfg = eng.config
    baseline = int(eng._baseline_free)
    ratio = min(0.995, max(cfg.memory_ratio, 1.0 - headroom_bytes / baseline))
    cache_per_page, fixed, _, _ = type(eng.kv_cache).kv_cost(cfg)
    fixed += state_pool_bytes(cfg)
    budget = net_cache_budget_bytes(ratio, baseline, int(eng._weights_bytes), fixed)
    per_expert = expert_bytes_per_slot(eng.moe_offload_cache.bank_sources)
    kv_bytes = int(eng.num_pages) * cache_per_page
    slots = max(floor_slots, (budget - kv_bytes) // per_expert)
    return int(slots), dict(
        ratio=round(ratio, 4),
        baseline_GiB=round(baseline / 2**30, 3),
        weights_GiB=round(eng._weights_bytes / 2**30, 3),
        fixed_GiB=round(fixed / 2**30, 3),
        kv_GiB=round(kv_bytes / 2**30, 3),
        budget_GiB=round(budget / 2**30, 3),
        per_expert_MiB=round(per_expert / 2**20, 2),
    )


def free_vram() -> int:
    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return int(torch.cuda.mem_get_info()[0])


def rebuild_slots(llm, slots: int, headroom_bytes: int) -> float:
    """Rebuild expert capacity and recapture native decode graphs.

    Match the backend memory ratio to the requested headroom and intercept
    rebuild acknowledgements before the offline detokenizer receives them.
    """
    import torch
    from freetoken.message import CacheRebuildBackendMsg, CacheRebuildResultMsg

    st = getattr(llm.engine.moe_offload_cache, "shift", None)
    if st is not None:
        st.pool = None  # Release the old arena pool before reallocating expert banks.
    cfg = llm.engine.config
    baseline = int(getattr(llm.engine, "_baseline_free", 0) or 0)
    if baseline:
        ratio = min(0.995, max(cfg.memory_ratio, 1.0 - headroom_bytes / baseline))
        object.__setattr__(cfg, "memory_ratio", ratio)
        if llm.config is not cfg:
            object.__setattr__(llm.config, "memory_ratio", ratio)
    replies = []
    original_send = llm.send_result

    def send(messages):
        keep = []
        for m in messages:
            (replies if isinstance(m, CacheRebuildResultMsg) else keep).append(m)
        if keep:
            original_send(keep)

    llm.send_result = send
    torch.cuda.synchronize()
    began = time.monotonic()
    try:
        llm._process_one_msg(
            CacheRebuildBackendMsg(
                request_id="shift_plan", mode="if_idle", moe_cache_size=int(slots)
            )
        )
        if llm._pending_rebuild is not None:
            llm._execute_pending_rebuild()
        torch.cuda.synchronize()
    finally:
        llm.send_result = original_send
    status = [(r.status, r.error) for r in replies]
    got = llm.engine.moe_offload_cache.cache_size
    if got != slots:
        raise RuntimeError(
            f"rebuild to {slots} slots did not take effect (cache_size={got}); replies={status}"
        )
    return time.monotonic() - began


def synthetic_prompt(n: int, seed: int, vocab: int) -> list:
    """Generate deterministic calibration IDs while avoiding reserved token IDs."""
    import random

    r = random.Random(seed)
    lo, hi = 1000, max(2000, vocab - 1000)
    return [999] + [r.randrange(lo, hi) for _ in range(n - 1)]


def calibration_prefill(llm, chunk: int, seed: int = 7, decode_tokens: int = 8) -> dict:
    """Run a synthetic prefill and decode, returning arena and VRAM measurements."""
    import torch
    from freetoken.core import SamplingParams

    try:
        from freetoken.moe import shift as _shift
    except ImportError:
        _shift = None
    cache = llm.engine.moe_offload_cache
    vocab = int(llm.engine.config.model_config.vocab_size)
    ids = synthetic_prompt(chunk, seed, vocab)
    sp = SamplingParams(max_tokens=decode_tokens, temperature=0, top_k=1, ignore_eos=True)
    began = time.monotonic()
    llm.generate([ids], sp)
    torch.cuda.synchronize()
    wall = time.monotonic() - began
    st = getattr(cache, "shift", None)
    stats = _shift.arena_stats(st) if (st is not None and _shift is not None) else {}
    return dict(
        tokens=chunk,
        wall_s=round(wall, 3),
        arena=stats,
        arena_hw_rows={str(k): v for k, v in sorted(getattr(st, "arena_hw", {}).items())}
        if st
        else {},
        fallback_events=int(getattr(st, "arena_fallback_events", 0)) if st else None,
        gpu_free_GiB=round(torch.cuda.mem_get_info()[0] / 2**30, 3),
        gpu_peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
    )


def clear_caches(llm, *, require_success: bool = True) -> dict:
    """Clear KV, prefix and GPU expert caches before serving.

    Return all tree-owned sliding-window slots before rebuilding the prefix
    cache; rebuilding alone does not reclaim them in FreeToken 0.1.2.
    """
    import torch

    assert not llm.prefill_manager.runnable and not llm.decode_manager.runnable
    torch.cuda.synchronize()
    cm = llm.cache_manager
    result = dict(kv_cleared=True, swa_evicted=False, note=None)
    if getattr(cm, "is_swa", False) and getattr(cm, "swa_pool", None) is not None:
        cap = int(cm.swa_pool.swa_num_tokens) - 1
        cm.ensure_swa_slots(cap)
        result["swa_evicted"] = True
    try:
        cm.rebuild(cm.num_pages, cm.page_table)
        cm.check_integrity()
    except AssertionError as exc:  # leak the engine's own bookkeeping cannot undo
        result.update(kv_cleared=False, note=f"KV/radix not cleared: {exc}")
    llm.engine.moe_offload_cache.reset()
    torch.cuda.synchronize()
    if require_success and result["kv_cleared"] is not True:
        raise RuntimeError(f"cold KV reset failed; measurement is not permitted: {result}")
    return result


def _calibration_errors(cal, before_arena, chunk, headroom_bytes):
    """Validate the current round, not lifetime-cumulative allocator counters."""
    arena = cal.get("arena", {})
    errors = []
    for key in ("arena_fallback_allocs", "arena_fallback_fail"):
        before, after = before_arena.get(key), arena.get(key)
        delta = None if before is None or after is None else int(after) - int(before)
        cal[key + "_delta"] = delta
        if delta != 0:
            errors.append(f"{key} changed or is unverified: {delta}")
    if cal.get("fallback_events") != 0:
        errors.append("fallback events in current calibration")
    if arena.get("arena_fallback_in_use_MiB") != 0:
        errors.append("fallback storage remains live or is unverified")
    if arena.get("arena_active_outside_head") != 0:
        errors.append("live activation outside protected arena head or missing audit")
    if cal.get("tokens") != chunk:
        errors.append("calibration token count differs from the plan")
    free = cal.get("gpu_free_GiB")
    if free is None or not math.isfinite(free) or free * 2**30 < headroom_bytes * 0.25:
        errors.append("free VRAM below the calibration margin or unverified")
    return errors


def fit_and_calibrate(
    llm, plan: ShiftPlan, *, total_experts: int, floor_slots: int, log=print, max_rounds: int = 3
) -> ShiftPlan:
    """Fit GPU slots, rebuild and calibrate, reducing capacity on arena fallback."""
    if not isinstance(max_rounds, int) or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer")
    from freetoken.moe import shift as _shift

    _shift.configure(act_bytes_per_token=plan.act_bytes_per_token)
    free = free_vram()
    plan.free_before_fit = free
    slots = fit_slots(
        free_bytes=free,
        headroom_bytes=plan.headroom_bytes,
        row_bytes=plan.row_bytes,
        provisional=plan.provisional_slots,
        total_experts=total_experts,
        floor_slots=floor_slots,
    )
    by_budget, account = budget_slots(llm, plan.headroom_bytes, floor_slots)
    plan.notes.append(
        f"measured-fit {slots} slots; engine budget account {by_budget} slots ({account})"
    )
    log(f"[shift_plan] measured-fit {slots} vs engine-budget {by_budget} slots; account {account}")
    slots = min(slots, by_budget)
    calibrated = False
    for round_ in range(max_rounds):
        arena_rows = int(math.ceil(plan.arena_bytes / plan.row_bytes))
        arena_rows = min(arena_rows, int(0.4 * slots))
        _shift.configure(arena_rows=arena_rows)
        os.environ["TIDEWAY_SHIFT_ARENA"] = str(arena_rows)
        log(
            f"[shift_plan] round {round_}: free {free / 2**30:.2f} GiB -> slots {slots} (arena {arena_rows} rows = {arena_rows * plan.row_bytes / 2**30:.2f} GiB)"
        )
        plan.rebuild_s = rebuild_slots(llm, slots, plan.headroom_bytes)
        plan.slots, plan.arena_rows = slots, arena_rows
        st = llm.engine.moe_offload_cache.shift
        assert st is not None and st.arena == arena_rows, (getattr(st, "arena", None), arena_rows)
        before_arena = _shift.arena_stats(st)
        cal = calibration_prefill(llm, plan.chunk, seed=7 + round_)
        cal["round"] = round_
        cal["slots"] = slots
        plan.calibration.append(cal)
        fb_mib = float(cal["arena"].get("arena_fallback_in_use_MiB", 0) or 0)
        cal["validation_errors"] = _calibration_errors(
            cal, before_arena, plan.chunk, plan.headroom_bytes
        )
        cal["validated"] = not cal["validation_errors"]
        if cal["validated"]:
            calibrated = True
            log(
                f"[shift_plan] calibrated: chunk {plan.chunk} used {cal['arena'].get('arena_hw_rows')} rows, free after {cal['gpu_free_GiB']} GiB"
            )
            break
        if round_ + 1 == max_rounds:
            break  # Keep plan geometry at the LAST ACTUALLY BUILT configuration.
        # Convert fallback bytes and the safety margin into expert slots to release.
        extra_rows = int(math.ceil((fb_mib * 2**20 + plan.headroom_bytes / 2) / plan.row_bytes))
        plan.notes.append(
            f"round {round_}: fallback {fb_mib:.0f} MiB / free {cal['gpu_free_GiB']} GiB -> shrink {extra_rows} slots"
        )
        slots = max(floor_slots, slots - extra_rows)
        plan.arena_bytes = int(plan.arena_bytes * 1.25)
        free = free_vram()
    if not calibrated:
        message = (
            f"Shift calibration failed after {max_rounds} rounds; "
            f"last built slots={plan.slots}, arena_rows={plan.arena_rows}; "
            f"errors={plan.calibration[-1]['validation_errors']}"
        )
        plan.notes.append(message)
        raise RuntimeError(message)
    plan.cold_cache_reset = clear_caches(llm)
    plan.notes.append(f"clear_caches: {plan.cold_cache_reset}")
    if plan.cold_cache_reset.get("kv_cleared") is not True:
        raise RuntimeError(f"calibration passed but cold KV reset failed: {plan.cold_cache_reset}")
    return plan
