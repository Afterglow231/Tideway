"""Tideway-Shift: phase-aware expert-cache policy on top of FreeToken's offload cache.

Enabled with TIDEWAY_SHIFT=1 (see ``enabled()``). Everything here is additive; with the
flag off the surrounding code runs the stock FreeToken paths byte-for-byte.

Mechanisms
----------
B. Seed-on-stream-through: during prefill every expert of a layer passes through the
   whole-layer double buffer. After the layer's GEMMs (still before the buffer is released
   to the copy stream) the top-N experts of this layer -- ranked by the routing histogram
   of the current request -- are copied device-to-device from the buffer into a PINNED
   slot range of the unified slot cache. Zero PCIe bytes; the decode that follows starts
   warm, and the next prefill chunk's hit-D2D split serves those experts from VRAM.
C. Two-tier decode policy: the pinned range is invisible to LRU eviction (the LRU kernel
   below masks it out of the victim scan); the remaining slots keep FreeToken's LRU as a
   tail for the long-tail experts.

Slot layout (unchanged from FreeToken except for the pinned range)::

    [0, 2E)               prefill double buffer (borrowed; usage 0 -> evicted first)
    [2E, 2E + L*N)        pinned: layer l owns [2E + l*N, 2E + (l+1)*N)
    [2E + L*N, cache_size) LRU tail
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

_TRUE = ("1", "true", "yes", "on")


def enabled() -> bool:
    return os.getenv("TIDEWAY_SHIFT", "0").strip().lower() in _TRUE


def stats_enabled() -> bool:
    return os.getenv("TIDEWAY_SHIFT_STATS", "0").strip().lower() in _TRUE


def freq_enabled() -> bool:
    """Collect the per-layer decode routing histogram (for oracle / overlap diagnostics)."""
    return os.getenv("TIDEWAY_SHIFT_FREQ", "0").strip().lower() in _TRUE


def pin_enabled() -> bool:
    """TIDEWAY_SHIFT_PIN=0 -> seed the hot set into the cache but leave it evictable (LRU-managed)."""
    return os.getenv("TIDEWAY_SHIFT_PIN", "1").strip().lower() in _TRUE


def small_prefill_max() -> int:
    """P. Prefill batches with at most this many tokens skip whole-layer streaming and run
    through the slot cache (admit the used experts, fetch only the misses). 0 = off."""
    return int(os.getenv("TIDEWAY_SHIFT_SMALL_PREFILL", "0"))


# Programmatic overrides (tideway.shift_plan sets them from the model geometry and the
# measured VRAM); the TIDEWAY_SHIFT_* env vars remain the fallback for hand-tuned runs.
_OVERRIDES: dict = {}


def configure(**kw) -> None:
    """Set planner overrides: ``arena_rows`` (int, wins over TIDEWAY_SHIFT_ARENA) and
    ``act_bytes_per_token`` (int, the model-derived prior used before any high-water mark
    of a token bucket has been observed)."""
    _OVERRIDES.update({k: v for k, v in kw.items() if v is not None})


def arena_slots() -> int:
    """A (full version). The LAST k slots of the cache back a torch MemPool (pluggable
    allocator) that the whole prefill forward allocates from; during decode they are ordinary
    LRU slots. Supersedes TIDEWAY_SHIFT_ELASTIC (which only moved the MoE intermediates)."""
    if _OVERRIDES.get("arena_rows") is not None:
        return int(_OVERRIDES["arena_rows"])
    return int(os.getenv("TIDEWAY_SHIFT_ARENA", "0"))


def arena_sized() -> bool:
    """A v2: confine each prefill to the first rows of the arena its token count needs (learned
    high-water table with margin) and invalidate only those slots. 0 -> always the whole arena."""
    return os.getenv("TIDEWAY_SHIFT_ARENA_SIZED", "1").strip().lower() in _TRUE


def arena_head_rows() -> int:
    """A v2: rows at the start of the arena reserved for small (<= 2 MiB) allocator segments; never
    expert slots. The prefill forward keeps a few tiny tensors alive across calls -- they must not
    share a row with an LRU slot."""
    return max(1, int(os.getenv("TIDEWAY_SHIFT_ARENA_HEAD", "2")))


def prestage_layers() -> int:
    """Idle prestage: whole expert layers 2..2+K-1 installed into the LRU slot cache while the
    scheduler is idle (after the double-buffer pre-begin of layers 0/1). 0 = off."""
    return max(0, int(os.getenv("TIDEWAY_SHIFT_PRESTAGE_LAYERS", "0")))


@torch.no_grad()
def idle_prestage_layers(cache, first_layer: int, count: int) -> int:
    """Install every expert of layers [first_layer, first_layer+count) into LRU slots (misses fetched
    from the host banks), protecting the double buffer, the arena head rows and any pinned range.
    Returns the number of experts fetched."""
    st = cache.shift
    E = st.num_experts
    prot = st.protect
    prot.copy_(st.pinned)
    prot[: 2 * E] = 1
    if st.arena > 0:
        prot[st.elastic_lo : st.elastic_lo + st.arena_head] = 1
    if st.pin_hard and st.pin_total > 0:
        prot[st.pin_base : st.pin_base + st.pin_total] = 1
    ids = torch.arange(E, dtype=torch.int32, device=cache.device)
    fetched = torch.zeros((1,), dtype=torch.int64, device=cache.device)
    for layer in range(first_layer, min(first_layer + count, st.num_layers)):
        cache._pending_src_layer = layer
        for i in range(0, E, 64):
            q = ids[i : i + 64].clone()  # the kernel rewrites the query in place
            lru_ensure_pinned(
                q,
                cache.slot_for_id.view(-1),
                cache.id_of_slot,
                cache.usage,
                cache.step,
                prot,
                q,
                cache.src_indices,
                cache.evict_slots,
                cache.num_indices,
                stats=None,
                id_base=layer * E,
            )
            fetched += cache.num_indices
            cache.copy_missing()
    n = int(fetched.item())
    st.prestage_fetched += n
    st.prestage_calls += 1
    return n


def elastic_slots() -> int:
    """A. Elastic region: the LAST k slots of the cache are lent to the prefill MoE workspace
    (fused_experts_impl intermediates) and are ordinary LRU slots during decode."""
    return int(os.getenv("TIDEWAY_SHIFT_ELASTIC", "0"))


def pin_per_layer(cache_size: int, num_experts: int, num_layers: int, elastic: int = 0) -> int:
    """Pinned experts per layer. Default: 75% of the hit region (slots >= 2E), equal share."""
    v = os.getenv("TIDEWAY_SHIFT_PIN_PER_LAYER", "").strip()
    hit_region = cache_size - 2 * num_experts - elastic
    if v:
        n = int(v)
    else:
        frac = float(os.getenv("TIDEWAY_SHIFT_PIN_FRACTION", "0.75"))
        n = int(math.floor(frac * hit_region / num_layers))
    # keep an LRU tail of at least 8 * (misses per step) slots so eviction never runs dry
    n = max(0, min(n, (hit_region - 64) // num_layers))
    return n


def hist_decay() -> float:
    return float(os.getenv("TIDEWAY_SHIFT_HIST_DECAY", "0.5"))


def policy() -> str:
    """Decode eviction policy: 'lru' (FreeToken's, optionally with pinned slots) or 'lrfu'
    (exponentially decayed hit frequency; victim = lowest score among untouched, unpinned slots)."""
    return os.getenv("TIDEWAY_SHIFT_POLICY", "lru").strip().lower()


def score_decay() -> float:
    """LRFU: multiplicative decay applied to every slot's score on each layer call
    (40 calls per token). 0.999 -> half-life ~17 tokens; 0.9995 -> ~35 tokens."""
    return float(os.getenv("TIDEWAY_SHIFT_SCORE_DECAY", "0.999"))


# ----------------------------------------------------------------------------- state
class ShiftState:
    """Per-cache state: pinned mask, per-layer routing histogram, counters."""

    def __init__(self, cache) -> None:
        E, L, S = cache.num_experts, cache.num_layers, cache.cache_size
        # KV sharing appends expert slots beyond the existing activation arena.
        # That arena must keep its addresses even while the appended tail shrinks.
        base_slots = getattr(cache, "elastic_base_slots", S)
        self.num_experts, self.num_layers, self.cache_size = E, L, S
        self.small_prefill_max = small_prefill_max()
        self.p_calls = torch.zeros((1,), dtype=torch.int64, device=cache.device)
        self.p_misses = torch.zeros((1,), dtype=torch.int64, device=cache.device)
        self.p_layers = 0
        self.p_num = torch.zeros((1,), dtype=torch.int64, device=cache.device)
        self.arena = min(max(0, arena_slots()), max(0, base_slots - 2 * E - 64))
        self.elastic = (
            self.arena
            if self.arena > 0
            else min(max(0, elastic_slots()), max(0, base_slots - 2 * E - 64))
        )
        self.elastic_lo = base_slots - self.elastic
        self.pool = None
        self.arena_lib = None
        self.arena_prefills = 0
        # Arena ownership and per-token-bucket high-water marks.
        self.arena_alloc = None
        self.arena_feat: list[int] = []
        self.arena_head = arena_head_rows() if self.arena > 0 else 0
        self.arena_main_rows = max(
            0, self.arena - self.arena_head
        )  # rows [elastic_lo, elastic_lo + head) hold small segments only
        self.arena_bases: list[int] = []
        self.arena_row_bytes = 0
        self.arena_hw: dict[int, int] = {}  # token bucket -> max rows a prefill of that size needed
        self.arena_limit_rows = self.arena_main_rows
        self.arena_inval_hi = S
        self.arena_sized = arena_sized()
        self.arena_cur_tokens = 0
        self.arena_last: dict = {}
        self.arena_fallback_events = 0
        self.prestage_fetched = 0
        self.prestage_calls = 0
        self.pin_n = pin_per_layer(base_slots, E, L, self.elastic)
        self.pin_base = 2 * E
        self.pin_total = self.pin_n * L
        dev = cache.device
        self.pinned = torch.zeros((S,), dtype=torch.int32, device=dev)
        self.pin_hard = pin_enabled()
        if self.pin_n > 0 and self.pin_hard:
            self.pinned[self.pin_base : self.pin_base + self.pin_total] = 1
        if self.arena > 0:
            # head rows of the arena: hold the small allocator segments (incl. the few long-lived
            # tiny tensors of the prefill forward); never expert slots (masked out of the LRU scan)
            self.pinned[self.elastic_lo : self.elastic_lo + self.arena_head] = 1
        self.hist = torch.zeros((L, E), dtype=torch.float32, device=dev)
        self.policy = policy()
        self.score_decay = score_decay()
        # LRFU per-slot score (decayed hit count); unused under 'lru'
        self.score = torch.zeros((S,), dtype=torch.float32, device=dev)
        self.seeded_layers = 0
        self.seeded_rows = 0
        self.prefill_chunks = 0
        self._arange = torch.arange(self.pin_n, dtype=torch.int32, device=dev)
        # seed-only mode: LRU-install scratch (kernel outputs) and the per-call protect mask
        self.protect = torch.zeros((S,), dtype=torch.int32, device=dev)
        self.seed_src = torch.zeros((E,), dtype=torch.int32, device=dev)
        self.seed_dst = torch.zeros((E,), dtype=torch.int32, device=dev)
        self.seed_num = torch.zeros((1,), dtype=torch.int64, device=dev)
        from freetoken.moe.seed_empty import EmptySeedState, enabled as empty_seed_enabled

        self.empty_seed = EmptySeedState(cache, self) if empty_seed_enabled() else None

    def describe(self) -> str:
        return (
            f"policy={self.policy} arena={self.arena} elastic={self.elastic} small_prefill_max={self.small_prefill_max} pin_per_layer={self.pin_n} hard_pin={self.pin_hard} pinned_slots={self.pin_total} "
            f"lru_tail={self.cache_size - self.pin_base - self.pin_total} hit_region={self.cache_size - self.pin_base}"
        )


def attach(cache) -> ShiftState | None:
    """Create (or recreate after a rebuild) the shift state for ``cache``."""
    if not enabled() or cache.device.type != "cuda":
        cache.shift = None
        return None
    cache.shift = ShiftState(cache)
    from freetoken.moe import fused as _fused

    if cache.shift.arena > 0 and cache.banks:
        _fused.set_shift_workspace(None)
        _arena_setup(cache, cache.shift)
    elif cache.shift.elastic > 0 and cache.banks:
        lo = cache.shift.elastic_lo
        hi = lo + cache.shift.elastic
        _fused.set_shift_workspace(tuple(bank[lo:hi].reshape(-1) for _, bank in cache.banks))
    else:
        _fused.set_shift_workspace(None)
    try:
        from freetoken.utils import init_logger

        init_logger(__name__).info_rank0("[SHIFT] attached: " + cache.shift.describe())
    except Exception:  # noqa: BLE001
        pass
    return cache.shift


_ARENA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena")


def _arena_setup(cache, st) -> None:
    """Build (once) and load the arena allocator, register the bank tails as its regions,
    and create the MemPool the prefill forward will allocate from."""
    import ctypes
    from pathlib import Path

    from tideway._build import build_shared_library

    from torch.cuda.memory import CUDAPluggableAllocator
    from .snapshot_compat import install_snapshot_refcount_guard

    st.snapshot_refcount_compat = install_snapshot_refcount_guard()

    src = Path(_ARENA_DIR) / "tideway_arena.c"
    so = str(build_shared_library(src))
    lib = ctypes.CDLL(so)
    lib.tideway_arena_set.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.tideway_arena_limit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    lib.tideway_arena_reset_hw.argtypes = []
    lib.tideway_arena_stats.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    st.arena_feat = []
    for i, (_, bank) in enumerate(cache.banks):
        feat = bank.stride(0) * bank.element_size()
        # region i = rows [elastic_lo, S) of bank i; head = the first arena_head rows (small segments only)
        base = bank.data_ptr() + st.elastic_lo * feat
        lib.tideway_arena_set(i, base, st.arena * feat, st.arena_head * feat)
        st.arena_feat.append(feat)
        st.arena_bases.append(base)
    st.arena_row_bytes = sum(st.arena_feat)
    st.arena_lib = lib
    st.arena_alloc = CUDAPluggableAllocator(so, "tideway_alloc", "tideway_free")
    st.pool = None  # one MemPool per prefill (arena_prepare); dropping it returns its segments


def arena_stats(st) -> dict:
    import ctypes

    if st is None or st.arena_lib is None:
        return {}
    out = (ctypes.c_size_t * 16)()
    st.arena_lib.tideway_arena_stats(out)
    d = dict(
        arena_in_use_GiB=round(out[0] / 2**30, 2),
        arena_peak_GiB=round(out[1] / 2**30, 2),
        arena_allocs=int(out[2]),
        arena_frees=int(out[3]),
        arena_misses=int(out[4]),
        arena_fallback_allocs=int(out[7]),
        arena_fallback_in_use_MiB=round(out[5] / 2**20, 1),
        arena_fallback_fail=int(out[13]),
        arena_small_spill=int(out[12]),
        arena_fallback_events=st.arena_fallback_events,
        arena_limit_rows=st.arena_limit_rows,
        arena_main_rows=st.arena_main_rows,
        arena_sized=st.arena_sized,
        arena_hw_rows={str(k): v for k, v in sorted(st.arena_hw.items())},
        arena_last=st.arena_last,
    )
    try:
        if st.pool is not None:
            snap = st.pool.snapshot(include_traces=False)
            act = []
            outside = 0
            for seg in snap:
                addr = int(seg.get("address", 0))
                for b in seg.get("blocks", []):
                    if b.get("state") == "active_allocated":
                        act.append(b.get("size", 0))
                        in_head = any(
                            base <= addr < base + st.arena_head * feat
                            for base, feat in zip(st.arena_bases, st.arena_feat)
                        )
                        outside += 0 if in_head else 1
                    addr += int(b.get("size", 0))
            d["arena_active_blocks"] = len(act)
            d["arena_active_outside_head"] = (
                outside  # must stay 0: a live tensor in a main row would be clobbered by decode
            )
            d["arena_active_MiB"] = [round(x / 2**20, 2) for x in sorted(act, reverse=True)[:12]]
            d["arena_segments"] = len(snap)
    except Exception:  # noqa: BLE001
        pass
    return d


def _arena_bucket(n_tok: int) -> int:
    return 1 << max(8, math.ceil(math.log2(max(1, n_tok))))


def _arena_rows_estimate(st, n_tok: int) -> int:
    """Rows (of one expert slot = sum of the bank row sizes) a prefill of ``n_tok`` tokens may reserve,
    with margin. Learned per power-of-two token bucket from the arena high-water marks; unobserved
    buckets scale linearly from the nearest observed one (activations are linear in tokens), else a
    prior fitted to the measured 32K chunk (5.5 GiB reserved)."""
    b = _arena_bucket(n_tok)
    if b in st.arena_hw:
        rows = st.arena_hw[b] * 1.25 + 8
    else:
        obs = sorted(st.arena_hw)
        if obs:
            nb = min(obs, key=lambda x: abs(math.log2(x) - math.log2(b)))
            rows = st.arena_hw[nb] * (b / nb)
        else:
            per_tok = _OVERRIDES.get("act_bytes_per_token")
            if per_tok:
                # model-derived prior (tideway.shift_plan.activation_bytes_per_token) --
                # replaces the constant fitted to Qwen3.6's measured 32K chunk (5.6 GiB)
                rows = (0.25 * 2**30 + float(per_tok) * b) / max(1, st.arena_row_bytes)
            else:
                rows = (0.25 * 2**30 + 5.6 * 2**30 * (b / 32768)) / max(1, st.arena_row_bytes)
        rows = rows * 1.6 + 16
    lim = int(math.ceil(rows))
    if lim >= 0.6 * st.arena_main_rows:
        lim = st.arena_main_rows
    return max(1, min(lim, st.arena_main_rows))


def arena_prepare(cache, n_tok: int) -> None:
    """Called by the engine right before a prefill forward (outside the pool context).
    1. drop the previous prefill's MemPool -> its cached segments go back to the arena
       (torch only releases a pool's blocks once its use count is zero);
    2. confine this prefill to the first ``rows`` main rows (size-aware) and invalidate exactly
       those slots (+ the head row) -- their bytes are about to be overwritten;
    3. create the pool the forward allocates from."""
    st = cache.shift
    lib = st.arena_lib
    if st.pool is not None:
        st.pool = None
    torch.cuda.empty_cache()
    rows = _arena_rows_estimate(st, n_tok) if st.arena_sized else st.arena_main_rows
    for i, feat in enumerate(st.arena_feat):
        lib.tideway_arena_limit(i, (st.arena_head + rows) * feat)
    lib.tideway_arena_reset_hw()
    st.arena_limit_rows = rows
    st.arena_inval_hi = st.elastic_lo + st.arena_head + rows
    st.arena_cur_tokens = n_tok
    _invalidate_range(cache, st.elastic_lo, st.arena_inval_hi)
    st.pool = torch.cuda.MemPool(st.arena_alloc.allocator())


def arena_after(st) -> None:
    """Called by the engine right after the prefill forward: learn this size's high-water mark
    (rows touched in the arena + any cudaMalloc fallback, expressed in rows)."""
    import ctypes

    out = (ctypes.c_size_t * 16)()
    st.arena_lib.tideway_arena_stats(out)
    rows_used = 0
    for i, feat in enumerate(st.arena_feat):
        rows_used = max(rows_used, math.ceil(max(0, int(out[8 + i]) - st.arena_head * feat) / feat))
    fb_rows = math.ceil(int(out[6]) / max(1, st.arena_row_bytes))
    need = rows_used + fb_rows
    b = _arena_bucket(st.arena_cur_tokens)
    st.arena_hw[b] = max(st.arena_hw.get(b, 0), need)
    if int(out[6]) > 0:
        st.arena_fallback_events += 1
        try:
            from freetoken.utils import init_logger

            init_logger(__name__).info_rank0(
                f"[SHIFT] arena fallback: prefill {st.arena_cur_tokens} tok limit {st.arena_limit_rows} rows used {rows_used} rows + {int(out[6]) / 2**20:.0f} MiB cudaMalloc"
            )
        except Exception:  # noqa: BLE001
            pass
    st.arena_last = dict(
        tokens=st.arena_cur_tokens,
        limit_rows=st.arena_limit_rows,
        rows_used=rows_used,
        fallback_MiB=round(int(out[6]) / 2**20, 1),
    )
    st.arena_prefills += 1


def on_begin_prefill(cache) -> None:
    st = getattr(cache, "shift", None)
    if st is None:
        return
    st.prefill_chunks += 1
    if st.arena > 0 and st.arena_lib is not None:
        pass  # A v2: engine.arena_prepare() already released the pool and invalidated the rows this prefill may touch
    elif st.elastic > 0:
        _invalidate_range(cache, st.elastic_lo, st.elastic_lo + st.elastic)
    d = hist_decay()
    if d < 1.0:
        st.hist.mul_(d)
    if st.empty_seed is not None:
        st.empty_seed.prepare(cache, st)


@torch.no_grad()
def admit_prefill_experts(cache, layer_id: int, topk_ids: torch.Tensor) -> None:
    """P. Small prefill through the slot cache.

    1. (layer 0 of the batch) drop residents of buffer 1 = slots [E, 2E): it is this batch's
       dense staging area (position == expert id), so no live expert may sit there.
    2. Admit every expert used by the batch via the LRU kernel, 64 unique ids per call, with
       [0, 2E) (and pinned slots) protected from eviction; fetch the misses from the host
       banks with ``copy_missing`` (SM gather, as decode does).
    3. Gather the used experts device-to-device from their slots into buffer 1 rows.
    The caller then runs the stock whole-layer GEMM on buffer 1 (n = E, raw expert ids).
    """
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

    st = cache.shift
    E = st.num_experts
    if layer_id == 0:
        _invalidate_range(cache, E, 2 * E)
        cache._prefill_buffer_layer = [None, None]
        cache._prefill_buffer_released = [True, True]
    uniq = torch.unique(topk_ids.reshape(-1)).to(torch.int32).contiguous()
    n = int(uniq.numel())
    prot = st.protect
    prot.copy_(st.pinned)
    prot[: 2 * E] = 1
    if st.pin_hard and st.pin_total > 0:
        prot[st.pin_base : st.pin_base + st.pin_total] = 1
    cache._pending_src_layer = layer_id
    for i in range(0, n, 64):
        # clone: the kernel rewrites the query in place to slot ids (out aliases query); uniq must stay expert ids
        q = uniq[i : i + 64].clone()
        lru_ensure_pinned(
            q,
            cache.slot_for_id.view(-1),
            cache.id_of_slot,
            cache.usage,
            cache.step,
            prot,
            q,
            cache.src_indices,
            cache.evict_slots,
            cache.num_indices,
            stats=None,
            id_base=layer_id * E,
        )
        st.p_misses += cache.num_indices
        cache.copy_missing()
    slots = cache.slot_for_id[layer_id][uniq.long()].contiguous()
    dst = (uniq + E).contiguous()
    st.p_num.fill_(n)
    if cache._copy_fused_ok:
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs,
            cache._copy_dst_ptrs,
            cache._copy_feat_bytes,
            dst,
            slots,
            st.p_num,
            blocks_per_bank=64,
        )
    else:
        for _, bank_cache in cache.banks:
            bank_cache.index_copy_(0, dst.long(), bank_cache.index_select(0, slots.long()))
    st.p_calls += 1
    st.p_layers += 1


@torch.no_grad()
def _invalidate_range(cache, lo: int, hi: int) -> None:
    """Drop residency of slots [lo, hi) (their bytes are about to be reused)."""
    ids = cache.id_of_slot[lo:hi]
    m = ids >= 0
    if bool(m.any()):
        cache.slot_for_id.view(-1)[ids[m].long()] = -1
    ids.fill_(-1)
    cache.usage[lo:hi] = 0
    st = getattr(cache, "shift", None)
    if st is not None and st.policy == "lrfu":
        st.score[lo:hi] = 0.0


@torch.no_grad()
def _seed_layer_lru(cache, st, layer_id: int, buffer_id: int, top: torch.Tensor) -> None:
    """Seed-only mode: install top-N through the LRU kernel (hits untouched, misses get the
    coldest victims), then D2D the missing rows buffer -> victim slots.

    Protected from eviction for this call: the double buffer [0, 2E) (volatile during prefill),
    the elastic region, and slots holding experts of layers >= layer_id + 2 -- those layers'
    hit-D2D classification for this chunk is still pending and relies on the begin-of-chunk
    snapshot, so their residency must not change until their prefetch has been issued."""
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

    E = st.num_experts
    prot = st.protect
    prot.copy_(st.pinned)
    prot[: 2 * E] = 1
    if st.elastic > 0:
        prot[st.elastic_lo : st.elastic_lo + st.elastic] = 1
    lay = torch.div(cache.id_of_slot, E, rounding_mode="floor")
    prot.masked_fill_(lay >= layer_id + 2, 1)
    if st.pin_hard and st.pin_total > 0:
        prot[st.pin_base : st.pin_base + st.pin_total] = 1
    q = top.clone()  # kernel rewrites the query in place
    lru_ensure_pinned(
        q,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        prot,
        q,
        st.seed_src,
        st.seed_dst,
        st.seed_num,
        stats=None,
        id_base=layer_id * E,
    )
    if cache._copy_fused_ok:
        src_abs = (
            st.seed_src + buffer_id * E
        )  # buffer rows are the first 2E slots: row = b*E + expert
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs,
            cache._copy_dst_ptrs,
            cache._copy_feat_bytes,
            st.seed_dst,
            src_abs,
            st.seed_num,
            blocks_per_bank=64,
        )
    else:
        n = int(st.seed_num.item())
        if n:
            dst = st.seed_dst[:n].long()
            src = st.seed_src[:n].long()
            for buffer, (_, bank_cache) in zip(cache.prefill_bank_buffers, cache.banks):
                bank_cache.index_copy_(0, dst, buffer[buffer_id].index_select(0, src))
    if st.policy == "lrfu":
        n = int(st.seed_num.item())
        if n:
            st.score[st.seed_dst[:n].long()] = 1.0
    if os.getenv("TIDEWAY_SHIFT_VERIFY_SEED", "0") in _TRUE:
        n = int(st.seed_num.item())
        _verify_seed(cache, layer_id, buffer_id, st.seed_src[:n].clone(), st.seed_dst[:n].clone())


@torch.no_grad()
def seed_layer(cache, layer_id: int, topk_ids: torch.Tensor) -> None:
    """Accumulate this chunk's routing for ``layer_id`` and pin its top-N experts.

    Must run on the compute stream AFTER the layer's GEMMs (they read the buffer) and
    BEFORE ``release_prefill_layer`` (which lets the copy stream overwrite the buffer):
    the D2D gather below reads the buffer rows, and the release event recorded afterwards
    orders that read before the next refill.
    """
    st = getattr(cache, "shift", None)
    if st is None or not cache.prefill_bank_buffers:
        return
    if st.empty_seed is not None:
        st.empty_seed.seed_layer(cache, st, layer_id, topk_ids)
        return
    if st.pin_n == 0:
        return
    E, N = st.num_experts, st.pin_n
    buffer_id = layer_id % 2
    if cache._prefill_buffer_layer[buffer_id] != layer_id:
        return  # buffer no longer holds this layer (should not happen inside _prefill_routed)
    ids = topk_ids.reshape(-1)
    if ids.dtype != torch.int64:
        ids = ids.long()
    st.hist[layer_id] += torch.bincount(ids, minlength=E)[:E].float()
    top = torch.topk(st.hist[layer_id], N).indices.to(torch.int32)  # expert ids of this layer
    if not st.pin_hard:
        _seed_layer_lru(cache, st, layer_id, buffer_id, top)
        st.seeded_layers += 1
        st.seeded_rows += N
        return
    gid = (layer_id * E + top).to(torch.int32)  # flat ids (layer * E + expert)
    dst = (st.pin_base + layer_id * N + st._arange).to(torch.int32)
    flat_slot_for_id = cache.slot_for_id.view(-1)
    # 1) evict whatever currently occupies this layer's pinned range
    old = cache.id_of_slot[dst]
    om = old >= 0
    if bool(om.any()):
        flat_slot_for_id[old[om].long()] = -1
    # 2) if a chosen expert currently lives elsewhere (LRU tail / old pinned pos), free that slot
    cur = flat_slot_for_id[gid.long()]
    cm = cur >= 0
    if bool(cm.any()):
        cur_slots = cur[cm].long()
        cache.id_of_slot[cur_slots] = -1
        cache.usage[cur_slots] = 0
    # 3) D2D: buffer rows (position == expert id) -> pinned slots, every bank
    for buffer, (_, bank_cache) in zip(cache.prefill_bank_buffers, cache.banks):
        rows = buffer[buffer_id].index_select(0, top.long())
        bank_cache.index_copy_(0, dst.long(), rows)
    if os.getenv("TIDEWAY_SHIFT_VERIFY_SEED", "0") in _TRUE:
        _verify_seed(cache, layer_id, buffer_id, top, dst)
    # 4) bookkeeping: resident, owned by this layer's experts
    cache.id_of_slot[dst.long()] = gid
    flat_slot_for_id[gid.long()] = dst
    cache.usage[dst.long()] = cache.step  # scalar tensor broadcast
    if st.policy == "lrfu":
        # seed score from the prefill histogram (normalised to ~hits-per-token scale)
        st.score[dst.long()] = (
            st.hist[layer_id][top.long()] / st.hist[layer_id].sum().clamp(min=1) * 8.0 * 32.0
        ).float()
    st.seeded_layers += 1
    st.seeded_rows += N


# ----------------------------------------------------------------------------- LRU kernel with a pinned mask
# Vendored from flashlib.kernels.slot_cache.triton.lru_ensure (sequential-argmin strategy),
# plus one extra input: ``pinned_ptr`` ([num_cached] int32); pinned slots are never victims.


@triton.jit
def _phase1(
    query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K: tl.constexpr, id_base
):
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & kmask[:, None] & kmask[None, :]
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    tl.store(lru_usage_ptr + s, step, mask=hit)
    return q, kmask, miss, first, first_miss, rank, num_missing, tl.where(hit, s, -1)


@triton.jit
def _stats(stats_ptr, first, num_missing):
    si = tl.arange(0, 4)
    v = tl.where(si == 0, tl.sum(first.to(tl.int32)), tl.where(si == 1, num_missing, 1))
    tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=si < 3)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _lru_ensure_pinned_kernel(
    query_ptr,
    slot_of_id_ptr,
    id_of_slot_ptr,
    lru_usage_ptr,
    lru_step_ptr,
    pinned_ptr,
    out_ptr,
    src_ptr,
    dst_ptr,
    num_copy_ptr,
    stats_ptr,
    K,
    num_cached,
    id_base,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    USAGE_MAX: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K, id_base
    )

    if num_missing > 0:
        tl.debug_barrier()
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
        p = tl.load(pinned_ptr + c, mask=cmask, other=1)
        umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
        u = tl.where((u == step) | (~cmask) | (p != 0), umax, u)
        for i in tl.range(num_missing):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(lru_usage_ptr + victim, step)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            out = tl.where((rank == i) & miss, victim, out)
            u = tl.where(c == victim, umax, u)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base", "decay"])
def _lrfu_ensure_kernel(
    query_ptr,
    slot_of_id_ptr,
    id_of_slot_ptr,
    lru_usage_ptr,
    lru_step_ptr,
    pinned_ptr,
    score_ptr,
    out_ptr,
    src_ptr,
    dst_ptr,
    num_copy_ptr,
    stats_ptr,
    K,
    num_cached,
    id_base,
    decay,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    USAGE_MAX: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    """LRU bookkeeping as in flashlib, but victims are chosen by the lowest DECAYED HIT
    FREQUENCY (score) among slots not touched this call and not pinned. Every call decays
    all scores by ``decay`` and adds 1 to each hit slot; a freshly installed slot starts at 1."""
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K, id_base
    )
    # decay every score, then credit the hits (out holds the hit slots, -1 elsewhere)
    c = tl.arange(0, BLOCK_C)
    cmask = c < num_cached
    sc = tl.load(score_ptr + c, mask=cmask, other=0.0) * decay
    tl.store(score_ptr + c, sc, mask=cmask)
    tl.debug_barrier()
    hit_slots = out
    tl.atomic_add(score_ptr + hit_slots, 1.0, mask=kmask & (hit_slots >= 0) & first)
    if num_missing > 0:
        tl.debug_barrier()
        u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
        p = tl.load(pinned_ptr + c, mask=cmask, other=1)
        sc = tl.load(score_ptr + c, mask=cmask, other=0.0)
        inf = tl.full([BLOCK_C], float("inf"), tl.float32)
        key = tl.where((u == step) | (~cmask) | (p != 0), inf, sc)
        for i in tl.range(num_missing):
            victim = tl.argmin(key, axis=0).to(tl.int32)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(lru_usage_ptr + victim, step)
            tl.store(score_ptr + victim, 1.0)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            out = tl.where((rank == i) & miss, victim, out)
            key = tl.where(c == victim, inf, key)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


def lrfu_ensure(
    query,
    slot_of_id,
    id_of_slot,
    lru_usage,
    lru_step,
    pinned,
    score,
    decay,
    out_indices,
    src_indices,
    dst_indices,
    num_copy,
    stats=None,
    id_base=0,
) -> None:
    k = query.numel()
    num_cached = id_of_slot.numel()
    block_c = triton.next_power_of_2(num_cached)
    _lrfu_ensure_kernel[(1,)](
        query,
        slot_of_id,
        id_of_slot,
        lru_usage,
        lru_step,
        pinned,
        score,
        out_indices,
        src_indices,
        dst_indices,
        num_copy,
        stats,
        k,
        num_cached,
        id_base,
        decay,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(block_c),
    )


def _num_warps_for(block_c: int) -> int:
    warps = 8 if block_c >= 2048 else 4
    return max(warps, min(triton.next_power_of_2(max(block_c // 1024, 1)), 32))


def lru_ensure_pinned(
    query,
    slot_of_id,
    id_of_slot,
    lru_usage,
    lru_step,
    pinned,
    out_indices,
    src_indices,
    dst_indices,
    num_copy,
    stats=None,
    id_base=0,
) -> None:
    """flashlib ``lru_ensure`` (sequential strategy) with pinned slots excluded from eviction."""
    k = query.numel()
    num_cached = id_of_slot.numel()
    block_c = triton.next_power_of_2(num_cached)
    _lru_ensure_pinned_kernel[(1,)](
        query,
        slot_of_id,
        id_of_slot,
        lru_usage,
        lru_step,
        pinned,
        out_indices,
        src_indices,
        dst_indices,
        num_copy,
        stats,
        k,
        num_cached,
        id_base,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(block_c),
    )


def ensure_experts_pinned(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    st = cache.shift
    if st.policy == "lrfu":
        lrfu_ensure(
            expert_ids,
            cache.slot_for_id.view(-1),
            cache.id_of_slot,
            cache.usage,
            cache.step,
            st.pinned,
            st.score,
            st.score_decay,
            expert_ids,
            cache.src_indices,
            cache.evict_slots,
            cache.num_indices,
            stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
            id_base=layer_id * cache.num_experts,
        )
        return
    if not st.pin_hard and st.arena == 0:
        from flashlib.kernels.slot_cache import lru_ensure

        lru_ensure(
            expert_ids,
            cache.slot_for_id.view(-1),
            cache.id_of_slot,
            cache.usage,
            cache.step,
            expert_ids,
            cache.src_indices,
            cache.evict_slots,
            cache.num_indices,
            stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
            id_base=layer_id * cache.num_experts,
        )
        return
    lru_ensure_pinned(
        expert_ids,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        st.pinned,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


@torch.no_grad()
def _verify_seed(
    cache, layer_id: int, buffer_id: int, top: torch.Tensor, dst: torch.Tensor
) -> None:
    """Debug: right after the D2D seed, compare seeded rows with host rows; also check what the
    buffer row itself holds (is the buffer content for this layer intact?)."""
    from freetoken.utils import init_logger

    log = init_logger(__name__)
    torch.cuda.synchronize(cache.device)
    bad = []
    for j, (e, s_) in enumerate(zip(top.tolist(), dst.tolist())):
        for bi, (per_layer, bank) in enumerate(cache.banks):
            host = per_layer[layer_id][e].to(cache.device)
            slot_ok = torch.equal(host, bank[s_])
            buf_ok = torch.equal(host, cache.prefill_bank_buffers[bi][buffer_id][e])
            if not (slot_ok and buf_ok):
                who = None
                if not buf_ok and bi == 0:
                    # which (layer, expert) does the buffer row actually hold? scan this and neighbouring layers
                    for L2 in range(max(0, layer_id - 2), min(cache.num_layers, layer_id + 3)):
                        rows = per_layer[L2].to(cache.device)
                        eq = (
                            rows.view(cache.num_experts, -1)
                            == cache.prefill_bank_buffers[bi][buffer_id][e].view(1, -1)
                        ).all(dim=1)
                        hits = torch.nonzero(eq).flatten().tolist()
                        if hits:
                            who = (L2, hits[:3])
                            break
                bad.append(
                    dict(
                        expert=e,
                        slot=s_,
                        bank=bi,
                        slot_ok=slot_ok,
                        buf_ok=buf_ok,
                        buffer_row_holds=who,
                    )
                )
                break
    if bad:
        log.info_rank0(
            f"[SHIFT-VERIFY-SEED] layer={layer_id} buffer={buffer_id} bad={len(bad)}/{len(top)} {bad[:4]}"
        )


def verify_enabled() -> bool:
    return os.getenv("TIDEWAY_SHIFT_VERIFY", "0").strip().lower() in _TRUE


@torch.no_grad()
def verify_residency(cache) -> dict:
    """Debug: check slot-map bijection and compare every resident slot's bytes with its host row."""
    E = cache.num_experts
    ids = cache.id_of_slot.clone()
    sfi = cache.slot_for_id.view(-1).clone()
    s_idx = torch.nonzero(ids >= 0).flatten()
    rev_bad = (
        int((sfi[ids[s_idx].long()] != s_idx.to(torch.int32)).sum().item()) if s_idx.numel() else 0
    )
    e_idx = torch.nonzero(sfi >= 0).flatten()
    s2 = sfi[e_idx].long()
    fwd_bad = int((ids[s2] != e_idx.to(torch.int32)).sum().item()) if e_idx.numel() else 0
    mism = 0
    checked = 0
    bad = []
    for e, s in zip(e_idx.tolist(), s2.tolist()):
        if int(ids[s].item()) != e:
            continue
        layer, expert = divmod(e, E)
        ok = True
        for per_layer, bank in cache.banks:
            dev = per_layer[layer][expert].to(cache.device)
            if not torch.equal(dev, bank[s]):
                ok = False
                break
        checked += 1
        if not ok:
            mism += 1
            if len(bad) < 8:
                bad.append((layer, expert, s))
    torch.cuda.synchronize(cache.device)
    return dict(
        checked=checked,
        byte_mismatch=mism,
        rev_inconsistent=rev_bad,
        fwd_inconsistent=fwd_bad,
        resident_ids=int(e_idx.numel()),
        occupied_slots=int(s_idx.numel()),
        examples=bad,
    )


def stats_line(cache) -> str:
    """One-line JSON for the engine log (parsed by shift/scripts/run_ab.py)."""
    import json

    d = cache.decode_miss_stats()
    st = getattr(cache, "shift", None)
    if getattr(cache, "collect_decode_freq", False):
        freq = cache.decode_freq.float()  # [L, E] decode routing counts of the finished request(s)
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) > 0:
            sf, si = torch.sort(freq, dim=1, descending=True)
            cdf = torch.cumsum(sf, dim=1) / total.clamp(min=1).unsqueeze(1)
            d["decode_routed"] = int(total.sum().item())
            d["oracle_static_hit"] = {
                str(C): round(cdf[valid, C - 1].mean().item(), 4)
                for C in (25, 46, 62, 75, 100, 128)
            }
            d["experts_for_90pct"] = round(
                ((cdf < 0.9).sum(dim=1).float() + 1)[valid].mean().item(), 1
            )
            if st is not None and st.pin_n > 0:
                N = st.pin_n
                top_pre = torch.topk(st.hist, N, dim=1).indices  # prefill-histogram top-N per layer
                mass = torch.gather(freq, 1, top_pre).sum(dim=1) / total.clamp(min=1)
                d["prefill_topN_decode_mass"] = round(mass[valid].mean().item(), 4)
                top_dec = si[:, :N]
                inter = torch.tensor(
                    [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top_pre, top_dec)],
                    dtype=torch.float32,
                )
                d["prefill_topN_overlap"] = round((inter / N).mean().item(), 4)
    if st is not None and st.arena > 0:
        d.update(arena_stats(st))
        d["arena_prefills"] = st.arena_prefills
    d["prestage_issued"] = getattr(cache, "prefill_prestage_issued", 0)
    d["prestage_used"] = getattr(cache, "prefill_prestage_used", 0)
    if st is not None:
        d["prestage_layers"] = prestage_layers()
        d["prestage_fetched"] = st.prestage_fetched
        d["prestage_calls"] = st.prestage_calls
    if st is not None:
        d["small_prefill_layers"] = st.p_layers
        d["small_prefill_misses_per_layer"] = round(
            float(st.p_misses.item()) / max(1, st.p_layers), 2
        )
        st.p_layers = 0
        st.p_misses.zero_()
    try:
        from freetoken.moe import fused as _fused

        d["ws_uses"] = dict(_fused._SHIFT_WS_STATS)
        free_b, total_b = torch.cuda.mem_get_info(cache.device)
        d["gpu_peak_alloc_GiB"] = round(torch.cuda.max_memory_allocated(cache.device) / 2**30, 2)
        d["gpu_reserved_GiB"] = round(torch.cuda.memory_reserved(cache.device) / 2**30, 2)
        d["gpu_free_GiB"] = round(free_b / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    d.update(
        shift=bool(st is not None),
        elastic=(st.elastic if st else 0),
        hard_pin=(st.pin_hard if st else None),
        policy=(st.policy if st else "lru"),
        pin_per_layer=(st.pin_n if st else 0),
        seeded_layers=(st.seeded_layers if st else 0),
        prefill_chunks=(st.prefill_chunks if st else 0),
        cache_size=cache.cache_size,
    )
    if st is not None and st.empty_seed is not None:
        d.update(st.empty_seed.statistics())
    return "[SHIFT] " + json.dumps(d, sort_keys=True)
