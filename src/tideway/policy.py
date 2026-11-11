"""Capacity-aware selection of Tideway's implemented expert ownership paths.

Pure planning only: the runtime still owns allocation, safety fences and admission.
Machine RAM is never confused with expert payload capacity; the caller supplies
the measured/declared runtime reserve and actual GPU expert slots.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemoryPolicy:
    engine: str
    host_rows: int
    total_experts: int
    gpu_slots: int
    core_rows: int
    decode_lru_slots: int
    volatile_core_rows: int
    stage_resident_without_spill: bool
    reason: str

    def to_dict(self):
        return asdict(self)


def choose_memory_policy(
    *,
    total_experts,
    host_rows,
    gpu_slots,
    experts_per_layer,
    prior_available=False,
    arena_rows=0,
    minimum_lru_slots=16,
):
    """Select a valid implemented path without access to future requests.

    Hybrid must leave a routing working set in addition to the static GPU core.
    Its volatile cores are supported explicitly, not called all-phase residency.
    Arbitrary activation arenas are not yet composed with hybrid core pinning.
    A missing independent prior falls back to the online bounded pager.
    """
    if min(total_experts, host_rows, gpu_slots, experts_per_layer) <= 0:
        raise ValueError("expert counts and capacities must be positive")
    if total_experts % experts_per_layer:
        raise ValueError("expert pool must contain complete layers")
    if arena_rows < 0 or minimum_lru_slots < 1:
        raise ValueError("invalid arena or minimum LRU capacity")
    if gpu_slots < 2 * experts_per_layer + 1:
        raise ValueError("streamed prefill needs two layers of buffers and a cache tail")
    h = min(host_rows, total_experts)
    core = total_experts - h
    lru = gpu_slots - core
    volatile = max(0, core - (gpu_slots - 2 * experts_per_layer))
    if h == total_experts:
        engine, reason = (
            "tideway_shift",
            "Full expert backing fits in RAM; native graph path with elastic VRAM.",
        )
    elif lru >= minimum_lru_slots and prior_available and arena_rows == 0:
        engine, reason = (
            "tideway_hybrid",
            "Fixed RAM rows and GPU core cover the pool; decode remains device controlled.",
        )
    else:
        engine = "tideway"
        if lru < minimum_lru_slots:
            reason = "Joint capacity cannot cover the pool with the required decode LRU reserve."
        elif arena_rows:
            reason = "Activation arena is supported by the bounded pager; hybrid arena ownership is not enabled."
        else:
            reason = "No independent routing prior supplied; use the online bounded pager."
    return MemoryPolicy(
        engine,
        h,
        total_experts,
        gpu_slots,
        core,
        max(0, lru),
        volatile,
        h + gpu_slots - 2 * experts_per_layer - arena_rows >= total_experts,
        reason,
    )
