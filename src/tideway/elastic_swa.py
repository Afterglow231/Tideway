"""SWA relocation kernels. Imported lazily only by the opt-in CUDA controller."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["MOVES"])
def _copy_disjoint_slots(
    KV,
    Source,
    Destination,
    MOVES,
    STRIPE_ELEMENTS: tl.constexpr,
    UNIT_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The virtual shape/strides never change. Only explicitly live source rows
    # and mapped low holes are touched, not the unmapped gaps between stripes.
    stripe = tl.program_id(1).to(tl.int64)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    move, element = offset // UNIT_ELEMENTS, offset % UNIT_ELEMENTS
    valid = move < MOVES
    source = tl.load(Source + move, valid, 0).to(tl.int64)
    destination = tl.load(Destination + move, valid, 0).to(tl.int64)
    base = stripe * STRIPE_ELEMENTS
    value = tl.load(KV + base + source * UNIT_ELEMENTS + element, valid, 0)
    tl.store(KV + base + destination * UNIT_ELEMENTS + element, value, valid)


def copy_disjoint_swa_slots(buffer, source, destination):
    """Copy a validated disjoint plan without allocating a KV scratch tensor.

    Caller fences earlier attention before this call and publishes the canonical
    full->SWA mapping afterwards on the same stream. Shrinking physical backing
    additionally requires waiting for this copy to complete.
    """
    import math

    if source.numel() != destination.numel():
        raise ValueError("SWA copy source/destination count differs")
    if not buffer.is_contiguous() or buffer.ndim != 6 or buffer.shape[3] != 1:
        raise ValueError("SWA relocation requires the original contiguous token-paged layout")
    moves = source.numel()
    if not moves:
        return
    unit = math.prod(buffer.shape[3:])
    stripes = buffer.shape[0] * buffer.shape[1]
    _copy_disjoint_slots[(triton.cdiv(moves * unit, 1024), stripes)](
        buffer, source, destination, moves, buffer.shape[2] * unit, unit, 1024
    )


def warmup_swa_copy(buffer):
    """Compile model geometry while idle, without reading uninitialized KV."""
    import math
    import torch

    index = torch.zeros(1, dtype=torch.int64, device=buffer.device)
    unit = math.prod(buffer.shape[3:])
    # Runtime MOVES=0 masks every load/store, yet compiles the same kernel used
    # for subsequent relocation counts (which are deliberately not constexpr).
    _copy_disjoint_slots[(1, buffer.shape[0] * buffer.shape[1])](
        buffer, index, index, 0, buffer.shape[2] * unit, unit, 1024
    )
