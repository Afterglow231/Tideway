"""Bit-preserving two-bank host copies with byte-proportional block counts."""

import triton
import triton.language as tl


@triton.jit
def _copy(
    dst_ptrs,
    src_ptrs,
    features,
    dst_indices,
    src_indices,
    count,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    bank = tl.where(program < B0, 0, 1)
    block = tl.where(bank == 0, program, program - B0)
    blocks = tl.where(bank == 0, B0, B1)
    n = tl.load(count).to(tl.int32)
    if n > 0:
        width = (tl.load(features + bank) // 4).to(tl.int32)
        source = tl.load(src_ptrs + bank).to(tl.pointer_type(tl.uint32))
        target = tl.load(dst_ptrs + bank).to(tl.pointer_type(tl.uint32))
        for i in range(n):
            si = tl.load(src_indices + i).to(tl.int64)
            di = tl.load(dst_indices + i).to(tl.int64)
            for start in range(block * BLOCK, width, blocks * BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                value = tl.load(
                    source + si * width + offsets,
                    mask=offsets < width,
                    other=0,
                    cache_modifier=".cg",
                )
                tl.store(target + di * width + offsets, value, mask=offsets < width)


def balanced_copy(
    dst_ptrs,
    src_ptrs,
    features,
    dst_indices,
    src_indices,
    count,
    *,
    blocks=(16, 8),
    block=4096,
    warps=8,
):
    if dst_ptrs.numel() != 2 or src_ptrs.numel() != 2 or features.numel() != 2:
        raise ValueError("This opt-in prototype supports exactly two pinned BF16 banks")
    if (
        any(type(b) is not int or b <= 0 for b in blocks)
        or block not in (1024, 2048, 4096, 8192)
        or warps not in (4, 8)
    ):
        raise ValueError("Unsupported copy launch geometry")
    _copy[(sum(blocks),)](
        dst_ptrs,
        src_ptrs,
        features,
        dst_indices,
        src_indices,
        count,
        B0=blocks[0],
        B1=blocks[1],
        BLOCK=block,
        num_warps=warps,
    )
