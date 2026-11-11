"""Original Gemma routing arithmetic with an additional private ID output.

Derivative of freetoken/kernel/triton/gemma4_fused.py. Its float operations,
sort order, launch geometry and output dtypes are unchanged. Only a second
int32 ID store replaces the later device-to-device clone in decode graphs.
"""

import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl

UPSTREAM_SHA256 = "693ee80511284d040ee70b98a1a3d748210273caff830a0ed95cd745130c226f"


def check_upstream():
    from freetoken.kernel.triton import gemma4_fused

    path = Path(gemma4_fused.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError("Unqualified original Gemma routing source")
    return dict(path=str(path), sha256=digest)


@triton.jit
def _routing_with_copy(
    logits_ptr,
    per_expert_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    private_ids_ptr,
    stride_logits_t,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E
    logits = tl.load(
        logits_ptr + token_id * stride_logits_t + offs_e, mask=valid, other=-float("inf")
    ).to(tl.float32)
    min_i32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ min_i32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    packed = ((key.to(tl.int64) & 0x00000000FFFFFFFF) << 32) | offs_e.to(tl.int64)
    sorted_packed = tl.sort(packed, descending=False)
    all_keys = ((sorted_packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_packed & 0x00000000FFFFFFFF).to(tl.int32)
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ min_i32)
    all_logits = all_bits.to(tl.float32, bitcast=True)
    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)
    denom = tl.sum(raw_exp, axis=0)
    weights = raw_exp / tl.where(denom > 0.0, denom, 1.0)
    scales = tl.load(per_expert_scale_ptr + all_ids.to(tl.int64), mask=top_mask, other=1.0).to(
        tl.float32
    )
    weights = weights * scales
    out_off = token_id * K + offs_e
    tl.store(topk_weights_ptr + out_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + out_off, all_ids, mask=top_mask)
    tl.store(private_ids_ptr + out_off, all_ids, mask=top_mask)


def routing_with_private_ids(logits, per_expert_scale, topk):
    if not (
        logits.is_cuda
        and logits.ndim == 2
        and per_expert_scale.is_cuda
        and per_expert_scale.ndim == 1
    ):
        raise ValueError("Original CUDA routing geometry required")
    tokens, experts = logits.shape
    if not (0 < topk <= experts <= 1024 and per_expert_scale.numel() == experts):
        raise ValueError("Invalid expert geometry")
    logits = logits.contiguous()
    per_expert_scale = per_expert_scale.contiguous()
    weights = torch.empty((tokens, topk), dtype=torch.float32, device=logits.device)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=logits.device)
    private = torch.empty_like(ids)
    if tokens:
        _routing_with_copy[(tokens,)](
            logits,
            per_expert_scale,
            weights,
            ids,
            private,
            logits.stride(0),
            E=experts,
            K=topk,
            BLOCK_E=triton.next_power_of_2(experts),
            num_warps=1,
        )
    return weights, ids, private
