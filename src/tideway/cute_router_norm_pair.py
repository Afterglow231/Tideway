"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

RMSNorm CuTe DSL Kernels
========================

Includes:
- RMSNormKernel: Basic RMSNorm (also handles Gemma variant with weight_bias=1.0)
- QKRMSNormKernel: RMSNorm for 3D tensors [batch, heads, head_dim]
- RMSNormQuantKernel: RMSNorm + FP8 quantization
"""

# Tideway opt-in derivative: dual raw-weight RMSNorm outputs, with the
# router's two original BF16 multiplication boundaries preserved. The native
# RMSNormKernel geometry and F32 reduction are reused, not reimplemented.
import functools
import hashlib
import importlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import row_reduce_sum_multirow, predicate_k, get_sm_version

UPSTREAM_SHA256 = {
    "rmsnorm.py": "b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0",
    "utils.py": "3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a",
}


@functools.cache
def check_upstream():
    paths = [
        Path(importlib.import_module(name).__file__)
        for name in ("flashinfer.norm.kernels.rmsnorm", "flashinfer.norm.utils")
    ]
    result = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != UPSTREAM_SHA256[path.name]:
            raise RuntimeError("Unqualified CuTe RMSNorm source: " + str(path))
        result[str(path)] = digest
    return result


class RouterNormPairKernel(RMSNormKernel):
    """Native CuTe RMSNorm reduction; two outputs and original router casts."""

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mW2: cute.Tensor,
        mScale: cute.Tensor,
        mY2: cute.Tensor,
        M: Int64,
        eps: Float32,
        root: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(
            mX, mW, mY, mW2, mScale, mY2, M, eps, root, enable_pdl, tv_layout, tiler_mn
        ).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mW2: cute.Tensor,
        mScale: cute.Tensor,
        mY2: cute.Tensor,
        M: Int64,
        eps: Float32,
        root: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        # The native input load, F32 square/reduction/rsqrt and barrier above
        # are copied unchanged from FlashInfer. Load the additional raw routed
        # weight and router scale using precisely the same thread-value layout.
        mScale_layout = cute.prepend(mScale.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        gScale = cute.local_tile(
            cute.make_tensor(mScale.iterator, mScale_layout), tiler_mn, (0, cluster_y)
        )
        tWgScale = thr_copy_W.partition_S(gScale)
        tWrScale = cute.make_fragment_like(tWgScale)
        tXrScale = thr_copy_X.retile(tWrScale)
        cute.copy(copy_atom_sync, tWgScale, tWrScale, pred=tWpW)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))
        # Native norm -> BF16; tensor-scale multiply -> BF16; Python-scalar
        # multiply (ATen opmath scalar is F32) -> BF16. Do not fold these muls.
        h0 = y.to(mY.element_type).to(Float32)
        h1 = (h0 * tXrScale.load().to(Float32)).to(mY.element_type).to(Float32)
        router_y = h1 * root
        tXrO.store(router_y.to(mY.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # Load the routed weight after retiring the router epilogue so both
        # weight vectors and both FP32 outputs need not remain live together.
        mW2_layout = cute.prepend(mW2.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        gW2 = cute.local_tile(cute.make_tensor(mW2.iterator, mW2_layout), tiler_mn, (0, cluster_y))
        tWgW2 = thr_copy_W.partition_S(gW2)
        tWrW2 = cute.make_fragment_like(tWgW2)
        tXrW2 = thr_copy_X.retile(tWrW2)
        cute.copy(copy_atom_sync, tWgW2, tWrW2, pred=tWpW)
        routed_y = x * rstd * (tXrW2.load().to(Float32) + Float32(weight_bias))
        gY2 = cute.local_tile(mY2, tiler_mn, (bidx, cluster_y))
        tXgO2 = thr_copy_O.partition_D(gY2)
        tXrO2 = cute.make_fragment_like(tXgO2)
        tXrO2.store(routed_y.to(mY2.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO2, tXgO2, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H, enable_pdl, sm_version):
    check_upstream()
    dtype = cutlass.BFloat16
    kernel = RouterNormPairKernel(dtype, H, 0.0, sm_version=sm_version)
    # Match native symbolic row layout and alignment, including for M=1.
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * 2)

    def matrix():
        return cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
        )

    def vector():
        return cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        matrix(),
        vector(),
        matrix(),
        vector(),
        vector(),
        matrix(),
        Int64(1),
        Float32(1e-6),
        Float32(0.1),
        enable_pdl,
        stream,
        options="--enable-tvm-ffi",
    )


def router_norm_pair(x, ones, routed_weight, scale, eps, root, *, enable_pdl=None):
    """Return (router norm/scaled input, routed norm input), BF16 BS1 only.

    This is a new operator: raw-bit microprobe and whole-model qualification
    are required. Inference allocations are fresh, and neither input is changed.
    """
    from flashinfer.norm import _use_cuda_norm
    from flashinfer.utils import device_support_pdl

    if _use_cuda_norm(x.device) or torch.compiler.is_dynamo_compiling():
        raise RuntimeError("Norm pair requires the native CuTe dispatch")
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[0] != 1
        or not x.is_contiguous()
        or x.requires_grad
    ):
        raise ValueError("Norm pair requires contiguous CUDA BF16 decode BS1")
    H = x.shape[1]
    if H <= 0 or H % 8:
        raise ValueError("Norm pair requires an eight-element-aligned hidden width")
    for w in (ones, routed_weight, scale):
        if (
            w.device != x.device
            or w.dtype != x.dtype
            or tuple(w.shape) != (H,)
            or not w.is_contiguous()
            or w.requires_grad
            or w.data_ptr() % 16
        ):
            raise ValueError("Norm pair requires native aligned contiguous BF16 weights")
    if x.data_ptr() % math.gcd(128, H * 2):
        raise ValueError("Norm pair input does not meet the native compact alignment")
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(root):
        raise ValueError("Invalid norm epsilon or router scalar")
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(x.device)
    compiled = _compiled(H, bool(enable_pdl), get_sm_version(x.device))
    router_out, routed_out = torch.empty_like(x), torch.empty_like(x)
    compiled(x, ones, router_out, routed_weight, scale, routed_out, 1, eps, root)
    return router_out, routed_out
