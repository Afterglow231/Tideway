# Copyright (c) 2025 by FlashInfer team. Licensed under Apache-2.0.
# Derivative of FlashInfer RMSNormKernel: unchanged row arithmetic/layout;
# only CTA-to-row-group mapping and the three original weight pointers differ.
import functools
import hashlib
import math
from pathlib import Path
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import row_reduce_sum_multirow, predicate_k, get_sm_version

UPSTREAM_SHA256 = "b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0"


class GroupedQKVNormKernel(RMSNormKernel):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mWQ: cute.Tensor,
        mWK: cute.Tensor,
        mWV: cute.Tensor,
        mY: cute.Tensor,
        Q: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        eps: Float32,
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

        self.kernel(mX, mWQ, mWK, mWV, mY, Q, K, V, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[
                cute.ceil_div(Q, self.rows_per_block)
                + cute.ceil_div(K, self.rows_per_block)
                + cute.ceil_div(V, self.rows_per_block),
                cluster_n,
                1,
            ],
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
        mWQ: cute.Tensor,
        mWK: cute.Tensor,
        mWV: cute.Tensor,
        mY: cute.Tensor,
        Q: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        q_blocks = cute.ceil_div(Q, self.rows_per_block)
        k_blocks = cute.ceil_div(K, self.rows_per_block)
        mW = mWQ
        M = Int64(Q)
        row_base = Int64(0)
        if bidx < q_blocks:
            mW = mWQ
            M = Int64(Q)
            row_base = Int64(0)
        elif bidx < q_blocks + k_blocks:
            mW = mWK
            M = Int64(K)
            row_base = Int64(Q)
            bidx = bidx - q_blocks
        else:
            mW = mWV
            M = Int64(V)
            row_base = Int64(Q + K)
            bidx = bidx - q_blocks - k_blocks
        mX = cute.domain_offset((row_base, 0), mX)
        mY = cute.domain_offset((row_base, 0), mY)

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

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H, Q, K, V, pdl, sm):
    from flashinfer.norm.kernels import rmsnorm as original

    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("Unqualified original RMSNorm kernel")
    kernel = GroupedQKVNormKernel(cutlass.BFloat16, H, 0.0, sm_version=sm)
    if kernel.cluster_n != 1:
        raise ValueError("Grouped head norms require the original single-CTA row reduction")
    symbol = cute.sym_int(64)

    def matrix():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (symbol, H), stride_order=(1, 0), assumed_align=math.gcd(128, H * 2)
        )

    def vector():
        return cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (H,), assumed_align=16)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        matrix(),
        vector(),
        vector(),
        vector(),
        matrix(),
        Q,
        K,
        V,
        Float32(1e-6),
        pdl,
        stream,
        options="--enable-tvm-ffi",
    )


def grouped_qkv_norm(x, wq, wk, wv, q_heads, kv_heads, eps):
    from flashinfer.norm import _use_cuda_norm
    from flashinfer.utils import device_support_pdl

    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_contiguous()
        or x.shape[0] != q_heads + 2 * kv_heads
        or x.shape[1] not in (256, 512)
        or x.requires_grad
        or _use_cuda_norm(x.device)
        or torch.compiler.is_dynamo_compiling()
    ):
        raise ValueError("Native contiguous CUDA BF16 decode head geometry required")
    H = x.shape[1]
    for w in (wq, wk, wv):
        if (
            w.device != x.device
            or w.dtype != x.dtype
            or tuple(w.shape) != (H,)
            or not w.is_contiguous()
            or w.requires_grad
            or w.data_ptr() % 16
        ):
            raise ValueError("Original aligned BF16 head weights required")
    if q_heads <= 0 or kv_heads <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid head counts/epsilon")
    output = torch.empty_like(x)
    compiled = _compiled(
        H, q_heads, kv_heads, kv_heads, bool(device_support_pdl(x.device)), get_sm_version(x.device)
    )
    compiled(x, wq, wk, wv, output, eps)
    return output
