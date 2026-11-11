"""Minimal CUDA VMM + DLPack bridge; imports do not initialize or query CUDA."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from ._build import build_shared_library


class VMM:
    def __init__(self, device: int, chunk_bytes: int = 2 << 20):
        source = Path(__file__).with_suffix(".c")
        output = build_shared_library(source, cuda_headers=True, libraries=("-ldl", "-pthread"))
        self.lib = lib = ctypes.CDLL(str(output))
        lib.tideway_vmm_error.restype = ctypes.c_char_p
        lib.tideway_vmm_init.argtypes = [ctypes.c_int]
        lib.tideway_vmm_reserve.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t]
        lib.tideway_vmm_reserve.restype = ctypes.c_void_p
        lib.tideway_vmm_address.argtypes = [ctypes.c_void_p]
        lib.tideway_vmm_address.restype = ctypes.c_uint64
        lib.tideway_vmm_chunk.argtypes = [ctypes.c_void_p]
        lib.tideway_vmm_chunk.restype = ctypes.c_size_t
        lib.tideway_vmm_set.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.tideway_vmm_release.argtypes = [ctypes.c_void_p]
        lib.tideway_vmm_release.restype = None
        lib.tideway_vmm_dlpack.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.tideway_vmm_dlpack.restype = ctypes.c_void_p
        lib.tideway_vmm_dlpack_release.argtypes = [ctypes.c_void_p]
        lib.tideway_vmm_dlpack_release.restype = None
        lib.tideway_vmm_pool_create.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
        lib.tideway_vmm_pool_create.restype = ctypes.c_void_p
        lib.tideway_vmm_pool_release.argtypes = [ctypes.c_void_p]
        lib.tideway_vmm_pool_release.restype = None
        lib.tideway_vmm_pool_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.tideway_vmm_pool_stats.restype = None
        lib.tideway_vmm_reserve_pooled.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
        lib.tideway_vmm_reserve_pooled.restype = ctypes.c_void_p
        self.check(lib.tideway_vmm_init(device))
        self.device, self.chunk_bytes = device, chunk_bytes
        self.pool = None
        self._regions_created = 0

    def enable_pool(self, physical_budget_bytes):
        if self.pool:
            raise RuntimeError("VMM physical pool already enabled")
        if self._regions_created:
            raise RuntimeError("enable the physical pool before creating any VMM tensors")
        self.pool = self.lib.tideway_vmm_pool_create(
            self.device, self.chunk_bytes, int(physical_budget_bytes)
        )
        if not self.pool:
            raise RuntimeError(self.lib.tideway_vmm_error().decode())

    def pool_snapshot(self):
        if not self.pool:
            return None
        values = (ctypes.c_uint64 * 10)()
        self.lib.tideway_vmm_pool_stats(self.pool, values)
        names = (
            "budget_bytes",
            "physical_owned_bytes",
            "physical_mapped_bytes",
            "physical_idle_bytes",
            "create_calls",
            "release_calls",
            "map_calls",
            "unmap_calls",
            "access_calls",
            "handle_reuses",
        )
        return dict(zip(names, map(int, values)))

    def __del__(self):
        if getattr(self, "pool", None):
            self.lib.tideway_vmm_pool_release(self.pool)
            self.pool = None

    def check(self, result):
        if result < 0:
            raise RuntimeError(self.lib.tideway_vmm_error().decode())

    def tensor(self, shape, dtype):
        import torch

        codes = {
            torch.bfloat16: (4, 16),
            torch.float16: (2, 16),
            torch.float32: (2, 32),
            torch.int8: (0, 8),
            torch.uint8: (1, 8),
            torch.int32: (0, 32),
        }
        if dtype not in codes:
            raise ValueError(f"unsupported VMM tensor dtype {dtype}")
        code, bits = codes[dtype]
        size = math.prod(shape) * bits // 8
        handle = (
            self.lib.tideway_vmm_reserve_pooled(size, self.pool)
            if self.pool
            else self.lib.tideway_vmm_reserve(size, self.device, self.chunk_bytes)
        )
        if not handle:
            raise RuntimeError(self.lib.tideway_vmm_error().decode())
        self._regions_created += 1
        region = Region(self, handle, size)
        dimensions = (ctypes.c_int64 * len(shape))(*shape)
        managed = self.lib.tideway_vmm_dlpack(handle, len(shape), dimensions, code, bits)
        if not managed:
            raise MemoryError("DLPack descriptor allocation failed")
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule_new.restype = ctypes.py_object
        capsule = None
        try:
            capsule = capsule_new(managed, b"dltensor", None)
            tensor = torch.utils.dlpack.from_dlpack(capsule)
        except BaseException:
            valid = ctypes.pythonapi.PyCapsule_IsValid
            valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
            valid.restype = ctypes.c_int
            # A consumed capsule is named used_dltensor; torch then owns its
            # descriptor and invokes the deleter itself, including error cleanup.
            if capsule is None or valid(capsule, b"dltensor"):
                self.lib.tideway_vmm_dlpack_release(managed)
            raise
        assert tensor.data_ptr() == region.address and tensor.is_contiguous()
        return tensor, region


class Region:
    def __init__(self, owner: VMM, handle: int, nbytes: int):
        self.owner, self.handle, self.nbytes = owner, handle, nbytes
        self.address = int(owner.lib.tideway_vmm_address(handle))
        self.chunk = int(owner.lib.tideway_vmm_chunk(handle))
        self.mapped: set[int] = set()

    def set_chunks(self, chunks: set[int], *, release_only: bool = False):
        if any(i < 0 or i * self.chunk >= self.nbytes for i in chunks):
            raise ValueError("physical mapping outside tensor storage")
        for index in sorted(self.mapped - chunks):
            self.owner.check(self.owner.lib.tideway_vmm_set(self.handle, index, 0))
            self.mapped.remove(index)
        if not release_only:
            for index in sorted(chunks - self.mapped):
                self.owner.check(self.owner.lib.tideway_vmm_set(self.handle, index, 1))
                self.mapped.add(index)

    def __del__(self):
        if getattr(self, "handle", None):
            self.owner.lib.tideway_vmm_release(self.handle)
            self.handle = None
