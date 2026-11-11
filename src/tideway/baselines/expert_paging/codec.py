"""Lossless independently decodable expert transport; CPU encoder, CPU/CUDA decoders."""

import ctypes
from pathlib import Path
import struct
import numpy as np

from tideway._build import build_shared_library
from ._native import build_cuda_decoder

HERE = Path(__file__).resolve().parent
HEADER = struct.Struct("<8sQIIII")
MAGIC = b"PRANS02\0"
RAW = b"PRAW001\0"
SCALE = 4096
CHUNK = 1024


def normalize(hist):
    hist = np.asarray(hist, dtype=np.uint64)
    if not int(hist.sum()):
        raise ValueError("Empty entropy stream")
    exact = hist.astype(np.float64) * SCALE / float(hist.sum())
    freq = np.maximum(np.floor(exact).astype(np.int64), (hist > 0).astype(np.int64))
    while int(freq.sum()) > SCALE:
        eligible = np.where(freq > 1)[0]
        i = int(eligible[np.argmax(freq[eligible] - exact[eligible])])
        freq[i] -= 1
    while int(freq.sum()) < SCALE:
        i = int(np.argmax(exact - freq))
        freq[i] += 1
    assert int(freq.sum()) == SCALE and np.all(freq[hist > 0] > 0)
    f = freq.astype(np.uint32)
    c = np.concatenate(([0], np.cumsum(freq[:-1]))).astype(np.uint32)
    lut = np.repeat(np.arange(256, dtype=np.uint8), freq)
    return f, c, lut


class Codec:
    def __init__(self):
        shared = build_shared_library(HERE / "rans.c", libraries=(), optimization="-O3")
        self.lib = ctypes.CDLL(str(shared))
        p = ctypes.c_void_p
        u = ctypes.c_uint32
        q = ctypes.c_uint64
        self.lib.rans_histogram.argtypes = [p, q, u, p]
        self.lib.rans_encode.argtypes = [p, u, u, p, p, p, q]
        self.lib.rans_encode.restype = ctypes.c_int64
        self.lib.rans_decode.argtypes = [p, q, p, u, u, p, p, p]
        self.lib.rans_decode.restype = ctypes.c_int
        self.lib.rans_histogram_nibbles.argtypes = self.lib.rans_histogram.argtypes
        self.lib.rans_encode_nibbles.argtypes = self.lib.rans_encode.argtypes
        self.lib.rans_encode_nibbles.restype = ctypes.c_int64
        self.lib.rans_decode_nibbles.argtypes = self.lib.rans_decode.argtypes
        self.lib.rans_decode_nibbles.restype = ctypes.c_int
        self.lib.bytes_checksum.argtypes = [p, u, u]
        self.lib.bytes_checksum.restype = u
        self.cuda = None

    def encode(self, raw, segments=None, force=False):
        raw = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1)
        segments = segments or [(0, len(raw), 1)]
        tables = []
        descs = []
        payload = []
        at = 0
        for segment in segments:
            begin, length, stride = segment[:3]
            nibble = len(segment) > 3 and segment[3] == "nibble"
            if length % stride:
                raise ValueError("Segment must contain complete elements")
            for plane in range(stride):
                start = begin + plane
                count = length // stride
                hist = np.zeros(256, dtype=np.uint64)
                histogram = self.lib.rans_histogram_nibbles if nibble else self.lib.rans_histogram
                histogram(raw.ctypes.data + start, count, stride, hist.ctypes.data)
                f, c, lut = normalize(hist)
                table = len(tables)
                tables.append((f, c, lut))
                scratch = np.empty(4 * CHUNK + 32, dtype=np.uint8)
                for offset in range(0, count, CHUNK):
                    n = min(CHUNK, count - offset)
                    source = raw.ctypes.data + start + offset * stride
                    encoder = self.lib.rans_encode_nibbles if nibble else self.lib.rans_encode
                    size = encoder(
                        source,
                        n,
                        stride,
                        f.ctypes.data,
                        c.ctypes.data,
                        scratch.ctypes.data,
                        len(scratch),
                    )
                    if size < 0:
                        raise RuntimeError(f"rANS encoder failed: {size}")
                    mode = 2 if nibble else 0
                    if size >= n:
                        chunk = raw[
                            start + offset * stride : start + (offset + n) * stride : stride
                        ].tobytes()
                        mode = 1
                    else:
                        chunk = scratch[:size].tobytes()
                    checksum = int(self.lib.bytes_checksum(source, n, stride))
                    descs.append(
                        [at, len(chunk), start + offset * stride, n, stride, table, mode, checksum]
                    )
                    payload.append(chunk)
                    at += len(chunk)
        frequencies = np.stack([t[0] for t in tables])
        cumulatives = np.stack([t[1] for t in tables])
        lookups = np.stack([t[2] for t in tables])
        descriptor = np.asarray(descs, dtype=np.uint32)
        meta = (
            frequencies.tobytes() + cumulatives.tobytes() + lookups.tobytes() + descriptor.tobytes()
        )
        result = (
            HEADER.pack(MAGIC, len(raw), len(tables), len(descs), len(meta), at)
            + meta
            + b"".join(payload)
        )
        if not force and len(result) >= len(raw) + HEADER.size:
            return HEADER.pack(RAW, len(raw), 0, 0, 0, len(raw)) + raw.tobytes()
        return result

    @staticmethod
    def parse(record):
        magic, size, nt, nc, meta, payload = HEADER.unpack_from(record)
        if magic not in (MAGIC, RAW):
            raise ValueError("Not an expert transport record")
        if HEADER.size + meta + payload > len(record):
            raise ValueError("Truncated expert record")
        if magic == RAW:
            if payload != size or nt or nc or meta:
                raise ValueError("Invalid raw record")
            return dict(raw=True, size=size, offset=HEADER.size, payload=payload)
        if meta != nt * 6144 + nc * 32:
            raise ValueError("Invalid rANS descriptor lengths")
        foff = HEADER.size
        coff = foff + nt * 1024
        loff = coff + nt * 1024
        doff = loff + nt * 4096
        f = np.frombuffer(record, np.uint32, count=nt * 256, offset=foff).reshape(nt, 256)
        c = np.frombuffer(record, np.uint32, count=nt * 256, offset=coff).reshape(nt, 256)
        lut = np.frombuffer(record, np.uint8, count=nt * 4096, offset=loff).reshape(nt, 4096)
        desc = np.frombuffer(record, np.uint32, count=nc * 8, offset=doff).reshape(nc, 8)
        if np.any(f.sum(axis=1) != SCALE):
            raise ValueError("Invalid entropy table")
        safe = desc.astype(np.uint64)
        if np.any(safe[:, 1] < 1) or np.any(safe[:, 0] + safe[:, 1] > payload):
            raise ValueError("Input descriptor out of range")
        if (
            np.any(safe[:, 3] < 1)
            or np.any(safe[:, 4] < 1)
            or np.any(safe[:, 2] + (safe[:, 3] - 1) * safe[:, 4] >= size)
        ):
            raise ValueError("Output descriptor out of range")
        if np.any(desc[:, 5] < 0) or np.any(desc[:, 5] >= nt):
            raise ValueError("Table index out of range")
        return dict(
            raw=False,
            size=size,
            offset=HEADER.size + meta,
            payload=payload,
            nt=nt,
            nc=nc,
            f=f,
            c=c,
            lut=lut,
            descs=desc,
            foff=foff,
            coff=coff,
            loff=loff,
            doff=doff,
        )

    def decode_cpu(self, record, output=None):
        record = np.frombuffer(record, dtype=np.uint8)
        p = self.parse(record)
        output = np.empty(p["size"], dtype=np.uint8) if output is None else output
        if output.nbytes != p["size"]:
            raise ValueError("Wrong output size")
        if p["raw"]:
            output[:] = record[p["offset"] : p["offset"] + p["size"]]
            return output
        for d in p["descs"]:
            offset, nbytes, dst, n, stride, table, raw, checksum = map(int, d)
            source = record[p["offset"] + offset : p["offset"] + offset + nbytes]
            if raw & 1:
                output[dst : dst + n * stride : stride] = source
            else:
                decoder = self.lib.rans_decode_nibbles if raw & 2 else self.lib.rans_decode
                rc = decoder(
                    source.ctypes.data,
                    nbytes,
                    output.ctypes.data + dst,
                    n,
                    stride,
                    p["f"][table].ctypes.data,
                    p["c"][table].ctypes.data,
                    p["lut"][table].ctypes.data,
                )
                if rc:
                    raise ValueError(f"rANS decoder rejected stream ({rc})")
            actual = self.lib.bytes_checksum(output.ctypes.data + dst, n, stride)
            if actual != checksum:
                raise ValueError("Decoded checksum mismatch")
        return output

    def enable_cuda(self):
        if self.cuda is not None:
            return
        shared = build_cuda_decoder(HERE / "rans_cuda.cu")
        self.cuda = ctypes.CDLL(str(shared))
        p = ctypes.c_void_p
        self.cuda.rans_cuda_decode.argtypes = [
            p,
            p,
            ctypes.c_int,
            p,
            p,
            p,
            p,
            p,
            ctypes.c_int,
            p,
            p,
        ]
        self.cuda.rans_cuda_decode.restype = ctypes.c_int

    def decode_cuda(self, device_record, metadata, output, stream, errors):
        self.enable_cuda()
        p = metadata
        outputs = output if isinstance(output, list) else [output]
        sizes = [t.numel() * t.element_size() for t in outputs]
        if sum(sizes) != p["size"]:
            raise ValueError("Wrong GPU output size")
        offsets = np.cumsum([0] + sizes[:-1]).astype(np.uint64)
        if p["raw"]:
            for tensor, offset, size in zip(outputs, offsets, sizes):
                tensor.view(dtype=__import__("torch").uint8).reshape(-1).copy_(
                    device_record[p["offset"] + int(offset) : p["offset"] + int(offset) + size],
                    non_blocking=True,
                )
            return
        address = device_record.data_ptr()
        pointers = np.asarray([t.data_ptr() for t in outputs], dtype=np.uint64)
        rc = self.cuda.rans_cuda_decode(
            address + p["offset"],
            address + p["doff"],
            p["nc"],
            address + p["foff"],
            address + p["coff"],
            address + p["loff"],
            pointers.ctypes.data,
            offsets.ctypes.data,
            len(outputs),
            errors.data_ptr(),
            stream.cuda_stream,
        )
        if rc:
            raise RuntimeError(f"CUDA decoder launch error {rc}")
