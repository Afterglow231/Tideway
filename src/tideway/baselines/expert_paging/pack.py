"""Bounded streaming preparation and aligned direct reads of expert-major packs."""

import hashlib
import json
import os
from pathlib import Path
import struct
import time
import threading
import numpy as np

from .codec import Codec, HEADER, RAW, CHUNK

ALIGN = 4096


def padded(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


class SourceExperts:
    """Independent raw checkpoint reader using the public neutral model geometry."""

    def __init__(self, geometry, direct=False):
        self.geo = geometry
        self.model = Path(geometry.path)
        self.direct = direct
        self.index = json.loads((self.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.headers = {}
        self.fds = {}

    def _entry(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            path = self.model / shard
            with path.open("rb") as file:
                length = struct.unpack("<Q", file.read(8))[0]
                self.headers[shard] = (length + 8, json.loads(file.read(length)))
            self.fds[shard] = os.open(path, os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
        start, header = self.headers[shard]
        return self.fds[shard], start + header[key]["data_offsets"][0]

    def read(self, expert):
        g = self.geo
        layer, local = divmod(int(expert), g.experts)
        if not 0 <= layer < g.layers:
            raise ValueError("Expert index outside model")
        raw = np.empty(g.row_bytes, dtype=np.uint8)
        for piece in g.pieces:
            fd, offset = self._entry(piece.resolve(layer, local))
            if piece.fused:
                offset += local * piece.nbytes
            blob = os.pread(fd, piece.nbytes, offset)
            if len(blob) != piece.nbytes:
                raise IOError("Short checkpoint read")
            target = g.bank_offsets[piece.bank] + piece.offset
            raw[target : target + piece.nbytes] = np.frombuffer(blob, np.uint8)
        return raw

    def independent_expected(self, expert):
        import torch
        from safetensors import safe_open

        g = self.geo
        layer, local = divmod(int(expert), g.experts)
        raw = np.empty(g.row_bytes, dtype=np.uint8)
        for piece in g.pieces:
            key = piece.resolve(layer, local)
            with safe_open(self.model / self.index[key], framework="pt", device="cpu") as file:
                value = file.get_slice(key)[local] if piece.fused else file.get_tensor(key)
                byte = value.contiguous().view(torch.uint8).numpy().reshape(-1)
                assert byte.nbytes == piece.nbytes
                target = g.bank_offsets[piece.bank] + piece.offset
                raw[target : target + piece.nbytes] = byte
        return raw

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def segments(geo):
    out = []
    for bank in geo.banks:
        base = geo.bank_offsets[bank.name]
        stride = 2 if bank.dtype in ("bfloat16", "float16") else 1
        segment = (base, bank.nbytes, stride)
        if geo.quant == "ds_fp4" and "packed" in bank.name:
            segment = (*segment, "nibble")
        out.append(segment)
    return out


def prepare(
    geo, output, expert_ids, raw_comparison=False, verify_every=1, reserve_GiB=12, resume=False
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    expert_ids = list(expert_ids)
    codec = Codec()
    source = SourceExperts(geo)
    manifest = dict(
        schema=1,
        model=str(geo.path),
        quant=geo.quant,
        row_bytes=geo.row_bytes,
        layers=geo.layers,
        experts_per_layer=geo.experts,
        banks=[
            dict(name=b.name, shape=list(b.shape), dtype=b.dtype, nbytes=b.nbytes)
            for b in geo.banks
        ],
        codec=f"12-bit byte/nibble rANS, independently decodable {CHUNK}-byte streams, optional BF16 byte-plane transform",
        rows={},
        complete=False,
        raw_comparison=raw_comparison,
    )
    if resume:
        previous = (output / "manifest.json").read_text()
        manifest = json.loads(previous)
        if manifest["complete"]:
            raise ValueError("Refusing to resume a completed pack")
        if manifest["model"] != str(geo.path) or manifest["raw_comparison"] != raw_comparison:
            raise ValueError("Resume configuration differs from partial pack")
        (output / f"manifest.partial-{len(manifest['rows'])}.json").write_text(previous)
    pack = (output / "experts.pack").open("r+b" if resume else "xb")
    raw_pack = (
        (output / "raw_comparison.pack").open("r+b" if resume else "xb") if raw_comparison else None
    )
    raw_bytes = len(manifest["rows"]) * geo.row_bytes
    stored = sum(r["aligned_bytes"] for r in manifest["rows"].values())
    if resume:
        end = max((r["offset"] + r["aligned_bytes"] for r in manifest["rows"].values()), default=0)
        raw_end = max(
            (
                r.get("raw_offset", 0) + r.get("raw_aligned_bytes", 0)
                for r in manifest["rows"].values()
            ),
            default=0,
        )
        if os.fstat(pack.fileno()).st_size != end or (
            raw_pack and os.fstat(raw_pack.fileno()).st_size != raw_end
        ):
            raise ValueError("Partial pack contains an uncommitted tail; manual audit required")
        if manifest["rows"]:
            last = list(manifest["rows"].values())[-1]
            payload = os.pread(pack.fileno(), last["record_bytes"], last["offset"])
            if hashlib.sha256(payload).hexdigest() != last["record_sha256"]:
                raise ValueError("Partial pack final committed record is corrupt")
        remaining = sum(str(int(e)) not in manifest["rows"] for e in expert_ids)
        upper = remaining * padded(geo.row_bytes + HEADER.size)
        fs = os.statvfs(output)
        needed = upper * (2 if raw_pack else 1)
        if fs.f_bavail * fs.f_frsize < needed + reserve_GiB * 2**30:
            raise RuntimeError("Resume worst-case reservation would violate free-space floor")
        # Reserve a bounded final size so speculative filesystem EOF allocation
        # cannot consume many additional GiB while the final records are written.
        os.posix_fallocate(pack.fileno(), 0, end + upper)
        pack.seek(end)
        if raw_pack:
            os.posix_fallocate(raw_pack.fileno(), 0, raw_end + upper)
            raw_pack.seek(raw_end)
        manifest["resumed_from_rows"] = len(manifest["rows"])
    started = time.monotonic()
    try:
        for index, expert in enumerate(expert_ids):
            if str(int(expert)) in manifest["rows"]:
                continue
            fs = os.statvfs(output)
            if fs.f_bavail * fs.f_frsize < reserve_GiB * 2**30 + 4 * geo.row_bytes:
                # Some filesystems reserve multi-GiB extents beyond EOF. Closing
                # our own writer releases speculative extents before deciding
                # whether actual committed data violates the disk floor.
                at = pack.tell()
                pack.flush()
                pack.truncate(at)
                pack.close()
                pack = (output / "experts.pack").open("r+b")
                pack.seek(at)
                if raw_pack:
                    at = raw_pack.tell()
                    raw_pack.flush()
                    raw_pack.truncate(at)
                    raw_pack.close()
                    raw_pack = (output / "raw_comparison.pack").open("r+b")
                    raw_pack.seek(at)
                fs = os.statvfs(output)
                if fs.f_bavail * fs.f_frsize < reserve_GiB * 2**30 + 4 * geo.row_bytes:
                    raise RuntimeError(
                        "Insufficient SSD reserve; preserving partial pack without deleting model files"
                    )
            raw = source.read(expert)
            if verify_every and index % verify_every == 0:
                assert np.array_equal(raw, source.independent_expected(expert)), (
                    f"Source reader differs: expert {expert}"
                )
            encoded = codec.encode(raw, segments(geo))
            decoded = codec.decode_cpu(encoded)
            assert np.array_equal(decoded, raw), f"CPU codec mismatch: expert {expert}"
            offset = pack.tell()
            size = padded(len(encoded))
            pack.write(encoded)
            pack.write(bytes(size - len(encoded)))
            row = dict(
                offset=offset,
                record_bytes=len(encoded),
                aligned_bytes=size,
                original_sha256=hashlib.sha256(raw).hexdigest(),
                record_sha256=hashlib.sha256(encoded).hexdigest(),
                compressed=encoded[:8] != RAW,
            )
            if raw_pack:
                raw_record = HEADER.pack(RAW, len(raw), 0, 0, 0, len(raw)) + raw.tobytes()
                row["raw_offset"] = raw_pack.tell()
                row["raw_aligned_bytes"] = padded(len(raw_record))
                raw_pack.write(raw_record)
                raw_pack.write(bytes(row["raw_aligned_bytes"] - len(raw_record)))
            manifest["rows"][str(expert)] = row
            raw_bytes += len(raw)
            stored += size
            progress = dict(
                experts=len(manifest["rows"]),
                raw_bytes=raw_bytes,
                stored_bytes=stored,
                raw_to_stored_ratio=raw_bytes / stored,
                seconds=time.monotonic() - started,
            )
            if index % 16 == 0:
                (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
        pack.flush()
        pack.truncate(pack.tell())
        os.fdatasync(pack.fileno())
        if raw_pack:
            raw_pack.flush()
            raw_pack.truncate(raw_pack.tell())
            os.fdatasync(raw_pack.fileno())
        manifest.update(
            complete=True,
            raw_bytes=raw_bytes,
            stored_bytes=stored,
            raw_to_stored_ratio=raw_bytes / stored,
            preparation_s=time.monotonic() - started,
        )
    finally:
        pack.flush()
        pack.truncate(pack.tell())
        if raw_pack:
            raw_pack.flush()
            raw_pack.truncate(raw_pack.tell())
        pack.close()
        if raw_pack:
            raw_pack.close()
        source.close()
        temp = output / "manifest.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(output / "manifest.json")
    return manifest


class PackReader:
    def __init__(self, path, raw_comparison=False):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if not self.manifest["complete"]:
            raise ValueError("Refusing incomplete expert pack")
        self.raw = raw_comparison
        if raw_comparison and not self.manifest["raw_comparison"]:
            raise ValueError("No raw comparison pack")
        self.fd = os.open(
            self.path / ("raw_comparison.pack" if raw_comparison else "experts.pack"),
            os.O_RDONLY | os.O_DIRECT,
        )
        self.bytes = 0
        self.seconds = 0.0
        self.max_size = max(self.size(int(e)) for e in self.manifest["rows"])
        self.lock = threading.Lock()

    def size(self, expert):
        row = self.manifest["rows"][str(int(expert))]
        return row["raw_aligned_bytes"] if self.raw else row["aligned_bytes"]

    def read_into(self, expert, tensor):
        row = self.manifest["rows"][str(int(expert))]
        size = self.size(expert)
        offset = row["raw_offset"] if self.raw else row["offset"]
        if tensor.data_ptr() % ALIGN or offset % ALIGN or size % ALIGN:
            raise ValueError("O_DIRECT alignment mismatch")
        view = tensor[:size].numpy()
        began = time.monotonic()
        got = os.preadv(self.fd, [memoryview(view)], offset)
        with self.lock:
            self.seconds += time.monotonic() - began
            self.bytes += got
        if got != size:
            raise IOError(f"Short expert read {got}/{size}")
        return view

    def close(self):
        os.close(self.fd)


class CheckpointRecordReader:
    """Bounded O_DIRECT source for raw baselines, without another full raw pack.

    Each worker owns one aligned anonymous CPU bounce region. All aligned source
    bytes and the CPU assembly time are charged. RAM hits retain normal raw
    records; there is no hidden host master or filesystem page-cache weight hit.
    """

    def __init__(self, geo, pack, workers=8):
        self.manifest = json.loads((Path(pack) / "manifest.json").read_text())
        if not self.manifest["complete"] or len(self.manifest["rows"]) != geo.total_experts:
            raise ValueError("Requires complete verified expert metadata")
        self.geo = geo
        self.source = SourceExperts(geo, direct=True)
        self.raw = True
        # Populate metadata/file handles before threaded demand reads.
        seen = set()
        for key, shard in self.source.index.items():
            if shard not in seen:
                self.source._entry(key)
                seen.add(shard)
        sizes = {fd: os.fstat(fd).st_size for fd in self.source.fds.values()}
        self.plans = {}
        self.read_sizes = {}
        for expert in range(geo.total_experts):
            layer, local = divmod(expert, geo.experts)
            pieces = []
            for piece in geo.pieces:
                fd, offset = self.source._entry(piece.resolve(layer, local))
                if piece.fused:
                    offset += local * piece.nbytes
                begin = offset // ALIGN * ALIGN
                size = padded(offset + piece.nbytes) - begin
                actual = min(size, sizes[fd] - begin)
                if actual < offset - begin + piece.nbytes:
                    raise ValueError("Checkpoint piece outside source file")
                pieces.append(
                    (
                        fd,
                        begin,
                        size,
                        offset - begin,
                        HEADER.size + geo.bank_offsets[piece.bank] + piece.offset,
                        piece.nbytes,
                        actual,
                    )
                )
            self.plans[expert] = pieces
            self.read_sizes[expert] = sum(p[-1] for p in pieces)
        self.max_size = padded(geo.row_bytes + HEADER.size)
        self.scratch_size = max(p[2] for pieces in self.plans.values() for p in pieces)
        self.workspace_bytes = workers * self.scratch_size
        self.local = threading.local()
        self.lock = threading.Lock()
        self.areas = []
        self.bytes = 0
        self.seconds = 0.0
        self.header = HEADER.pack(RAW, geo.row_bytes, 0, 0, 0, geo.row_bytes)

    def size(self, expert):
        return self.read_sizes[int(expert)]

    def read_into(self, expert, tensor):
        import mmap

        if not hasattr(self.local, "area"):
            area = mmap.mmap(-1, self.scratch_size)
            self.local.area = area
            with self.lock:
                self.areas.append(area)
        area = self.local.area
        view = tensor[: self.max_size].numpy()
        view[: HEADER.size] = np.frombuffer(self.header, np.uint8)
        began = time.monotonic()
        total = 0
        for fd, offset, size, skip, dest, nbytes, expected in self.plans[int(expert)]:
            window = memoryview(area)[:size]
            got = os.preadv(fd, [window], offset)
            if got != expected:
                raise IOError(f"Short checkpoint read {got}/{expected}")
            view[dest : dest + nbytes] = np.frombuffer(window, np.uint8, count=nbytes, offset=skip)
            total += got
            del window
        with self.lock:
            self.bytes += total
            self.seconds += time.monotonic() - began
        return view

    def close(self):
        self.source.close()
        for area in self.areas:
            area.close()
        self.areas.clear()
