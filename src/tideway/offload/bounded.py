"""Aligned O_DIRECT checkpoint reads with bounded worker-local buffers."""

import ctypes
import json
import mmap
import os
from pathlib import Path
import struct
import time


def _geo(model):
    from .geometry import from_checkpoint

    return from_checkpoint(model)


def read_bytes():
    return int(
        next(
            x.split(":", 1)[1]
            for x in Path("/proc/self/io").read_text().splitlines()
            if x.startswith("read_bytes:")
        )
    )


class DirectExperts:
    """Demand reads of one expert's bytes straight out of the checkpoint (O_DIRECT).

    The plan is built from the Geometry, not from Qwen's shape. Two families are
    handled: a **fused** source is a ``[E, ...]`` tensor whose expert ``e`` is a
    row at ``e * stride``; an **unfused** source (DeepSeek-V4) is that one
    expert's own tensor, so every expert needs its own file offset.

    """

    def __init__(self, model, geo=None, workers=None):
        model = Path(model).expanduser().resolve()
        self.geo = geo if geo is not None else _geo(model)
        g = self.geo
        self.row = g.row_bytes
        self.fds = {}
        headers = {}
        index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]

        def entry(key):
            path = model / index[key]
            if path not in headers:
                with path.open("rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    headers[path] = (8 + n, json.loads(f.read(n)))
                self.fds[path] = os.open(path, os.O_RDONLY | os.O_DIRECT)
            start, h = headers[path]
            return self.fds[path], start, h[key]

        # plan[e] = [(fd, file_offset, width, bank, offset within that bank), ...]
        self.plan = [[] for _ in range(g.total_experts)]
        for piece in g.pieces:
            for layer in range(g.layers):
                if piece.fused:
                    fd, base, e = entry(piece.resolve(layer, 0))
                    span = e["data_offsets"][1] - e["data_offsets"][0]
                    stride = span // g.experts
                    assert stride == piece.nbytes, (
                        f"{piece.bank}: stride {stride} != {piece.nbytes}"
                    )
                    first = base + e["data_offsets"][0]
                    for local in range(g.experts):
                        self.plan[layer * g.experts + local].append(
                            (fd, first + local * stride, piece.nbytes, piece.bank, piece.offset)
                        )
                else:
                    for local in range(g.experts):
                        fd, base, e = entry(piece.resolve(layer, local))
                        span = e["data_offsets"][1] - e["data_offsets"][0]
                        assert span == piece.nbytes, f"{piece.bank}: {span} != {piece.nbytes}"
                        self.plan[layer * g.experts + local].append(
                            (
                                fd,
                                base + e["data_offsets"][0],
                                piece.nbytes,
                                piece.bank,
                                piece.offset,
                            )
                        )

        # Each reader owns an aligned bounce buffer; preadv and memmove release the GIL.
        self.workers = max(
            1, int(os.environ.get("TIDEWAY_SSD_WORKERS", "8")) if workers is None else int(workers)
        )
        bounce = max(p.nbytes for p in g.pieces) + 8192
        self.buffers = [mmap.mmap(-1, bounce) for _ in range(self.workers)]
        self.addrs = [ctypes.addressof(ctypes.c_char.from_buffer(b)) for b in self.buffers]
        assert all(a % 4096 == 0 for a in self.addrs)
        self.buffer, self.addr = self.buffers[0], self.addrs[0]
        self.pool = None
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="ssd")
        import threading

        self._lock = threading.Lock()
        self.bytes = 0
        self.seconds = 0  # Wall time per read call or concurrent batch.
        self.thread_seconds = 0  # Sum of worker read times.
        self.batches = 0
        self.batch_rows = 0

    def _read_into(self, expert, destinations, slot):
        buf, addr = self.buffers[slot], self.addrs[slot]
        got_total = 0
        spent = 0
        for fd, offset, width, bank, within in self.plan[expert]:
            aligned = offset // 4096 * 4096
            length = ((offset - aligned + width + 4095) // 4096) * 4096
            begin = time.monotonic()
            got = os.preadv(fd, [memoryview(buf)[:length]], aligned)
            spent += time.monotonic() - begin
            # A tensor ending at file EOF has no trailing alignment padding.
            # The payload must be complete; padding beyond EOF is unnecessary.
            if got < offset - aligned + width:
                raise IOError(f"short O_DIRECT payload {got}/{offset - aligned + width}")
            got_total += got
            ctypes.memmove(destinations[bank].data_ptr() + within, addr + (offset - aligned), width)
        return got_total, spent

    def _read_piece(self, piece, destinations, slot):
        fd, offset, width, bank, within = piece
        buf, addr = self.buffers[slot], self.addrs[slot]
        aligned = offset // 4096 * 4096
        length = ((offset - aligned + width + 4095) // 4096) * 4096
        begin = time.monotonic()
        got = os.preadv(fd, [memoryview(buf)[:length]], aligned)
        spent = time.monotonic() - begin
        if got < offset - aligned + width:
            raise IOError(f"short O_DIRECT payload {got}/{offset - aligned + width}")
        ctypes.memmove(destinations[bank].data_ptr() + within, addr + (offset - aligned), width)
        return got, spent

    def _read_range(self, fd, offset, width, dst_ptr, slot):
        """Read a byte range through a worker-owned, 4 KiB-aligned bounce buffer."""
        buf, addr = self.buffers[slot], self.addrs[slot]
        aligned = offset // 4096 * 4096
        length = ((offset - aligned + width + 4095) // 4096) * 4096
        begin = time.monotonic()
        got = os.preadv(fd, [memoryview(buf)[:length]], aligned)
        spent = time.monotonic() - begin
        if got < offset - aligned + width:
            raise IOError(f"short O_DIRECT payload {got}/{offset - aligned + width}")
        ctypes.memmove(dst_ptr, addr + (offset - aligned), width)
        return got, spent

    def read(self, expert, destinations):
        """Read one expert, splitting aligned ranges across available I/O workers."""
        begin = time.monotonic()
        pieces = self.plan[expert]
        if self.pool is None:
            got, spent = self._read_into(expert, destinations, 0)
        else:
            total = sum(p[2] for p in pieces)
            part = max(
                256 * 1024, ((total + self.workers - 1) // self.workers + 4095) // 4096 * 4096
            )
            tasks = []
            for fd, offset, width, bank, within in pieces:
                dst = destinations[bank].data_ptr() + within
                at = 0
                while at < width:
                    w = min(part, width - at)
                    tasks.append((fd, offset + at, w, dst + at))
                    at += w
            # Serialize each lane to prevent concurrent reuse of its bounce buffer.
            lanes = [[] for _ in range(self.workers)]
            for i, t in enumerate(tasks):
                lanes[i % self.workers].append(t)

            def run(lane_id):
                got = spent = 0
                for t in lanes[lane_id]:
                    g, sp = self._read_range(*t, lane_id)
                    got += g
                    spent += sp
                return got, spent

            results = list(self.pool.map(run, [i for i in range(self.workers) if lanes[i]]))
            got = sum(r[0] for r in results)
            spent = sum(r[1] for r in results)
        with self._lock:
            self.bytes += got
            self.seconds += time.monotonic() - begin
            self.thread_seconds += spent

    def read_many(self, items):
        """Read a batch of experts into pairwise-disjoint destinations before returning."""
        items = list(items)
        if not items:
            return
        if self.pool is None or len(items) == 1:
            for e, d in items:
                self.read(e, d)
            return
        begin = time.monotonic()
        # Each worker exclusively owns one bounce buffer for its lane.
        lanes = [[] for _ in range(self.workers)]
        for i, (e, d) in enumerate(items):
            lanes[i % self.workers].append((e, d))

        def run(lane_id):
            got = spent = 0
            for e, d in lanes[lane_id]:
                g, sp = self._read_into(e, d, lane_id)
                got += g
                spent += sp
            return got, spent

        results = list(self.pool.map(run, [i for i in range(self.workers) if lanes[i]]))
        with self._lock:
            self.bytes += sum(r[0] for r in results)
            self.thread_seconds += sum(r[1] for r in results)
            self.seconds += time.monotonic() - begin
            self.batches += 1
            self.batch_rows += len(items)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        for b in self.buffers:
            b.close()
