"""Checkpoint-derived expert banks and tensor locations.

Fused BF16 checkpoints store one expert per tensor row. DS-FP4 checkpoints
store separate expert tensors with packed weights and block scales. Geometry
defines the contiguous RAM row, backend tensor views and source byte ranges.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

# safetensors dtype -> (torch dtype name, bytes per element)
_DTYPES = {
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "I8": ("uint8", 1),
    "U8": ("uint8", 1),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E8M0": ("float8_e8m0fnu", 1),
}


@dataclass(frozen=True)
class Bank:
    """One offload-cache bank, described per expert."""

    name: str
    shape: tuple  # Per-expert shape within this bank.
    dtype: str  # Torch dtype name.
    itemsize: int

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.itemsize


@dataclass(frozen=True)
class Piece:
    """One checkpoint tensor and where it lands inside an expert's RAM row.

    ``fused`` distinguishes the two families: a fused source is a ``[E, ...]``
    tensor in which expert ``e`` sits at row ``e``; an unfused source is that one
    expert's own tensor.  ``offset`` is the byte offset **within the bank's
    per-expert buffer** -- non-zero only where two checkpoint tensors are stacked
    into one bank (ds_fp4's w1/w3).
    """

    bank: str
    key: str  # Checkpoint key template with {layer} and {expert} fields.
    fused: bool
    offset: int
    nbytes: int

    def resolve(self, layer: int, expert: int) -> str:
        return self.key.format(layer=layer, expert=expert)


@dataclass(frozen=True)
class Geometry:
    name: str
    path: Path
    layers: int  # layers that carry routed experts
    experts: int  # experts per layer
    banks: tuple  # tuple[Bank], in bank order
    pieces: tuple  # tuple[Piece]
    quant: str = "bf16"  # Backend quantization format.
    # Additional layout fields for fused BF16 checkpoints.
    gate_up_shape: tuple = ()
    down_shape: tuple = ()
    key_template: str = ""

    @property
    def bank_names(self) -> tuple:
        return tuple(b.name for b in self.banks)

    @property
    def bank_offsets(self) -> dict:
        """Byte offset of each bank within a contiguous expert row."""
        out, at = {}, 0
        for b in self.banks:
            out[b.name] = at
            at += b.nbytes
        return out

    @property
    def row_bytes(self) -> int:
        """One expert: every bank, contiguous, in bank order."""
        return sum(b.nbytes for b in self.banks)

    @property
    def gate_up_bytes(self) -> int:
        return self.banks[0].nbytes

    @property
    def down_bytes(self) -> int:
        return self.banks[1].nbytes

    @property
    def total_experts(self) -> int:
        return self.layers * self.experts

    @property
    def double_buffer(self) -> int:
        """Prefill borrows two whole layers of GPU slots."""
        return 2 * self.experts

    def key(self, layer: int, proj: str) -> str:
        return self.key_template.format(layer=layer, proj=proj)

    def host_rows_for(self, ram_gib: float, runtime_gib: float) -> int:
        return int((ram_gib - runtime_gib) * 2**30) // self.row_bytes

    def host_views(self, row) -> dict:
        """Split one expert's RAM row (a uint8 tensor) into the banks' tensors."""
        import torch

        out, at = {}, 0
        for b in self.banks:
            chunk = row[at : at + b.nbytes]
            out[b.name] = chunk.view(getattr(torch, b.dtype)).view(*b.shape)
            at += b.nbytes
        return out


# Checkpoints differ in where the fused expert tensors live.
_TEMPLATES = (
    "model.language_model.layers.{layer}.mlp.experts.{proj}",  # qwen3.6
    "model.language_model.layers.{layer}.experts.{proj}",  # gemma-4
    "model.layers.{layer}.mlp.experts.{proj}",
)

# DeepSeek-V4: one tensor per expert, not a fused [E, ...] row.
_DS_KEY = "layers.{layer}.ffn.experts.{expert}.%s.%s"


def _header(path: Path, shard: str) -> dict:
    with (path / shard).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _entry(path: Path, index: dict, key: str, cache: dict) -> dict:
    shard = index[key]
    if shard not in cache:
        cache[shard] = _header(path, shard)
    return cache[shard][key]


def _bf16(path: Path, index: dict) -> Geometry:
    """Qwen3.6 / gemma-4: two fused BF16 banks."""
    template = next(
        (t for t in _TEMPLATES if t.format(layer=0, proj="gate_up_proj") in index), None
    )
    if template is None:
        return None

    layers = 0
    while template.format(layer=layers, proj="gate_up_proj") in index:
        layers += 1

    heads, shapes = {}, {}
    for proj in ("gate_up_proj", "down_proj"):
        key = template.format(layer=0, proj=proj)
        entry = _entry(path, index, key, heads)
        if entry["dtype"] != "BF16":
            raise RuntimeError(f"{key} is {entry['dtype']}, the fused arms assume BF16")
        shapes[proj] = entry["shape"]

    experts = shapes["gate_up_proj"][0]
    if shapes["down_proj"][0] != experts:
        raise RuntimeError("gate_up and down disagree on the expert count")

    gu, dn = tuple(shapes["gate_up_proj"][1:]), tuple(shapes["down_proj"][1:])
    banks = (Bank("gate_up", gu, "bfloat16", 2), Bank("down", dn, "bfloat16", 2))
    pieces = tuple(
        Piece(b.name, template.replace("{proj}", proj), True, 0, b.nbytes)
        for b, proj in zip(banks, ("gate_up_proj", "down_proj"))
    )
    return Geometry(
        name=path.name,
        path=path,
        layers=layers,
        experts=experts,
        banks=banks,
        pieces=pieces,
        quant="bf16",
        gate_up_shape=gu,
        down_shape=dn,
        key_template=template,
    )


def _ds_fp4(path: Path, index: dict) -> Geometry:
    """DeepSeek-V4-Flash: four ds_fp4 banks built from per-expert w1/w2/w3.

    Bank shapes and the w1/w3 stacking order are taken from FreeToken's own
    loader (``models/deepseek_v4/weight.py``: ``w1 -> gate_up[:, :I]``,
    ``w3 -> gate_up[:, I:]``, ``w2 -> down``), not guessed -- a swapped gate/up
    would still run and silently produce wrong tokens.
    """
    if _DS_KEY % ("w1", "weight") not in index.get("__template_probe__", index):
        probe = (_DS_KEY % ("w1", "weight")).format(layer=0, expert=0)
        if probe not in index:
            return None

    layers = 0
    while (_DS_KEY % ("w1", "weight")).format(layer=layers, expert=0) in index:
        layers += 1
    experts = 0
    while (_DS_KEY % ("w1", "weight")).format(layer=0, expert=experts) in index:
        experts += 1

    heads = {}
    got = {}
    for proj in ("w1", "w2", "w3"):
        for kind in ("weight", "scale"):
            key = (_DS_KEY % (proj, kind)).format(layer=0, expert=0)
            e = _entry(path, index, key, heads)
            if e["dtype"] not in _DTYPES:
                raise RuntimeError(f"{key}: unhandled dtype {e['dtype']}")
            got[proj, kind] = (tuple(e["shape"]), *_DTYPES[e["dtype"]])

    (w1w, w1_dt, w1_is) = got["w1", "weight"]
    (w1s, s_dt, s_is) = got["w1", "scale"]
    (w2w, _, _) = got["w2", "weight"]
    (w2s, _, _) = got["w2", "scale"]
    for a, b in (("w1", "w3"),):
        if got[a, "weight"] != got[b, "weight"] or got[a, "scale"] != got[b, "scale"]:
            raise RuntimeError("w1 and w3 differ in shape; the gate_up stacking assumes they match")

    banks = (
        Bank("gate_up_packed", (2 * w1w[0], w1w[1]), w1_dt, w1_is),
        Bank("gate_up_scale", (2 * w1s[0], w1s[1]), s_dt, s_is),
        Bank("down_packed", w2w, w1_dt, w1_is),
        Bank("down_scale", w2s, s_dt, s_is),
    )
    half_w = w1w[0] * w1w[1] * w1_is
    half_s = w1s[0] * w1s[1] * s_is
    pieces = (
        Piece("gate_up_packed", _DS_KEY % ("w1", "weight"), False, 0, half_w),
        Piece("gate_up_packed", _DS_KEY % ("w3", "weight"), False, half_w, half_w),
        Piece("gate_up_scale", _DS_KEY % ("w1", "scale"), False, 0, half_s),
        Piece("gate_up_scale", _DS_KEY % ("w3", "scale"), False, half_s, half_s),
        Piece("down_packed", _DS_KEY % ("w2", "weight"), False, 0, w2w[0] * w2w[1] * w1_is),
        Piece("down_scale", _DS_KEY % ("w2", "scale"), False, 0, w2s[0] * w2s[1] * s_is),
    )
    return Geometry(
        name=path.name,
        path=path,
        layers=layers,
        experts=experts,
        banks=banks,
        pieces=pieces,
        quant="ds_fp4",
    )


def from_checkpoint(path) -> Geometry:
    path = Path(path).expanduser().resolve()
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    for builder in (_bf16, _ds_fp4):
        geo = builder(path, index)
        if geo is not None:
            return geo
    raise RuntimeError(f"no known expert key layout in {path}")


# Known checkpoint geometries for explicit model-layout validation.
QWEN36 = dict(
    layers=40,
    experts=256,
    row_bytes=6 * 2**20,
    total_experts=10240,
    double_buffer=512,
    gate_up_shape=(1024, 2048),
    down_shape=(2048, 512),
)

# DeepSeek-V4-Flash-0731, derived once from the checkpoint and pinned here for the
# same reason: 43 x 256 experts, 12.75 MiB each -> a 137.06 GiB routed pool.
DSV4 = dict(
    layers=43,
    experts=256,
    row_bytes=int(12.75 * 2**20),
    total_experts=11008,
    double_buffer=512,
    quant="ds_fp4",
)


def check_qwen(geo: Geometry) -> None:
    for field, expected in QWEN36.items():
        actual = getattr(geo, field)
        assert actual == expected, f"{field}: {actual} != {expected}"


def check_dsv4(geo: Geometry) -> None:
    for field, expected in DSV4.items():
        actual = getattr(geo, field)
        assert actual == expected, f"{field}: {actual} != {expected}"
    names = geo.bank_names
    assert names == ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"), names
    # Stack gate w1 before up w3; reversing them silently changes the output.
    order = [(p.bank, p.key.split(".")[-2], p.offset) for p in geo.pieces]
    assert order[0] == ("gate_up_packed", "w1", 0), order
    assert order[1][:2] == ("gate_up_packed", "w3") and order[1][2] == geo.banks[0].nbytes // 2, (
        order
    )
