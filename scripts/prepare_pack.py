#!/usr/bin/env python3
"""Prepare aligned expert records for SSD-LLaMA and FlashMoE."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sample", action="store_true", help="prepare nine experts for a small check"
    )
    parser.add_argument("--raw-comparison", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-every", type=int, default=1)
    parser.add_argument("--reserve-gib", type=int, default=12)
    args = parser.parse_args()
    from tideway.offload.geometry import from_checkpoint
    from tideway.baselines.expert_paging.pack import prepare

    geometry = from_checkpoint(args.model.expanduser().resolve())
    experts = list(range(geometry.total_experts))
    if args.sample:
        experts = sorted(
            {
                layer * geometry.experts + expert
                for layer in [0, geometry.layers // 2, geometry.layers - 1]
                for expert in [0, geometry.experts // 3, geometry.experts - 1]
            }
        )
    result = prepare(
        geometry,
        args.out.expanduser().resolve(),
        experts,
        args.raw_comparison,
        args.verify_every,
        args.reserve_gib,
        args.resume,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
