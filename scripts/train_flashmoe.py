#!/usr/bin/env python3
"""Train FlashMoE predictors from per-layer calibration route records."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True, help="GPU expert slots per layer")
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()
    from tideway.baselines.expert_paging.learned import train

    train(
        args.routes.expanduser().resolve(),
        args.out.expanduser().resolve(),
        experts=args.experts,
        layers=args.layers,
        cap=args.capacity,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
