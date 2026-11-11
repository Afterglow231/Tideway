#!/usr/bin/env python3
"""Assemble FreeToken 0.1.2 with the checked Tideway source overlay."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "backends" / "freetoken_tideway"
DEFAULT_OUT = ROOT / "build" / "freetoken_tideway"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="directory that will contain freetoken/"
    )
    ap.add_argument("--force", action="store_true", help="replace an existing assembled backend")
    a = ap.parse_args()

    changes = json.loads((OVERLAY / "changes.json").read_text())
    try:
        version = importlib.metadata.version("freetoken")
    except importlib.metadata.PackageNotFoundError:
        ap.error(
            "FreeToken is not installed; install this checkout with python -m pip install -e ."
        )
    if version != "0.1.2":
        ap.error(f"FreeToken {version} is installed; Tideway is built on FreeToken 0.1.2")
    upstream = Path(importlib.util.find_spec("freetoken").origin).resolve().parent
    output = a.out.expanduser().resolve()
    target = output / "freetoken"
    if target.resolve() == upstream or target.is_symlink():
        ap.error(
            "output must be a separate backend directory, not the installed FreeToken package or a symlink"
        )
    if target.exists() and not a.force:
        ap.error(f"{target} exists; pass --force to rebuild")

    mismatched = [
        c["path"]
        for c in changes["changed_source_files"]
        if c["upstream_sha256"]
        and (
            not (upstream.parent / c["path"]).is_file()
            or sha256(upstream.parent / c["path"]) != c["upstream_sha256"]
        )
    ]
    if mismatched:
        ap.error(
            f"installed FreeToken differs from pristine 0.1.2 in {mismatched}; reinstall freetoken==0.1.2"
        )

    for c in changes["changed_source_files"]:
        src = OVERLAY / c["path"]
        if not src.is_file() or sha256(src) != c["tideway_sha256"]:
            ap.error(f"Tideway overlay checksum mismatch: {src}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tideway-build-", dir=output) as temporary:
        staging = Path(temporary)
        shutil.copytree(
            upstream, staging / "freetoken", ignore=shutil.ignore_patterns("__pycache__")
        )
        for c in changes["changed_source_files"]:
            dst = staging / c["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OVERLAY / c["path"], dst)
        if target.exists():
            shutil.rmtree(target)
        (staging / "freetoken").replace(target)
    marker = dict(base_version=version, overlay_sha256=sha256(OVERLAY / "changes.json"))
    (output / "tideway-backend.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(
        f"Tideway backend: {target} (FreeToken {version} + {len(changes['changed_source_files'])} Tideway files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
