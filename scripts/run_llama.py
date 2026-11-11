#!/usr/bin/env python3
"""Start the pinned llama-server baseline with an optional GGUF RSS governor."""

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=ROOT / "build/llama.cpp")
    parser.add_argument("--ram-gib", type=float)
    parser.add_argument("--headroom-mib", type=int, default=512)
    parser.add_argument(
        "--out", type=Path, required=True, help="fresh directory for server and governor logs"
    )
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    model, build = args.model.expanduser().resolve(), args.build.expanduser().resolve()
    server = build / "bin/llama-server"
    if not model.is_file() or not server.is_file():
        parser.error("model or llama-server is missing; build the baseline and supply a GGUF file")
    if args.ram_gib is not None and (
        args.headroom_mib <= 0 or args.ram_gib * 1024 <= 2 * args.headroom_mib
    ):
        parser.error("RAM budget must exceed twice the positive governor headroom")
    library = build / "libllama_budget.so"
    if args.ram_gib is not None and not library.is_file():
        parser.error("libllama_budget.so is missing; run baselines/llama_cpp/build.sh")
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    if args.ram_gib is not None:
        budget = int(args.ram_gib * 2**30)
        control = output / "budget_control.txt"
        control.write_text(f"{budget} 0\n")
        environment.update(
            TIDEWAY_LLAMA_MODEL=str(model),
            TIDEWAY_LLAMA_RSS_BUDGET_BYTES=str(budget),
            TIDEWAY_LLAMA_HEADROOM_BYTES=str(args.headroom_mib * 2**20),
            TIDEWAY_LLAMA_BUDGET_LOG=str(output / "governor.jsonl"),
            TIDEWAY_LLAMA_BUDGET_CONTROL=str(control),
            TIDEWAY_LLAMA_BUDGET_ACK=str(output / "budget_ack.json"),
            TIDEWAY_LLAMA_EXPECTED_PARENT=str(os.getpid()),
            LD_PRELOAD=":".join(filter(None, [str(library), environment.get("LD_PRELOAD")])),
        )
    extra = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [str(server), "-m", str(model), *extra],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
