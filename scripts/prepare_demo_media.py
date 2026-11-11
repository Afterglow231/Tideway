#!/usr/bin/env python3
"""Prepare Tideway demo videos and GIF previews from the original rendered clips."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [
    dict(
        source="demo_split_v1.mp4",
        name="single-request",
        scale=1,
        box=(184, 176, 155, 40),
        text=(192, 182, 28),
        color="0xeef1f5",
        font="DejaVuSans-Bold.ttf",
        speed=4,
        kind="recorded generation",
    ),
    dict(
        source="session_S09_lr_720p.mp4",
        name="agent-session",
        scale=2 / 3,
        box=(178, 176, 160, 40),
        text=(184, 182, 28),
        color="0xeef1f5",
        font="DejaVuSans-Bold.ttf",
        speed=70,
        kind="measured session replay",
        crf=25,
    ),
    dict(
        source="race_ds64.mp4",
        name="request-race",
        scale=1,
        box=(72, 278, 215, 58),
        text=(80, 288, 36),
        color="white",
        font="DejaVuSans-Bold.ttf",
        speed=10,
        kind="measured request replay",
    ),
    dict(
        source="residency_ds64.mp4",
        name="expert-residency",
        scale=1,
        box=(72, 510, 215, 45),
        text=(80, 518, 32),
        color="white",
        font="DejaVuSans-Bold.ttf",
        speed=None,
        kind="expert-residency visualization",
    ),
    dict(
        source="fig5_flat.mp4",
        name="phase-aware-layout",
        scale=1,
        box=(996, 326, 240, 52),
        text=(1008, 332, 36),
        color="white",
        font="DejaVuSerif-Bold.ttf",
        speed=None,
        kind="mechanism illustration",
    ),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="original clip directory")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/media")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--fonts", type=Path, default=Path("/usr/share/fonts/truetype/dejavu"))
    parser.add_argument(
        "--only",
        action="append",
        choices=[clip["name"] for clip in CLIPS],
        help="regenerate a selected clip; repeat to select more than one",
    )
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("supply --ffmpeg or install ffmpeg on PATH")
    source, output = args.source.expanduser().resolve(), args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run(arguments):
        subprocess.run(
            [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True
        )

    def prepare(clip):
        original = source / clip["source"]
        target = output / (clip["name"] + ".mp4")
        factor = clip["scale"]
        x, y, w, h = [round(n * factor) for n in clip["box"]]
        tx, ty, size = [round(n * factor) for n in clip["text"]]
        font = args.fonts / clip["font"]
        if not font.is_file():
            raise FileNotFoundError(font)
        escaped_font = str(font).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filters = (
            f"drawbox=x={x}:y={y}:w={w}:h={h}:color={clip['color']}:t=fill,"
            f"drawtext=fontfile='{escaped_font}':text=Tideway:x={tx}:y={ty}:"
            f"fontsize={size}:fontcolor=0x171a1e"
        )
        run(
            [
                "-i",
                str(original),
                "-vf",
                filters,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                str(clip.get("crf", 20)),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
        poster = output / (clip["name"] + "-poster.png")
        run(["-sseof", "-0.08", "-i", str(target), "-frames:v", "1", "-update", "1", str(poster)])
        products = [target, poster]
        if clip["name"] == "single-request":
            preview = output / "single-request-preview.gif"
            palette = (
                "fps=8,scale=960:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3"
            )
            run(
                [
                    "-ss",
                    "24",
                    "-t",
                    "12",
                    "-i",
                    str(target),
                    "-filter_complex",
                    palette,
                    "-loop",
                    "0",
                    str(preview),
                ]
            )
            products.append(preview)
        if clip["name"] in (
            "agent-session",
            "request-race",
            "phase-aware-layout",
            "expert-residency",
        ):
            gif = output / (clip["name"] + ".gif")
            # Keep the full session replay compact enough for inline playback.
            fps = 2 if clip["name"] == "agent-session" else 10
            colors = ":max_colors=96" if clip["name"] == "agent-session" else ""
            dither = (
                "none:diff_mode=rectangle"
                if clip["name"] == "agent-session"
                else "bayer:bayer_scale=3"
            )
            palette = (
                f"fps={fps},scale=960:-1:flags=lanczos,split[a][b];"
                f"[a]palettegen=stats_mode=diff{colors}[p];"
                f"[b][p]paletteuse=dither={dither}"
            )
            run(["-i", str(target), "-filter_complex", palette, "-loop", "0", str(gif)])
            products.append(gif)
        print(f"Prepared {clip['name']}", flush=True)
        return dict(
            name=clip["name"],
            kind=clip["kind"],
            playback_speed=clip["speed"],
            source_file=original.name,
            source_sha256=sha256(original),
            label_region=[x, y, w, h],
            preview_excerpt_seconds=[24, 36] if clip["name"] == "single-request" else None,
            changes="Display label updated to Tideway; original timing and numerical overlays retained.",
            files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in products},
        )

    selected = [clip for clip in CLIPS if not args.only or clip["name"] in args.only]
    previous = output / "manifest.json"
    manifest = json.loads(previous.read_text())["clips"] if args.only and previous.is_file() else []
    with ThreadPoolExecutor(max_workers=2) as executor:
        prepared = list(executor.map(prepare, selected))
    combined = {item["name"]: item for item in [*manifest, *prepared]}
    manifest = [combined[clip["name"]] for clip in CLIPS if clip["name"] in combined]
    (output / "manifest.json").write_text(
        json.dumps(dict(schema=1, clips=manifest), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
