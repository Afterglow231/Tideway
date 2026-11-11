# Demo media

These clips accompany the project README.

| Clip | Type | Playback |
|---|---|---|
| `single-request.mp4` | Recorded single-request generation, Tideway and llama.cpp | 4× |
| `single-request-preview.gif` | A 12-second excerpt from the single-request clip (24–36 s) | 4× |
| `agent-session.mp4` / `.gif` | Measured replay of all 29 turns in session S09 | 70× |
| `request-race.mp4` / `.gif` | Measured five-system request replay; the README's opening demo | 10× |
| `expert-residency.mp4` / `.gif` | RAM residency and SSD reads across prefill chunks | Visualization |
| `phase-aware-layout.mp4` / `.gif` | Expert placement during prefill and decode | Mechanism illustration |

The video captions retain the original numeric results and playback rates.
Only the project display labels were updated to Tideway. `manifest.json` records
the original file hashes, edited label regions and output hashes.

The README's collapsed demos use animated GIFs. The agent-session GIF includes
the full replay at 2 frames per second; its MP4 retains 15 frames per second.
MP4 links provide the full videos. PNG posters contain the final video frames.

To regenerate from the original rendered clips, with FFmpeg on PATH:

```bash
python scripts/prepare_demo_media.py --source /path/to/original/clips
```

Use `--only agent-session` to regenerate one clip, `--ffmpeg` for an explicit
executable, and `--fonts` for a DejaVu font directory.
