#!/usr/bin/env python3
"""TRIBE v2 brain-response prediction for the opening seconds of ONE video, on your own Hugging Face Space (nothing runs on this machine).

    .venv/bin/python scripts/run_tribe.py --input VIDEO.mp4 --artifacts artifacts/e2e [--seconds 10]

Cuts the first N seconds (local ffmpeg), sends that clip to the Space and stores tribe.json + tribe_preds.npy under <artifacts>/features/<video_id>/.
Feeds the separate TRIBE popularity score (scripts/score_video.py); it is not combined with TTCC. TRIBE predicts fMRI-like responses of an average
subject, not retention or virality. Licence: CC BY-NC 4.0 (non-commercial use). Needs TRIBE_ENDPOINT_URL and HF_TOKEN (environment or .env).
Exit code 0 only when TRIBE produced a measured output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import TribeConfig, file_sha256  # noqa: E402
from hookline.context import MediaContext  # noqa: E402
from hookline.tribe import cached_tribe, extract_tribe, tribe_key, write_tribe  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--job-wait", type=float, default=1800.0)
    args = ap.parse_args(argv)
    video = args.input.expanduser().resolve()
    vid = video.stem
    out = args.artifacts / "features" / vid
    out.mkdir(parents=True, exist_ok=True)
    clip = args.artifacts / "trim" / f"{vid}.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    tmp = clip.with_suffix(".part.mp4")
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-t", str(args.seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"{vid}: could not cut the clip ({exc})")
        return 1
    tmp.replace(clip)
    dur = min(float(args.seconds), _duration(clip))
    cfg = TribeConfig.from_env(enabled=True, allow_remote=True, approved_ids=(vid,), api="job", job_wait_s=args.job_wait)
    ctx = MediaContext(video_id=vid, source_path=str(clip), source_sha256=file_sha256(clip), prep_key=None, duration=dur)
    if cached_tribe(out, tribe_key(ctx, cfg)):
        print(f"{vid}: TRIBE output cached")
        return 0
    art, preds = extract_tribe(ctx, cfg)
    write_tribe(out, art, preds, write_json)
    ok = bool(art and art.get("measured"))
    print(f"{vid}: TRIBE {'ok' if ok else 'FAILED: ' + str((art or {}).get('reason'))}")
    return 0 if ok else 1


def _duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 10.0


if __name__ == "__main__":
    sys.exit(main())
