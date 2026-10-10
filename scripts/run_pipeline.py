#!/usr/bin/env python3
"""Preprocess videos and extract timestamped low-level features (Task 2). Originals are only read.

Run with the feature environment: .venv/bin/python scripts/run_pipeline.py ... (never .venv-models).

    python scripts/run_pipeline.py                         # defaults: data/raw and data/sample
    python scripts/run_pipeline.py --input path/to/dir_or_file [--input ...]
    python scripts/run_pipeline.py --force                 # ignore caches and rebuild
    python scripts/run_pipeline.py --window 1.0 --proxy-fps 15

Writes artifacts/cache/<video_id>/<key>/ (proxy.mp4, audio.wav, preprocess.json)
and artifacts/features/ (per-video segments.csv, frames.csv, report.json, plus
segments_all.csv, first5s_summary.csv, run_summary.json, rebuilt from every video in the root;
feature_env.json records the OpenCV/NumPy/FFmpeg that built the root's features).
Exit code: 0 if every video of this run was processed (ok or partial), 1 if any failed,
2 if FFmpeg is missing or an input does not exist, 3 if this is the wrong environment.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import PipelineConfig  # noqa: E402
from hookline.media import find_tool  # noqa: E402
from hookline.pipeline import default_inputs, run  # noqa: E402
from hookline.provenance import FeatureEnvironmentError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", type=Path, help="video file or directory (repeatable); default: data/raw and data/sample")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts", help="output root (default: artifacts/)")
    ap.add_argument("--force", action="store_true", help="rebuild proxies, audio and features even if cached")
    ap.add_argument("--window", type=float, help="segment window in seconds (default 0.5)")
    ap.add_argument("--proxy-fps", type=int, help="proxy frame rate (default 10)")
    ap.add_argument("--limit", type=int, help="process at most N videos")
    ap.add_argument("--new-feature-env", action="store_true",
                    help="this root's features were built with other OpenCV/NumPy/FFmpeg versions: rebuild them here")
    args = ap.parse_args(argv)

    if not (find_tool("ffmpeg") and find_tool("ffprobe")):
        print("ffmpeg and ffprobe are required (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg)", file=sys.stderr)
        return 2

    cfg = PipelineConfig()
    if args.window:
        cfg = replace(cfg, window_seconds=args.window)
    if args.proxy_fps:
        cfg = replace(cfg, proxy_fps=args.proxy_fps)

    missing = [str(p) for p in args.input or [] if not p.expanduser().exists()]
    if missing:
        print(f"input not found: {', '.join(missing)} (paths are relative to {Path.cwd()})", file=sys.stderr)
        return 2
    inputs = args.input or default_inputs()
    print("inputs: " + ", ".join(str(p) for p in inputs))
    try:
        summary = run(inputs, cfg, root=args.artifacts, force=args.force, limit=args.limit, new_feature_env=args.new_feature_env)
    except FeatureEnvironmentError as exc:
        print(f"wrong environment for Task 2 features: {exc}", file=sys.stderr)
        return 3
    c, total = summary["last_run"]["counts"], summary["counts"]
    print(f"\n{c['videos']} videos: {c['ok']} ok, {c['partial']} partial, {c['failed']} failed"
          f" (artifacts root now holds {total['videos']} videos)")
    print(f"outputs: {Path(summary['artifacts_root']) / 'features'}")
    return 1 if c["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
