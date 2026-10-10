#!/usr/bin/env python3
"""TTCC retention predictions for explicitly selected videos, then the per-video report.

    # one video (features reused or built first; TTCC needs the TTCC environment, see docs/ttcc.md)
    .venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company --enable-ttcc
    # a chosen set: repeat --input (files or directories); --limit caps the count
    .venv/bin/python scripts/run_ttcc.py --input a.mp4 --input b.mp4 --artifacts artifacts/company --enable-ttcc
    # runs on this Mac too: Apple GPU (MPS) or CPU are picked automatically when there is no CUDA
    # in your Hugging Face Space, nothing heavy on this machine (company reels also need --ttcc-approve <id>)
    .venv/bin/python scripts/run_ttcc.py --input my_clip.mp4 --artifacts artifacts/demo --enable-ttcc --ttcc-remote --ttcc-allow-remote
    # TTCC off (the default): only (re)writes video_report.md from the artifacts on disk
    .venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company

TTCC is off unless ENABLE_TTCC=true or --enable-ttcc. It never runs on a default input set: every
video must be named with --input. Per video, steps: validate the input, reuse or build the Task 2
features (--features), run TTCC in the TTCC environment (one model load for the whole batch),
keep the raw output, write the normalised prediction and plot, record the status, write
video_report.md. One video failing never stops the others, and a failure never removes an
earlier successful prediction or any other artifact.

Writes, per video in <artifacts>/features/<video_id>/: ttcc_status.json, and on success
ttcc_retention.json, ttcc_raw.json, ttcc_retention.svg; video_report.md. Run level:
<artifacts>/features/ttcc_run_summary.json.
Exit code: 0 if no video failed, 1 if any failed, 2 for bad inputs or missing FFmpeg,
3 if TTCC is enabled but its environment or checkpoint is unavailable.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import ATTN_IMPLS, DEVICE_MAPS, PipelineConfig, TtccConfig  # noqa: E402
from hookline.dropoffs import write_dropoffs  # noqa: E402
from hookline.media import find_tool  # noqa: E402
from hookline.pipeline import FeaturesUnavailable, discover, load_features, merge_run_summary, run  # noqa: E402
from hookline.provenance import FeatureEnvironmentError  # noqa: E402
from hookline.ttcc import DISABLED, FAILED, SUCCESS, UNAVAILABLE, artifact_paths, run_ttcc  # noqa: E402
from hookline.video_report import REPORT_FILE, write_video_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", type=Path, required=True, help="video file or directory (repeatable; required)")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts", help="output root (default: artifacts/)")
    ap.add_argument("--limit", type=int, help="process at most N videos")
    ap.add_argument("--features", choices=["build", "reuse", "skip"], default="build",
                    help="build: reuse cached Task 2 features, compute missing ones (feature environment only); "
                         "reuse: use existing features, never compute; skip: TTCC only (default build)")
    ap.add_argument("--window", type=float, help="must match run_pipeline.py to reuse its features")
    ap.add_argument("--proxy-fps", type=int, help="must match run_pipeline.py to reuse its features")
    t = ap.add_argument_group("TTCC")
    t.add_argument("--enable-ttcc", action="store_true", help="run TTCC (same as ENABLE_TTCC=true)")
    t.add_argument("--force", action="store_true", help="re-run TTCC even when a prediction with the same cache key exists")
    t.add_argument("--download-models", action="store_true", help="allow downloading the ~9.4 GB checkpoint")
    t.add_argument("--ttcc-python", help="TTCC environment interpreter (default: $TTCC_PYTHON or .venv-ttcc/bin/python)")
    t.add_argument("--ttcc-repo", help="go_viral checkout on branch ttcc-rl (default: $TTCC_REPO or ./go_viral)")
    t.add_argument("--checkpoint-dir", help="local copy of the checkpoint (default: $TTCC_CHECKPOINT_DIR or the Hugging Face cache)")
    t.add_argument("--device-map", default=TtccConfig.device_map, choices=DEVICE_MAPS,
                   help="detect (default: CUDA, else Apple MPS, else CPU), cuda, auto (split over several GPUs), mps or cpu")
    t.add_argument("--attn-impl", default=TtccConfig.attn_impl, choices=ATTN_IMPLS,
                   help="detect (default: flash_attn where FlashAttention 2 runs, else sdpa), flash_attn, sdpa or eager")
    t.add_argument("--dtype", default=TtccConfig.torch_dtype, choices=["bfloat16", "float16", "float32"])
    t.add_argument("--ttcc-remote", action="store_true",
                   help="run TTCC in your Hugging Face Space (endpoint/ttcc_space/) instead of a local environment; "
                        "needs TTCC_ENDPOINT_URL and HF_TOKEN (environment or untracked .env). Nothing runs on this machine")
    t.add_argument("--ttcc-allow-remote", action="store_true", help="required to upload any video to the Space")
    t.add_argument("--ttcc-approve", action="append", default=[], metavar="VIDEO_ID",
                   help="allow uploading this company video (under 01-Hookline/); repeatable")
    t.add_argument("--long-videos", choices=["hook", "windows", "first-60s", "skip"], default=TtccConfig.long_videos,
                   help="hook (default): one pass over the first min(duration, --hook-s) s, giving R(3)/R(5)/R(10); windows: overlapping windows "
                        "merged into one approximate curve (experimental); first-60s: only the first 60 s; skip: no prediction beyond 60 s")
    t.add_argument("--hook-s", type=int, default=TtccConfig.hook_s, help="hook mode: seconds of the opening shown to the model (default 30)")
    t.add_argument("--window-mode", choices=["auto", "fixed"], default=TtccConfig.window_mode,
                   help="auto (default): a video up to --window-s is one pass; a longer one gets a window by its length (20 s up to 45 s of video, 15 s up to 90 s, "
                        "10 s beyond; never under 5 s). fixed: every window is --window-s")
    t.add_argument("--window-s", type=int, default=TtccConfig.window_s,
                   help="largest window / single-pass limit in seconds (default 30: a memory limit of the deployment, not of the model; up to 60 on a bigger GPU)")
    t.add_argument("--stride-s", type=int, default=TtccConfig.stride_s, help="seconds between window starts (default 15; windows overlap by window - stride)")
    t.add_argument("--warmup-s", type=float, default=TtccConfig.warmup_s, help="seconds at the start of each later window that count little in the merge (default 5)")
    t.add_argument("--min-window-s", type=int, default=TtccConfig.min_window_s, help="smallest window the out-of-memory fallback may shrink to (default 10)")
    t.add_argument("--no-shrink-window", action="store_true", help="fail instead of retrying with shorter windows after an out-of-memory error")
    t.add_argument("--timeout", type=float, default=TtccConfig.timeout_s, help="seconds for the whole TTCC runner call")
    t.add_argument("--verify-weights", action="store_true", help="also sha256 the weight shards (~9.4 GB read)")
    args = ap.parse_args(argv)

    missing = [str(p) for p in args.input if not p.expanduser().exists()]
    if missing:
        print(f"input not found: {', '.join(missing)} (paths are relative to {Path.cwd()})", file=sys.stderr)
        return 2
    if not (find_tool("ffmpeg") and find_tool("ffprobe")):
        print("ffmpeg and ffprobe are required (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg)", file=sys.stderr)
        return 2
    try:
        cfg = TtccConfig.from_env(
            enabled=True if args.enable_ttcc else None, python=args.ttcc_python, repo_dir=args.ttcc_repo,
            checkpoint_dir=args.checkpoint_dir, allow_download=args.download_models or None, device_map=args.device_map,
            attn_impl=args.attn_impl, torch_dtype=args.dtype, timeout_s=args.timeout,
            long_videos=args.long_videos.replace("-", "_"), hook_s=args.hook_s, window_mode=args.window_mode, window_s=args.window_s, stride_s=args.stride_s, warmup_s=args.warmup_s,
            min_window_s=args.min_window_s, auto_shrink_window=False if args.no_shrink_window else None, remote=args.ttcc_remote or None,
            allow_remote=args.ttcc_allow_remote or None, approved_ids=tuple(args.ttcc_approve) or None,
            verify_weights_sha256=args.verify_weights or None)
    except ValueError as exc:
        ap.error(str(exc))
    pcfg = PipelineConfig()
    if args.window:
        pcfg = replace(pcfg, window_seconds=args.window)
    if args.proxy_fps:
        pcfg = replace(pcfg, proxy_fps=args.proxy_fps)

    root = args.artifacts
    videos = discover(args.input)[: args.limit or None]
    if not videos:
        print("no videos found in the given inputs", file=sys.stderr)
        return 2
    print(f"{len(videos)} video(s): " + ", ".join(v.video_id for v in videos))
    started = time.time()

    features: dict[str, str] = {}
    if args.features == "build":
        try:
            summary = run(args.input, pcfg, root=root, limit=args.limit, log=lambda *_: None)
            by_id = {r["video_id"]: r for r in summary["videos"]}
            features = {v.video_id: (by_id.get(v.video_id) or {}).get("processing_status") or "missing" for v in videos}
        except FeatureEnvironmentError as exc:
            print(f"features not built here ({exc}); continuing with TTCC, which needs only the video", file=sys.stderr)
            args.features = "reuse"
    if args.features == "reuse":
        for v in videos:
            try:
                features[v.video_id] = load_features(v, pcfg, root)["processing_status"]
            except FeaturesUnavailable as exc:
                features[v.video_id] = f"unavailable: {exc}"
    print(f"features ({args.features}): " + ", ".join(f"{k}={str(s)[:40]}" for k, s in features.items()) if features else
          "features: skipped")

    result = run_ttcc(videos, root, cfg, force=args.force)
    records = []
    for v, rec in zip(videos, result["records"]):
        out_dir = root / "features" / v.video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_dropoffs(out_dir)  # local and cheap: only when a successful TTCC prediction is on disk
        report = write_video_report(out_dir, v.video_id)
        rec.update(features_status=features.get(v.video_id, "skipped"), artifacts=artifact_paths(out_dir) | {REPORT_FILE: str(report)},
                   seconds=round((rec.get("validation_seconds") or 0) + (rec.get("inference_seconds") or 0), 3))
        records.append(rec)
        line = f"{v.video_id}: TTCC {rec['status']}"
        if rec.get("cached"):
            line += " (cached)"
        if rec.get("reason"):
            line += f" - {rec['reason'][:300]}"
        print(line)

    batch = result["batch"]
    summary = merge_run_summary(
        root / "features" / "ttcc_run_summary.json", records, "status", statuses=(SUCCESS, FAILED, UNAVAILABLE, DISABLED),
        last_run={"inputs": [str(p) for p in args.input], "features_mode": args.features, "config": cfg.to_dict(),
                  "batch": batch, "wall_seconds": round(time.time() - started, 2)},
    )
    c = summary["last_run"]["counts"]
    print(f"\n{c['videos']} videos: {c[SUCCESS]} success, {c[FAILED]} failed, {c[UNAVAILABLE]} unavailable, {c[DISABLED]} disabled"
          + (f"; model load {batch['model_load_seconds']} s, runner {batch['runner_seconds']} s" if batch.get("runner_seconds") else "")
          + (f"; device {batch['runtime']['device_map']}, attention {batch['runtime']['attn_impl']}" if batch.get("runtime") else ""))
    print(f"summary: {root / 'features' / 'ttcc_run_summary.json'}")
    if c[FAILED]:
        return 1
    if any(r.get("status") == UNAVAILABLE and r.get("stage") == "environment" for r in records):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
