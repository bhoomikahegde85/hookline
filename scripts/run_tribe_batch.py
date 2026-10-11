#!/usr/bin/env python3
"""TRIBE v2 brain-response predictions for the first N seconds of many reels, on your own Hugging Face Space (nothing runs on this machine).

    python scripts/run_tribe_batch.py --all --seconds 10 --artifacts artifacts/tribe --allow-remote --approve-all

For each reel: cut the first N seconds into <artifacts>/trim/<id>.mp4 (local ffmpeg), send that clip to the Space (``--tribe-api job``), and store
tribe.json + tribe_preds.npy under <artifacts>/features/<id>/. Finished reels are skipped on a rerun. TRIBE predicts fMRI-like responses of an average subject,
NOT retention or virality; scripts/eval_views.py --tribe-artifacts <artifacts> tests whether its summaries help predict views. Licence: CC BY-NC 4.0 (non-commercial use).
Company reels need --approve-all (or --approve <id>): the clips are uploaded to your private Space.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import TribeConfig, file_sha256  # noqa: E402
from hookline.context import MediaContext  # noqa: E402
from hookline.tribe import cached_tribe, extract_tribe, tribe_key, write_tribe  # noqa: E402
from hookline.views_eval import load_dataset  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def trim(src: Path, dst: Path, seconds: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="every reel of 01-Hookline (face, faceless, outliers)")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "tribe")
    ap.add_argument("--seconds", type=int, default=10, help="seconds of the opening to send (default 10)")
    ap.add_argument("--allow-remote", action="store_true", help="required: the clips are uploaded to your TRIBE Space")
    ap.add_argument("--approve-all", action="store_true", help="approve uploading every selected company reel's clip")
    ap.add_argument("--approve", action="append", default=[], metavar="VIDEO_ID")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--job-wait", type=float, default=1800.0)
    args = ap.parse_args(argv)
    rows = load_dataset(args.data)
    by_id = {r["video_id"]: r for r in rows}
    ids = [r["video_id"] for r in rows] if args.all else args.video_ids
    unknown = [i for i in ids if i not in by_id]
    if unknown or not ids:
        ap.error(f"unknown or no video ids: {unknown}")
    if not args.allow_remote:
        ap.error("--allow-remote is required (the clips are uploaded to your Space)")
    approved = set(ids) if args.approve_all else set(args.approve)
    missing = [i for i in ids if i not in approved]
    if missing:
        ap.error(f"not approved for upload: {missing[:5]}{'...' if len(missing) > 5 else ''} (use --approve-all or --approve <id>)")
    cfg = TribeConfig.from_env(enabled=True, allow_remote=True, approved_ids=tuple(approved), api="job", job_wait_s=args.job_wait)
    ids = ids[: args.limit] if args.limit else ids
    done = failed = skipped = 0
    t0 = time.time()
    for n, vid in enumerate(ids, 1):
        out = args.artifacts / "features" / vid
        out.mkdir(parents=True, exist_ok=True)
        clip = args.artifacts / "trim" / f"{vid}.mp4"
        try:
            trim(Path(by_id[vid]["video_path"]), clip, args.seconds)
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"[{n}/{len(ids)}] {vid}: could not cut the clip ({exc})", flush=True)
            continue
        ctx = MediaContext(video_id=vid, source_path=str(clip), source_sha256=file_sha256(clip), prep_key=None, duration=float(args.seconds))
        if cached_tribe(out, tribe_key(ctx, cfg)):
            skipped += 1
            print(f"[{n}/{len(ids)}] {vid}: cached", flush=True)
            continue
        art, preds = extract_tribe(ctx, cfg)
        write_tribe(out, art, preds, write_json)
        ok = bool(art and art.get("measured"))
        done += ok
        failed += not ok
        print(f"[{n}/{len(ids)}] {vid}: {'ok' if ok else 'FAILED: ' + str((art or {}).get('reason'))[:200]} ({time.time() - t0:.0f} s elapsed)", flush=True)
    print(f"done {done}, cached {skipped}, failed {failed}; outputs in {args.artifacts / 'features'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
