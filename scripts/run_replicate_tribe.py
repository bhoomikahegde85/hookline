#!/usr/bin/env python3
"""Third-party TRIBE v2 "virality" wrapper on Replicate (prakhar-bhartiya/meta-tribev2-social-media-content-signal), for the reference reels.

    python scripts/run_replicate_tribe.py --all --seconds 30 --allow-remote --approve-all
    python scripts/run_replicate_tribe.py --all --limit 2 --allow-remote --approve-all    # try two first (costs money per run)

For each reel: cut the first N seconds (local ffmpeg, <=60 s as the wrapper requires), upload that clip to Replicate, run the model, and store the
raw output under <artifacts>/features/<id>/replicate.json. Finished reels are skipped on a rerun. Needs REPLICATE_API_TOKEN or REPLICATE_TOKEN (environment or the
untracked .env). The clips leave this machine and go to Replicate, a third party: --allow-remote and --approve-all are both required.
The score is the wrapper author's fixed-weight heuristic over TRIBE v2 (CC BY-NC 4.0), not a measured or validated quantity.
scripts/eval_replicate.py tests it against real views.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.tribe import load_secret  # noqa: E402
from hookline.views_eval import load_dataset  # noqa: E402

API = "https://api.replicate.com/v1"
MODEL = "prakhar-bhartiya/meta-tribev2-social-media-content-signal"
SCORE_KEYS = ("virality_score", "viral_score", "score", "overall", "overall_score", "total_score", "headline_score")


def headline_score(output) -> float | None:
    """The 0-100 headline score from the model's output (the exact field name is not documented, so a few likely names are tried)."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    if not isinstance(output, dict):
        return None
    for k in SCORE_KEYS:
        v = output.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, dict):
            for kk in ("value", "score", "total"):
                if isinstance(v.get(kk), (int, float)):
                    return float(v[kk])
    return None


def trim(src: Path, dst: Path, seconds: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(dst)


class Replicate:
    def __init__(self, token: str, session: requests.Session | None = None):
        self.s = session or requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def latest_version(self) -> str:
        r = self.s.get(f"{API}/models/{MODEL}", timeout=60)
        r.raise_for_status()
        return r.json()["latest_version"]["id"]

    def upload(self, path: Path) -> str:
        with path.open("rb") as f:
            r = self.s.post(f"{API}/files", files={"content": (path.name, f, "video/mp4")}, timeout=300)
        r.raise_for_status()
        return r.json()["urls"]["get"]

    def run(self, version: str, video_url: str, wait_s: float) -> dict:
        r = self.s.post(f"{API}/predictions", json={"version": version, "input": {"video": video_url}}, headers={"Prefer": "wait=60"}, timeout=120)
        r.raise_for_status()
        pred, t0 = r.json(), time.time()
        while pred["status"] not in ("succeeded", "failed", "canceled"):
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"prediction {pred['id']} still {pred['status']} after {wait_s:.0f} s")
            time.sleep(3)
            g = self.s.get(pred["urls"]["get"], timeout=60)
            g.raise_for_status()
            pred = g.json()
        if pred["status"] != "succeeded":
            raise RuntimeError(f"prediction {pred['status']}: {pred.get('error')}")
        return pred


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="every reel of 01-Hookline (face, faceless, outliers)")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "replicate")
    ap.add_argument("--seconds", type=int, default=30, help="seconds of the opening to send, at most 60 (default 30, the same hook window as TTCC)")
    ap.add_argument("--allow-remote", action="store_true", help="required: the clips are uploaded to Replicate (a third party)")
    ap.add_argument("--approve-all", action="store_true", help="approve uploading every selected company reel's clip")
    ap.add_argument("--approve", action="append", default=[], metavar="VIDEO_ID")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wait", type=float, default=900.0)
    args = ap.parse_args(argv)
    if not 1 <= args.seconds <= 60:
        ap.error("--seconds must be 1..60 (the wrapper accepts clips up to 60 s)")
    rows = load_dataset(args.data)
    by_id = {r["video_id"]: r for r in rows}
    ids = [r["video_id"] for r in rows] if args.all else args.video_ids
    unknown = [i for i in ids if i not in by_id]
    if unknown or not ids:
        ap.error(f"unknown or no video ids: {unknown}")
    if not args.allow_remote:
        ap.error("--allow-remote is required (the clips are uploaded to Replicate)")
    approved = set(ids) if args.approve_all else set(args.approve)
    missing = [i for i in ids if i not in approved]
    if missing:
        ap.error(f"not approved for upload: {missing[:5]}{'...' if len(missing) > 5 else ''} (use --approve-all or --approve <id>)")
    token = load_secret("REPLICATE_API_TOKEN") or load_secret("REPLICATE_TOKEN")
    if not token:
        ap.error("REPLICATE_API_TOKEN (or REPLICATE_TOKEN) not set (environment or untracked .env)")
    ids = ids[: args.limit] if args.limit else ids
    rep = Replicate(token)
    version = rep.latest_version()
    print(f"model {MODEL} version {version[:12]}; {len(ids)} reels, first {args.seconds} s each")
    done = failed = skipped = 0
    for n, vid in enumerate(ids, 1):
        out = args.artifacts / "features" / vid / "replicate.json"
        if out.exists():
            skipped += 1
            continue
        try:
            clip = args.artifacts / "trim" / f"{vid}.mp4"
            trim(Path(by_id[vid]["video_path"]), clip, args.seconds)
            pred = rep.run(version, rep.upload(clip), args.wait)
            score = headline_score(pred.get("output"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"video_id": vid, "model": MODEL, "version": version, "clip_seconds": args.seconds, "score": score,
                                       "output": pred.get("output"), "predict_time_s": (pred.get("metrics") or {}).get("predict_time"),
                                       "note": "third-party heuristic over TRIBE v2; not validated"}, indent=2))
            done += 1
            print(f"[{n}/{len(ids)}] {vid}: score {score}" + ("" if score is not None else "  (headline field not recognised: see output in the json)"))
        except Exception as exc:  # one bad reel must not stop the batch
            failed += 1
            print(f"[{n}/{len(ids)}] {vid}: FAILED {type(exc).__name__}: {exc}")
    print(f"done {done}, skipped {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
