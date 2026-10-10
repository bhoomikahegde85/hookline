#!/usr/bin/env python3
"""Is TTCC reading the video? Sensitivity checks for the hook predictions (R(1), R(3), R(5), R(10)), on your TTCC Space.

    python scripts/ttcc_sanity.py FC002 FL031 FL002 --allow-remote --approve FC002 --approve FL031 --approve FL002
    python scripts/ttcc_sanity.py FC002 FL031 --plan-only        # build the variants and print the plan; nothing is sent

For each selected reel the opening the model sees (first min(duration, --hook-s) s) is predicted as:
- original      the reel as the hook analysis sends it
- reencoded     the same opening cut out and re-encoded (the noise floor: same content, different file)
- black         the same opening with every frame black (audio is off in TTCC, so the model gets no content)
- shuffled      the same opening with its 1-second chunks in a random (seeded) order
- prompt_<P>    the original, but the model is told the ad is P seconds long (supplied-duration sensitivity)
and the originals are compared with each other (different videos). If black / shuffled move R(3) no more than re-encoding does, the
predictions are not driven by the content. Results: 01-Hookline/evaluation/sanity_checks/<timestamp>/ (never overwritten).

Variants are derived from company reels, so they are written inside 01-Hookline/ and are uploaded only if their source reel is approved
(--approve). Every pass runs on your Hugging Face Space (paid GPU time); nothing runs a model on this machine.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import TtccConfig  # noqa: E402
from hookline.pipeline import VideoInput  # noqa: E402
from hookline.ttcc import SUCCESS, hook_metrics, plan_hook, run_ttcc, validate_video  # noqa: E402
from hookline.views_eval import load_dataset, new_run_dir, now_stamp  # noqa: E402

KEYS = ("R1", "R3", "R5", "R10")
VARIANTS = ("reencoded", "black", "shuffled")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_variants(src: Path, vid: str, seconds: int, out: Path, seed: int) -> dict[str, Path]:
    """Re-encoded, black and chunk-shuffled copies of the first ``seconds`` s (video only; TTCC runs with audio off)."""
    out.mkdir(parents=True, exist_ok=True)
    enc = ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18"]
    paths = {k: out / f"{vid}__{k}.mp4" for k in VARIANTS}
    ffmpeg("-i", str(src), "-t", str(seconds), *enc, str(paths["reencoded"]))
    ffmpeg("-i", str(paths["reencoded"]), "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill", *enc, str(paths["black"]))
    chunks = out / f"{vid}__chunks"
    chunks.mkdir(exist_ok=True)
    for i in range(seconds):
        ffmpeg("-ss", str(i), "-i", str(paths["reencoded"]), "-t", "1", *enc, str(chunks / f"{i:03d}.mp4"))
    order = list(range(seconds))
    random.Random(f"{seed}:{vid}").shuffle(order)
    (chunks / "list.txt").write_text("".join(f"file '{i:03d}.mp4'\n" for i in order))
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(chunks / "list.txt"), *enc, str(paths["shuffled"]))
    (out / f"{vid}__shuffle_order.json").write_text(json.dumps(order))
    return paths


def hook_of(root: Path, vid: str) -> dict | None:
    p = root / "features" / vid / "ttcc_retention.json"
    if not p.is_file():
        return None
    art = json.loads(p.read_text())
    if art.get("status") != SUCCESS:
        return None
    return art.get("hook") or hook_metrics(art["retention"], (art.get("inference") or {}).get("window_end_s"))


def diff(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return None
    return {f"abs_d{k}": abs(a[k] - b[k]) for k in KEYS if a.get(k) is not None and b.get(k) is not None}


def summarize(results: dict) -> dict:
    """Mean absolute differences per comparison kind, and plain-language flags (no verdict beyond what the numbers show)."""
    kinds: dict[str, list[dict]] = {}
    for vid, r in results["per_video"].items():
        for kind, d in (r.get("vs_original") or {}).items():
            if d:
                kinds.setdefault(kind, []).append(d)
    kinds["different_videos"] = [d for d in results["between_videos"] if d]
    mean = {k: {m: sum(x[m] for x in v if m in x) / max(1, sum(m in x for x in v)) for m in (f"abs_d{k_}" for k_ in KEYS)} | {"pairs": len(v)}
            for k, v in kinds.items() if v}
    flags = []
    noise = (mean.get("reencoded") or {}).get("abs_dR3")
    for kind in ("black", "shuffled", "different_videos"):
        m = (mean.get(kind) or {}).get("abs_dR3")
        if m is not None and noise is not None:
            flags.append(f"{kind}: mean |dR3| {m:.4f} vs re-encoding noise {noise:.4f} -> "
                         + ("clearly larger (content affects R(3))" if m > 3 * noise + 0.005 else "not clearly larger than noise (R(3) may not depend on this)"))
    return {"mean_abs_differences": mean, "flags": flags}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline")
    ap.add_argument("--out", type=Path, default=None, help="default <data>/evaluation/sanity_checks")
    ap.add_argument("--hook-s", type=int, default=TtccConfig.hook_s)
    ap.add_argument("--prompts", default="45,60", help="told lengths (s) for the supplied-duration check, comma-separated (5..60)")
    ap.add_argument("--allow-remote", action="store_true", help="required: the openings and variants are uploaded to your Space")
    ap.add_argument("--approve", action="append", default=[], metavar="VIDEO_ID", help="approve uploading this company reel (and its variants)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--price-per-hour", type=float, default=1.80, help="Space hardware price for the cost estimate (L40S: 1.80)")
    ap.add_argument("--plan-only", action="store_true", help="build variants and print the plan; send nothing")
    args = ap.parse_args(argv)

    meta = {r["video_id"]: r for r in load_dataset(args.data)}
    unknown = [v for v in args.video_ids if v not in meta]
    if unknown:
        ap.error(f"unknown video id(s): {unknown}")
    prompts = [int(p) for p in args.prompts.split(",") if p.strip()]
    stamp = now_stamp()
    run_dir = new_run_dir(args.out or (args.data / "evaluation" / "sanity_checks"), stamp)
    variants_dir = run_dir / "variants"
    plan, inputs = {}, []
    for vid in args.video_ids:
        src = Path(meta[vid]["video_path"])
        info, problem = validate_video(src)
        if problem:
            ap.error(f"{vid}: {problem}")
        prompt_T, shown, _ = plan_hook(info["duration_s"], args.hook_s)
        seconds = int(shown or prompt_T)
        paths = make_variants(src, vid, seconds, variants_dir, args.seed)
        plan[vid] = {"duration_s": info["duration_s"], "seconds_shown": seconds, "variants": {k: str(p.relative_to(run_dir)) for k, p in paths.items()}}
        inputs.append(VideoInput(vid, src))
        inputs += [VideoInput(p.stem, p) for p in paths.values()]
    approved = tuple(args.approve) + tuple(f"{v}__{k}" for v in args.approve for k in VARIANTS)
    passes = len(inputs) + len(prompts) * len(args.video_ids)
    plan_doc = {"timestamp_utc": stamp, "videos": plan, "prompts": prompts, "hook_s": args.hook_s, "passes": passes, "seed": args.seed}
    (run_dir / "plan.json").write_text(json.dumps(plan_doc, indent=2))
    print(f"{passes} TTCC passes planned ({len(inputs)} openings/variants + {len(prompts)} told-length variations per reel); folder {run_dir}")
    if args.plan_only:
        print("--plan-only: nothing sent")
        return 0
    if not args.allow_remote:
        ap.error("--allow-remote is required (the openings and variants are uploaded to your Space)")
    unapproved = [v for v in args.video_ids if v not in args.approve]
    if unapproved:  # every reel here is a company reel; its variants may sit outside 01-Hookline/ (--out), so approval is checked here too
        ap.error(f"not approved for upload: {unapproved} (add --approve <id> for each)")

    cfg = TtccConfig.from_env(enabled=True, remote=True, allow_remote=True, approved_ids=approved, long_videos="hook", hook_s=args.hook_s)
    t0 = time.time()
    main_run = run_ttcc(inputs, run_dir / "artifacts", cfg, log=print)
    prompt_runs = {}
    for P in prompts:
        prompt_runs[P] = run_ttcc([VideoInput(v, Path(meta[v]["video_path"])) for v in args.video_ids], run_dir / f"artifacts_prompt{P}",
                                  replace(cfg, hook_prompt_s=P), log=print)
    wall = time.time() - t0
    runner_s = (main_run["batch"].get("runner_seconds") or 0) + sum((r["batch"].get("runner_seconds") or 0) for r in prompt_runs.values())

    results = {"per_video": {}, "between_videos": []}
    originals = {v: hook_of(run_dir / "artifacts", v) for v in args.video_ids}
    for v in args.video_ids:
        hooks = {"original": originals[v]} | {k: hook_of(run_dir / "artifacts", f"{v}__{k}") for k in VARIANTS} | \
                {f"prompt_{P}": hook_of(run_dir / f"artifacts_prompt{P}", v) for P in prompts}
        results["per_video"][v] = {"hooks": hooks, "vs_original": {k: diff(originals[v], h) for k, h in hooks.items() if k != "original"}}
    ids = [v for v in args.video_ids if originals[v]]
    results["between_videos"] = [diff(originals[a], originals[b]) | {"pair": [a, b]} for i, a in enumerate(ids) for b in ids[i + 1:]]
    status = {r["video_id"]: {"status": r["status"], "reason": r.get("reason")} for r in main_run["records"]}
    status |= {f"{r['video_id']}@prompt{P}": {"status": r["status"], "reason": r.get("reason")} for P, pr in prompt_runs.items() for r in pr["records"]}
    out = {"kind": "sensitivity check of TTCC predictions (model outputs compared with each other; no ground truth involved)",
           "plan": plan_doc, "status": status, "results": results, "summary": summarize(results),
           "cost": {"runner_seconds": round(runner_s, 1), "wall_seconds": round(wall, 1), "price_per_hour": args.price_per_hour,
                    "estimated_usd_runner_time": round(runner_s / 3600 * args.price_per_hour, 2),
                    "note": "estimate from the runner's own time; the Space bills while it is running (startup and idle included): check the "
                            "organisation's billing page for the actual amount"}}
    (run_dir / "sanity_results.json").write_text(json.dumps(out, indent=2))
    with (run_dir / "sanity_results.csv").open("w") as f:
        f.write("video_id,variant," + ",".join(KEYS) + "," + ",".join(f"abs_d{k}" for k in KEYS) + "\n")
        for v, r in results["per_video"].items():
            for kind, h in r["hooks"].items():
                d = r["vs_original"].get(kind) or {}
                f.write(f"{v},{kind}," + ",".join("" if not h or h.get(k) is None else f"{h[k]:.6f}" for k in KEYS) + ","
                        + ",".join("" if f"abs_d{k}" not in d else f"{d[f'abs_d{k}']:.6f}" for k in KEYS) + "\n")
    (run_dir.parent / "LATEST").write_text(run_dir.name + "\n")
    print(json.dumps(out["summary"], indent=2))
    print(f"cost estimate: ~${out['cost']['estimated_usd_runner_time']} of runner time ({runner_s:.0f} s); see the billing page for the actual amount")
    print(f"results: {run_dir / 'sanity_results.json'}")
    return 0 if all(s["status"] == SUCCESS for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
