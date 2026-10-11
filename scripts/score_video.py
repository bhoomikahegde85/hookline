#!/usr/bin/env python3
"""0-100 hook and breakout scores for one analysed video, each with the measured accuracy of the model behind it.

    python scripts/score_video.py FC002 --artifacts artifacts/e2e --ttcc-reference artifacts/hook
    python scripts/score_video.py NEWREEL --artifacts artifacts/demo

Reads saved files only (no Space, no model, no upload). Writes <artifacts>/features/<id>/hook_scores.json. The reference set is the 100
regular reels of 01-Hookline (labels: real views relative to the follower band's median). See src/hookline/hook_score.py for what each
score means, and 01-Hookline/evaluation/ for the accuracy behind it. Scores are model predictions, not audience data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.hook_score import compute_scores, write_scores  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "e2e", help="artifact root holding features/<video_id>/")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline")
    ap.add_argument("--reference-features", type=Path, default=PROJECT_ROOT / "artifacts" / "company" / "features")
    ap.add_argument("--tribe-reference", type=Path, default=PROJECT_ROOT / "artifacts" / "tribe", help="artifact root with TRIBE outputs of the reference reels (popularity score)")
    ap.add_argument("--ttcc-reference", type=Path, default=PROJECT_ROOT / "artifacts" / "hook", help="artifact root with TTCC hook predictions of the reference reels")
    ap.add_argument("--eval", type=Path, default=None, help="evaluation folder (default <data>/evaluation)")
    args = ap.parse_args(argv)
    d = args.artifacts / "features" / args.video_id
    if not d.is_dir():
        print(f"no artifacts for {args.video_id} in {d}", file=sys.stderr)
        return 1
    ttcc_ref = args.ttcc_reference / "features" if (args.ttcc_reference / "features").is_dir() else None
    tribe_ref = args.tribe_reference / "features" if (args.tribe_reference / "features").is_dir() else None
    scores = compute_scores(d, args.data, args.reference_features, ttcc_ref, args.eval or args.data / "evaluation", tribe_root=tribe_ref)
    if scores.get("status") != "ok":
        print(scores["reason"], file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in scores.items() if k != "note"}, indent=2))
    print(f"written: {write_scores(d, scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
