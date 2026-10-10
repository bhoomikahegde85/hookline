#!/usr/bin/env python3
"""Compare TTCC, TRIBE-derived features and both combined against real audience-retention labels.

    python scripts/eval_retention.py --labels labels.csv --artifacts artifacts/eval --target ecr --check-alignment FC002
    python scripts/eval_retention.py --labels labels.csv --artifacts artifacts/eval --target r5 --out report.json

labels.csv: columns video_id,label (a real measurement: SnapUGC ECR, or a point of a measured retention curve).
Needs, per video in --artifacts: features/<id>/ttcc_retention.json (scripts/run_ttcc.py) and tribe.json + tribe_preds.npy
(scripts/run_speech_ocr.py --tribe). Views and follower counts are NOT retention labels; do not use them here.
Reads saved files only: no Space, no model, no upload. See src/hookline/evaluation.py for the method and its limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.evaluation import TARGETS, check_alignment, collect, compare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    ap.add_argument("--target", choices=sorted(TARGETS), default="ecr")
    ap.add_argument("--id-col", default="video_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-alignment", metavar="VIDEO_ID", help="only check that TTCC and TRIBE outputs share a timeline for this video")
    args = ap.parse_args()
    if args.check_alignment:
        print(json.dumps(check_alignment(args.artifacts / "features" / args.check_alignment), indent=2))
        return 0
    if not args.labels:
        ap.error("--labels is required (or use --check-alignment)")
    rows, missing = collect(args.artifacts, args.labels, args.target, args.id_col, args.label_col)
    result = compare(rows, n_boot=args.bootstrap) | {"target": args.target, "missing": {k: len(v) for k, v in missing.items()}}
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
