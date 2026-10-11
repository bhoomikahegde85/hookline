#!/usr/bin/env python3
"""Why is the hook strong or weak, and which moments are the strongest and weakest? Writes hook_report.json and moments.json for one analysed video.

    python scripts/explain_video.py FC002 --artifacts artifacts/e2e

Reads saved files only (no model, no Space, no upload). The hook report compares the video's own measured opening and TTCC's prediction with the 100 reference
reels; the moments are ranked by TTCC's predicted hazard within the video. Everything is a model prediction or an unvalidated rule of thumb, not audience data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.hook_report import build_hook_report, write_hook_report  # noqa: E402
from hookline.moments import find_moments, write_moments  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "e2e")
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline")
    ap.add_argument("--reference-features", type=Path, default=PROJECT_ROOT / "artifacts" / "company" / "features")
    ap.add_argument("--ttcc-reference", type=Path, default=PROJECT_ROOT / "artifacts" / "hook")
    args = ap.parse_args(argv)
    d = args.artifacts / "features" / args.video_id
    if not d.is_dir():
        print(f"no artifacts for {args.video_id} in {d}", file=sys.stderr)
        return 1
    ttcc_ref = args.ttcc_reference / "features" if (args.ttcc_reference / "features").is_dir() else None
    rep = build_hook_report(d, args.data, args.reference_features, ttcc_ref)
    print(f"hook report: {rep.get('status')} verdict={ (rep.get('verdict') or {}).get('label')} ({len(rep.get('factors') or [])} factors)" + (f" - {rep.get('reason')}" if rep.get("reason") else ""))
    write_hook_report(d, rep)
    mo = find_moments(d)
    if mo is None:
        print("moments: no TTCC prediction for this video")
    else:
        write_moments(d, mo)
        print(f"moments: {len(mo['strongest'])} strongest, {len(mo['weakest'])} weakest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
