#!/usr/bin/env python3
"""Print the drop-off analysis of a video's TTCC retention curve, second by second.

    python scripts/show_dropoffs.py FC002 --artifacts artifacts/tribe_test
    python scripts/show_dropoffs.py FC002 --artifacts artifacts/tribe_test --refresh   # recompute from the files on disk

Reads artifacts only (no Space, no model, no upload). The curve is a pretrained model's prediction, not
measured retention; "observed" items were measured in the video; "candidate causes" are unvalidated hypotheses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.dropoffs import DROPOFFS_FILE, write_dropoffs  # noqa: E402

BAR = 30


def obs_short(o: dict) -> str:
    bits = []
    w, t, c = o["words"], o["on_screen_text"], o["scene_cuts"]
    bits.append("words ?" if w == "not run" else (f"words: {w['text'][:40]}" if w["count"] else "no words"))
    if t != "not run" and t:
        bits.append("text: " + ",".join(x["text"] for x in t))
    bits.append("cuts ?" if c == "not run" else f"{c['count']} cut(s)")
    if o["quiet_audio"] not in ("not run", []):
        bits.append("quiet")
    return "; ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    d = args.artifacts / "features" / args.video_id
    if args.refresh or not (d / DROPOFFS_FILE).is_file():
        if write_dropoffs(d) is None:
            print(f"no successful TTCC prediction for {args.video_id} in {d}; run scripts/run_ttcc.py first", file=sys.stderr)
            return 1
    dr = json.loads((d / DROPOFFS_FILE).read_text())
    cv = dr["curve"]
    print(f"\n{dr['video_id']}: predicted retention by TTCC, horizon {dr['source']['horizon_s']} s"
          f"{' (PARTIAL: only the start of the video was predicted)' if dr['source']['partial_coverage'] else ''}")
    print(f"total predicted fall {cv['total_predicted_loss']:.3f}; 50% of it by {cv['50pct_of_predicted_loss_by_s']} s, "
          f"90% by {cv['90pct_of_predicted_loss_by_s']} s")
    print("evidence: " + ", ".join(f"{k}={v}" for k, v in dr["evidence_availability"].items()))
    print("\n== INTERVALS ==")
    for iv in dr["intervals"]:
        share = iv["share_of_total_predicted_loss"]
        print(f"[{iv['id']}] {iv['kind'].replace('_', ' ').upper():18} {iv['start_s']:>2}-{iv['end_s']:<2} s   R {iv['retention_start']:.3f} -> "
              f"{iv['retention_end']:.3f}" + (f"   ({100 * share:.0f}% of the fall)" if share is not None else ""))
        print(f"      observed : {obs_short(iv['observed'])}")
        for c in iv["candidate_causes"][:3]:
            print(f"      candidate (inferred, unvalidated): [{c['rule']}] {c['suggestion']}")
    print("\n== SECOND BY SECOND ==   t   R(t)    drop  rel.loss  class     what was measured in that second")
    for s in dr["seconds"]:
        bar = "█" * int(round(s["retention_end"] * BAR))
        print(f"  {s['t']:>2} s  {s['retention_end']:.3f}  {s['drop']:.3f}  {100 * s['relative_loss']:5.1f}%   {s['kind']:8}  {obs_short(s['observed'])}")
        print(f"        {bar}")
    print(f"\nConfidence: {dr['confidence_note']}.\n{dr['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
