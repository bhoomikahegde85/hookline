#!/usr/bin/env python3
"""Print a readable view of one video's feature report (plus speech/OCR if run_speech_ocr.py ran,
and TTCC predicted retention if run_ttcc.py ran).

    python scripts/show_features.py syn_av_testpattern_8s
    python scripts/show_features.py syn_av_testpattern_8s --segments 20
    python scripts/show_features.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    ("start_s", 6), ("end_s", 6), ("scene_index", 5), ("visual_status", 10), ("brightness_mean", 7),
    ("motion_mean", 6), ("scene_change_max", 6), ("audio_status", 10), ("rms_dbfs", 7),
    ("near_silence_ratio", 5), ("onset_count", 4),
]


def fmt(v, width: int) -> str:
    if v is None:
        s = "-"
    elif isinstance(v, bool):
        s = "y" if v else "n"
    elif isinstance(v, float):
        s = f"{v:.2f}"
    else:
        s = str(v)
    return s[:width].rjust(width)


def show_speech_text(out_dir: Path, limit: int = 8) -> None:
    """Speech and on-screen text from run_speech_ocr.py, if present."""
    for name, items_key in (("speech", "segments"), ("ocr", "events")):
        path = out_dir / f"{name}.json"
        if not path.exists():
            continue
        a = json.loads(path.read_text())
        detected = a.get("speech_detected" if name == "speech" else "text_detected")
        extra = f" language={a.get('language')}" if name == "speech" and a.get("language") else ""
        print(f"\n{name}: status={a['status']} detected={detected}{extra}" + (f"  ({a['reason']})" if a.get("reason") else ""))
        items = a.get(items_key) or []
        for it in items[:limit]:
            if name == "speech":
                print(f"  {it['start_s']:6.2f}-{it['end_s']:6.2f}  {it['text']}" + (f"  [{', '.join(it['flags'])}]" if it["flags"] else ""))
            else:
                rep = f"  (repeat of {it['repeat_of']})" if it["repeat_of"] is not None else ""
                print(f"  {it['first_seen_s']:6.2f}-{it['visible_until_s']:6.2f}  {it['text']}{rep}")
        if len(items) > limit:
            print(f"  ... {len(items) - limit} more")


def show_ttcc(out_dir: Path) -> None:
    """TTCC status and predicted retention from run_ttcc.py, if present (a prediction, not a measurement)."""
    st_path, art_path = out_dir / "ttcc_status.json", out_dir / "ttcc_retention.json"
    if not st_path.exists() and not art_path.exists():
        return
    st = json.loads(st_path.read_text()) if st_path.exists() else {}
    print(f"\nttcc (predicted retention): latest attempt {st.get('status')}" + (f"  ({st['reason']})" if st.get("reason") else ""))
    art = json.loads(art_path.read_text()) if art_path.exists() else None
    if art and art.get("status") == "success":
        current = "" if st.get("prediction_is_current") else "  [not confirmed current]"
        pairs = "  ".join(f"{t}s:{r:.3f}" for t, r in zip(art["timestamps_s"], art["retention"]))
        print(f"  R(t), fraction of viewers, 1 s steps{current}: {pairs}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id", nargs="?")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    ap.add_argument("--segments", type=int, default=12, help="segment rows to print (0 = all)")
    ap.add_argument("--list", action="store_true", help="list processed videos from run_summary.json")
    args = ap.parse_args(argv)
    features = args.artifacts / "features"

    if args.list or not args.video_id:
        summary = json.loads((features / "run_summary.json").read_text())
        for v in summary["videos"]:
            print(f"{v['processing_status']:8} {v['video_id']:40} visual={v['visual_stream']} audio={v['audio_stream']} segments={v['segments']}")
        return 0

    path = features / args.video_id / "report.json"
    if not path.exists():
        print(f"no report at {path}; run scripts/run_pipeline.py first", file=sys.stderr)
        return 1
    r = json.loads(path.read_text())
    p = r.get("probe") or {}
    print(f"{r['video_id']}  status={r['processing_status']}  duration={r.get('duration_s')}s")
    print(f"source: {r['source_path']}")
    print(f"probe: {p.get('width')}x{p.get('height')} @ {p.get('fps')} fps, video={p.get('has_video')}, audio={p.get('has_audio')}, codec={p.get('video_codec')}/{p.get('audio_codec')}")
    for m, info in (r.get("modalities") or {}).items():
        if isinstance(info, dict):
            print(f"  {m}: {info.get('stream_status')}")
    for label in ("warnings", "errors"):
        for w in r.get(label) or []:
            print(f"  {label[:-1]}: {w}")
    o = r.get("opening") or {}
    print(f"\nopening ({o.get('analysed_seconds')} s): valid windows {o.get('valid_windows')}, coverage {o.get('coverage')}")
    if o.get("visual"):
        print(f"  visual: {o['visual']}")
    if o.get("audio"):
        print(f"  audio: {o['audio']}")
    print(f"scene cuts: {r.get('scene_cut_times')}")
    show_speech_text(features / args.video_id)
    show_ttcc(features / args.video_id)

    segs = r.get("segments") or []
    n = len(segs) if args.segments == 0 else args.segments
    print(f"\nsegments ({len(segs)} total, showing {min(n, len(segs))}):")
    print(" ".join(name[:w].rjust(w) for name, w in COLUMNS))
    for s in segs[:n]:
        print(" ".join(fmt(s.get(name), w) for name, w in COLUMNS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
