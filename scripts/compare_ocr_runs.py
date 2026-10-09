#!/usr/bin/env python3
"""Count on-screen-text episodes that a sparser OCR run missed, against a denser reference run.

    python scripts/compare_ocr_runs.py --ref artifacts/company_ocr_ref --test artifacts/company FC003 FL004 FL019

Both roots must hold features/<video_id>/ocr.json for the video. Only the time range
sampled by both runs is compared.

Text. Matching uses normalised text (NFKC, case-folded, whitespace including line breaks
collapsed, punctuation/symbols ignored). Two texts are the same caption when their
similarity is >= 0.8 (--similarity), or when one contains the other as whole words and the
shorter has >= 4 characters, so a caption read as one line in one run and split over two
lines/boxes in the other still matches ("SALE" does not match inside "WHOLESALE").

Episodes. The reference run's events are first grouped into visibility episodes: one
stretch of time during which one piece of text stays on screen. Two reference events join
the same episode when their intervals overlap or touch (no reference sample between them
without either) and either they are the same caption (line split / merged lines), or they
sit in the same place (box IoU >= 0.3) and read alike (similarity >= 0.6, --link-similarity,
or one is a >= 4-character prefix of the other): OCR misreads and partial reads of one
text. An episode with reads in >= 2 reference samples is "persistent".

Timing. A test sample at t stands for [t, t + interval_s); a reference event stands for
[first_seen_s, visible_until_s), and an episode for the union of its events. Intervals are
half-open: a sample matches only if it overlaps the episode by more than 1e-6 s, so
intervals that merely touch do not match.

Outcomes, per episode. "found": a test read in an overlapping sample is the same caption
as any of the episode's reads. "missed": no such read. "merged": every candidate test
event was already credited to an earlier, separate episode of the same caption, i.e. the
text left the screen (in the reference) and came back, but the sparse run linked both
appearances into one event. Each test event is credited to at most one episode per
caption; different captions (and different fragments of one caption) may share it.
Episodes are matched in time order, persistent ones first.

Samples taken at the same timestamp in both runs show the same frame, so "identical reads"
measures OCR instability rather than sampling. Text that neither run recognised is not
counted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.ocr import box_iou, normalise_text  # noqa: E402

EPS = 1e-6  # seconds; float tolerance on interval ends
SIMILARITY = 0.8  # same caption
LINK_SIMILARITY = 0.6  # misread / partial read of one text in the same place
LINK_IOU = 0.3


def load(root: Path, video_id: str) -> dict | None:
    path = root / "features" / video_id / "ocr.json"
    try:
        art = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return art if art.get("measured") else None


def match_key(text: str) -> str:
    """Normalised text without punctuation/symbols (letters, digits and combining marks are kept)."""
    kept = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in normalise_text(text))
    return re.sub(r"\s+", " ", kept).strip()


def frame_texts(art: dict, frame: dict) -> set[str]:
    by_id = {e["event_id"]: e["text"] for e in art["events"]}
    return {match_key(d.get("text") or by_id[d["event_id"]]) for d in frame["detections"]}


def ratio_at_least(a: str, b: str, similarity: float) -> bool:
    return bool(a) and bool(b) and SequenceMatcher(None, a, b).ratio() >= similarity


def similar(a: str, b: str, similarity: float = SIMILARITY) -> bool:
    """Same caption: similar text, or one contains the other as whole words (line split/merge)."""
    if not a or not b:
        return False
    if min(len(a), len(b)) >= 4 and (f" {a} " in f" {b} " or f" {b} " in f" {a} "):
        return True
    return ratio_at_least(a, b, similarity)


def misread(a: str, b: str, box_a: list[float], box_b: list[float], link_similarity: float = LINK_SIMILARITY) -> bool:
    """Likely the same text read differently: same place, and alike or one a prefix of the other."""
    if box_iou(box_a, box_b) < LINK_IOU:
        return False
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    return ratio_at_least(a, b, link_similarity)


def overlaps(t: float, interval: float, span: dict) -> bool:
    return t < span["visible_until_s"] - EPS and t + interval > span["first_seen_s"] + EPS


def touch(a: dict, b: dict) -> bool:
    return a["first_seen_s"] <= b["visible_until_s"] + EPS and b["first_seen_s"] <= a["visible_until_s"] + EPS


def episodes(art: dict, end: float, similarity: float = SIMILARITY, link_similarity: float = LINK_SIMILARITY) -> list[dict]:
    """Group a run's events (first seen before ``end``) into visibility episodes, in time order."""
    events = [e for e in art["events"] if e["first_seen_s"] < end]
    keys = {e["event_id"]: match_key(e["text"]) for e in events}
    seen: dict[int, set[float]] = {e["event_id"]: set() for e in events}
    for f in art["frames"]:
        for d in f["detections"]:
            if d.get("event_id") in seen:
                seen[d["event_id"]].add(f["t_s"])
    parent = {e["event_id"]: e["event_id"] for e in events}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(events):
        for b in events[i + 1:]:
            ka, kb = keys[a["event_id"]], keys[b["event_id"]]
            if touch(a, b) and (similar(ka, kb, similarity) or misread(ka, kb, a["box_norm"], b["box_norm"], link_similarity)):
                parent[find(a["event_id"])] = find(b["event_id"])
    groups: dict[int, list[dict]] = {}
    for e in events:
        groups.setdefault(find(e["event_id"]), []).append(e)
    out = []
    for members in groups.values():
        main_event = max(members, key=lambda e: (e["frames"], -e["first_seen_s"]))
        out.append({
            "text": main_event["text"],
            "texts": sorted({e["text"] for e in members}),
            "keys": {keys[e["event_id"]] for e in members},
            "first_seen_s": min(e["first_seen_s"] for e in members),
            "visible_until_s": max(e["visible_until_s"] for e in members),
            "samples": len(set().union(*(seen[e["event_id"]] for e in members))),
            "events": len(members),
        })
    return sorted(out, key=lambda ep: (ep["first_seen_s"], ep["text"]))


def compare(ref: dict, test: dict, similarity: float = SIMILARITY, link_similarity: float = LINK_SIMILARITY) -> dict:
    end = min(ref["sampling"]["range_s"][1], test["sampling"]["range_s"][1])
    interval = test["sampling"]["interval_s"]
    ref_eps = episodes(ref, end, similarity, link_similarity)
    test_frames = [f for f in test["frames"] if f["status"] == "available" and f["t_s"] < end]
    test_text = {e["event_id"]: e["text"] for e in test["events"]}
    # (sample time, test event id, match key) for every sparse detection
    reads = [(f["t_s"], d["event_id"], match_key(d.get("text") or test_text[d["event_id"]]))
             for f in test_frames for d in f["detections"]]
    credited: dict[int, list[dict]] = {}  # test event id -> reference episodes credited to it

    def same_caption(a: dict, b: dict) -> bool:
        return any(similar(x, y, similarity) for x in a["keys"] for y in b["keys"])

    def match(ep: dict) -> str:
        candidates = sorted({(t, eid) for t, eid, x in reads
                             if overlaps(t, interval, ep) and any(similar(k, x, similarity) for k in ep["keys"])})
        if not candidates:
            return "missed"
        for _, eid in candidates:
            if not any(same_caption(ep, other) for other in credited.get(eid, [])):
                credited.setdefault(eid, []).append(ep)
                return "found"
        return "merged"

    persistent = [ep for ep in ref_eps if ep["samples"] >= 2]
    single = [ep for ep in ref_eps if ep["samples"] < 2]
    for ep in persistent + single:
        ep["outcome"] = match(ep)

    def count(eps: list[dict], kind: str) -> int:
        return sum(1 for ep in eps if ep["outcome"] == kind)

    def listed(kind: str) -> list[dict]:
        return [{"text": ep["text"], "first_seen_s": ep["first_seen_s"], "visible_s": round(ep["visible_until_s"] - ep["first_seen_s"], 3),
                 "samples": ep["samples"], "events": ep["events"], "texts": ep["texts"]} for ep in persistent if ep["outcome"] == kind]

    ref_events = sum(1 for e in ref["events"] if e["first_seen_s"] < end)
    ref_at = {round(f["t_s"], 3): f for f in ref["frames"] if f["status"] == "available"}
    shared = [(f, ref_at[round(f["t_s"], 3)]) for f in test_frames if round(f["t_s"], 3) in ref_at]
    return {
        "range_s": [0.0, end],
        "ref_fps": ref["sampling"]["fps"],
        "test_fps": test["sampling"]["fps"],
        "ref_events": ref_events,
        "test_events": sum(1 for e in test["events"] if e["first_seen_s"] < end),
        "ref_episodes": len(ref_eps),
        "test_episodes": len(episodes(test, end, similarity, link_similarity)),
        "ref_fragment_events": ref_events - len(ref_eps),  # extra events from line splits / misreads
        "ref_persistent": len(persistent),
        "missed_persistent": count(persistent, "missed"),
        "merged_persistent": count(persistent, "merged"),
        "ref_single_sample": len(single),
        "missed_single_sample": count(single, "missed"),
        "merged_single_sample": count(single, "merged"),
        "shared_samples": len(shared),
        "identical_reads": sum(1 for a, b in shared if frame_texts(test, a) == frame_texts(ref, b)),
        "missed": listed("missed"),
        "merged": listed("merged"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--ref", type=Path, required=True, help="artifacts root of the denser (reference) OCR run")
    ap.add_argument("--test", type=Path, required=True, help="artifacts root of the run being evaluated")
    ap.add_argument("--json", type=Path, help="also write the comparison to this file")
    ap.add_argument("--similarity", type=float, default=SIMILARITY, help=f"same-caption text similarity (default {SIMILARITY})")
    ap.add_argument("--link-similarity", type=float, default=LINK_SIMILARITY,
                    help=f"similarity that links misreads in the same place into one episode (default {LINK_SIMILARITY})")
    args = ap.parse_args(argv)

    results, code = {}, 0
    for vid in args.video_ids:
        ref, test = load(args.ref, vid), load(args.test, vid)
        if ref is None or test is None:
            print(f"{vid}: missing or unmeasured ocr.json ({'ref' if ref is None else 'test'})")
            code = 1
            continue
        r = results[vid] = compare(ref, test, args.similarity, args.link_similarity)
        print(f"{vid} [0, {r['range_s'][1]:g}] s, {r['test_fps']:g} fps vs {r['ref_fps']:g} fps reference: "
              f"missed {r['missed_persistent']}/{r['ref_persistent']} persistent episodes "
              f"(+{r['merged_persistent']} reappearances merged into one sparse event), "
              f"{r['missed_single_sample']}/{r['ref_single_sample']} single-sample episodes "
              f"(episodes: test {r['test_episodes']}, ref {r['ref_episodes']} from {r['ref_events']} events; "
              f"identical reads on {r['identical_reads']}/{r['shared_samples']} shared samples)")
        for kind in ("missed", "merged"):
            for m in r[kind]:
                print(f"    {kind} {m['first_seen_s']:6.2f}s for {m['visible_s']:.2f}s ({m['samples']} ref samples, "
                      f"{m['events']} events): {' / '.join(m['texts'])[:80]}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
