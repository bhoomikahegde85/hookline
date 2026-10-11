"""Drop-off detection and diagnosis on TTCC's predicted retention curve (``ttcc_dropoffs.json``).

Input: the predicted curve ``ttcc_retention.json`` (R(t) on a 1 s grid, a pretrained model's prediction, not
measured retention) plus the measured evidence Hookline already has for the same video (Whisper words,
PaddleOCR text events, scene cuts, motion, librosa quiet intervals, rule-based observations).

What it does, with fixed thresholds written below (chosen for readability, not validated):
1. Classifies every predicted second by the share of the *remaining* viewers lost: ``steep`` (a large fall),
   ``decline`` (a steady fall), ``hold`` (almost flat) or unlabelled.
2. Merges runs into intervals: ``steep_drop``, ``sustained_decline`` (>= 3 s) and ``holds`` (>= 3 s), each with
   its share of the curve's total predicted loss.
3. Lists what was **observed** inside each interval (words, text, cuts, quiet, motion). A source that was not
   run is reported as "not run", never as "nothing happened".
4. Lists **inferred** candidate causes, only from existing rule-based observations that overlap the interval.
   The model does not say why it predicts a fall, and nothing here has been checked against real viewer
   behaviour, so these are labelled hypotheses.

There is no confidence value: TTCC outputs a point curve with no calibrated uncertainty on these videos.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DROPOFFS_FILE = "ttcc_dropoffs.json"
DROPOFFS_VERSION = "2"  # 2: even, duration-based segments (was: runs of similar seconds, which merged everything after the opening)
SEGMENT_TARGET_S = 4
MAX_SEGMENTS = 12
SEGMENT_MIN_LOSS = 0.01  # a segment that loses less than this much of the audience is not flagged as a decline
THRESHOLDS = {
    "steep_relative_loss": 0.15,   # >= 15 % of the viewers still watching are lost within one second ...
    "steep_min_drop": 0.02,        # ... and the fall is at least 0.02 in absolute terms (ignores tiny tails)
    "decline_relative_loss": 0.05,  # a steady fall: >= 5 % of the remaining viewers per second
    "decline_min_seconds": 3,
    "hold_relative_loss": 0.02,    # nearly flat: < 2 % per second
    "hold_min_seconds": 3,
    "no_speech_min_seconds": 3,   # interval at least this long with speech measured and no word
    "fast_cut_rate": 1.5,         # scene cuts per second at or above this count as very fast cutting
}
NOTE = ("Derived from a pretrained model's predicted curve, not measured viewer behaviour, and not validated on these "
        "videos. 'Observed' items were measured in the video; 'candidate causes' are unvalidated rule-based hypotheses.")
# What kind of edit each rule proposes. "trim": remove the flagged range (the creator adjusts it before accepting);
# "add": something should be added at the start of the range (no footage is removed); "review": look at it, no specific edit.
EDIT_ACTIONS = {
    "late_first_word": "trim", "quiet_stretch": "trim", "static_stretch": "trim",
    "opening_without_words": "add", "no_early_visual_change": "add", "interval_no_speech": "add",
    "long_shot": "review", "repeated_text": "review", "interval_fast_cutting": "review",
}
CONFIDENCE_NOTE = "none: TTCC outputs a point curve with no calibrated uncertainty on these videos"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def classify_seconds(retention: list[float]) -> list[dict]:
    """Per-second loss: second t covers (t-1, t]."""
    out = []
    for t in range(1, len(retention)):
        prev, cur = retention[t - 1], retention[t]
        drop = prev - cur
        rel = drop / prev if prev > 0 else 0.0
        if rel >= THRESHOLDS["steep_relative_loss"] and drop >= THRESHOLDS["steep_min_drop"]:
            kind = "steep"
        elif rel >= THRESHOLDS["decline_relative_loss"]:
            kind = "decline"
        elif rel < THRESHOLDS["hold_relative_loss"]:
            kind = "hold"
        else:
            kind = "gradual"
        out.append({"t": t, "start_s": t - 1, "end_s": t, "retention_start": prev, "retention_end": cur,
                    "drop": drop, "relative_loss": rel, "kind": kind})
    return out


def merge_runs(seconds: list[dict], total_loss: float) -> list[dict]:
    """Intervals from runs of the same second-kind (steep always; decline/hold only if long enough)."""
    intervals, run = [], []

    def close() -> None:
        if not run:
            return
        kind, n = run[0]["kind"], len(run)
        label = {"steep": "steep_drop", "decline": "sustained_decline", "hold": "holds"}.get(kind)
        minimum = {"decline": THRESHOLDS["decline_min_seconds"], "hold": THRESHOLDS["hold_min_seconds"]}.get(kind, 1)
        if label and n >= minimum:
            start, end = run[0]["start_s"], run[-1]["end_s"]
            loss = run[0]["retention_start"] - run[-1]["retention_end"]
            intervals.append({"kind": label, "start_s": start, "end_s": end, "retention_start": run[0]["retention_start"],
                              "retention_end": run[-1]["retention_end"], "loss": loss,
                              "share_of_total_predicted_loss": loss / total_loss if total_loss > 0 else None})

    for sec in seconds:
        if run and sec["kind"] != run[0]["kind"]:
            close()
            run = []
        run.append(sec)
    close()
    return intervals


def even_segments(retention: list[float], total_loss: float, target_s: int = SEGMENT_TARGET_S, max_segments: int = MAX_SEGMENTS) -> list[dict]:
    """Consecutive segments of (nearly) equal length over the whole predicted span, chosen from the duration alone: about ``target_s`` seconds
    each (0-4, 4-8, ...), at most ``max_segments`` (a 120 s video gets 12 segments of 10 s). Boundaries are whole seconds, so lengths differ by at
    most one second. Each segment is typed from its own predicted fall: steep_drop (it contains a steep second), sustained_decline (a steady fall
    that is large enough to matter), holds (nearly flat) or unflagged (a small ordinary fall)."""
    horizon = len(retention) - 1
    if horizon < 1:
        return []
    n = min(max_segments, max(1, round(horizon / target_s)))
    if horizon >= 2 * target_s:
        n = max(n, 2)
    bounds = [round(i * horizon / n) for i in range(n + 1)]
    seconds = classify_seconds(retention)
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        inside = seconds[a:b]
        loss = retention[a] - retention[b]
        rel = (1 - (retention[b] / retention[a]) ** (1 / (b - a))) if retention[a] > 0 and retention[b] > 0 else 1.0  # mean per-second relative loss
        if any(x["kind"] == "steep" for x in inside):
            kind = "steep_drop"
        elif rel >= THRESHOLDS["decline_relative_loss"] and loss >= SEGMENT_MIN_LOSS:
            kind = "sustained_decline"
        elif rel < THRESHOLDS["hold_relative_loss"]:
            kind = "holds"
        else:
            kind = "unflagged"
        out.append({"kind": kind, "start_s": a, "end_s": b, "retention_start": retention[a], "retention_end": retention[b], "loss": loss,
                    "share_of_total_predicted_loss": loss / total_loss if total_loss > 0 else None})
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def edit_proposal(rule: str, start: float, end: float) -> dict:
    """A structured, editable proposal for a rule: an action and the time range it applies to."""
    action = EDIT_ACTIONS.get(rule, "review")
    return {"action": action, "start_s": start, "end_s": end,
            **({"note": "default range is the whole flagged stretch; adjust it before accepting"} if action == "trim" else {})}


class Evidence:
    """Measured evidence for one video, each source flagged available or not run."""

    def __init__(self, d: Path):
        words_path = d / "words.csv"
        speech, ocr, librosa, report = (_load(d / n) for n in ("speech.json", "ocr.json", "librosa.json", "report.json"))
        self.speech_language = (speech or {}).get("language")
        self.words: list[dict] = []
        if speech and speech.get("measured") and words_path.is_file():
            with words_path.open(newline="") as f:
                self.words = [{"start_s": float(r["start_s"]), "end_s": float(r["end_s"]), "word": r["word"].strip(),
                               "word_index": int(r["word_index"])} for r in csv.DictReader(f)]
        self.text_events = [{"event_id": e["event_id"], "text": e["text"], "start_s": e["first_seen_s"], "end_s": e["visible_until_s"]}
                            for e in (ocr or {}).get("events") or []] if ocr and ocr.get("measured") else []
        self.quiet = [{"interval_id": i["interval_id"], "start_s": i["start_s"], "end_s": i["end_s"]}
                      for i in (librosa or {}).get("low_energy_intervals") or []] if librosa and librosa.get("measured") else []
        self.cuts = list((report or {}).get("scene_cut_times") or []) if report else []
        self.segments = [s for s in (report or {}).get("segments") or [] if s.get("visual_status") == "available"
                         and s.get("motion_mean") is not None]
        self.recs = (_load(d / "recommendations.json") or {}).get("recommendations") or []
        self.availability = {
            "speech": "available" if speech and speech.get("measured") else "not run",
            "text": "available" if ocr and ocr.get("measured") else "not run",
            "audio_quiet": "available" if librosa and librosa.get("measured") else "not run",
            "scene_cuts_and_motion": "available" if report and self.segments else "not run",
            "rule_observations": "available" if self.recs or (d / "recommendations.json").is_file() else "not run",
        }

    def observed(self, start: float, end: float) -> dict:
        """What was measured in [start, end)."""
        av = self.availability
        out: dict[str, Any] = {}
        words = [w for w in self.words if w["start_s"] < end and w["end_s"] > start]
        out["words"] = ({"count": len(words), "text": " ".join(w["word"] for w in words[:14]) + (" ..." if len(words) > 14 else ""),
                         "first_s": words[0]["start_s"] if words else None} if av["speech"] == "available" else "not run")
        texts = [e for e in self.text_events if _overlap(start, end, e["start_s"], e["end_s"])]
        out["on_screen_text"] = ([{"event_id": e["event_id"], "text": e["text"]} for e in texts] if av["text"] == "available" else "not run")
        cuts = [c for c in self.cuts if start <= c < end]
        out["scene_cuts"] = {"count": len(cuts), "times_s": cuts[:12]} if av["scene_cuts_and_motion"] == "available" else "not run"
        segs = [s for s in self.segments if _overlap(start, end, s["start_s"], s["end_s"])]
        out["motion_mean"] = round(sum(s["motion_mean"] for s in segs) / len(segs), 2) if segs else None
        quiet = [q for q in self.quiet if _overlap(start, end, q["start_s"], q["end_s"])]
        out["quiet_audio"] = [{"interval_id": q["interval_id"], "start_s": q["start_s"], "end_s": q["end_s"]} for q in quiet] \
            if av["audio_quiet"] == "available" else "not run"
        return out

    def candidates(self, start: float, end: float, observed: dict | None = None) -> list[dict]:
        """Rule-based hypotheses: existing recommendations that overlap the interval, plus two interval rules over what was observed."""
        out = [{"rule": r["rule"], "start_s": r["start_s"], "end_s": r["end_s"], "suggestion": r["suggestion"], "evidence": r.get("evidence"),
                "status": "inferred, unvalidated", "edit": edit_proposal(r["rule"], r["start_s"], r["end_s"])}
               for r in self.recs if _overlap(start, end, r["start_s"], r["end_s"])]
        length = end - start
        o = observed or {}
        w, c = o.get("words"), o.get("scene_cuts")
        if isinstance(w, dict) and w["count"] == 0 and length >= THRESHOLDS["no_speech_min_seconds"]:
            out.append({"rule": "interval_no_speech", "start_s": start, "end_s": end, "status": "inferred, unvalidated",
                        "edit": edit_proposal("interval_no_speech", start, end),
                        "suggestion": f"No speech was transcribed for these {length:g} s: consider whether a voice line or caption should carry the message here.",
                        "evidence": [{"type": "words", "observed": "0 words"}]})
        if isinstance(c, dict) and length > 0 and c["count"] / length >= THRESHOLDS["fast_cut_rate"]:
            out.append({"rule": "interval_fast_cutting", "start_s": start, "end_s": end, "status": "inferred, unvalidated",
                        "edit": edit_proposal("interval_fast_cutting", start, end),
                        "suggestion": f"{c['count']} scene cuts in {length:g} s: check that each shot is readable and that the pace is intended.",
                        "evidence": [{"type": "scene_cuts", "observed": f"{c['count']} cuts"}]})
        return out


def _agg(values: list[float]) -> dict:
    return {"mean": round(sum(values) / len(values), 1), "max": round(max(values), 1)} if values else {"mean": None, "max": None}


def score_segments(retention: list[float], intervals: list[dict], ev: "Evidence", heuristic: list[dict] | None) -> list[dict]:
    """Partition [0, horizon] into consecutive segments (the flagged intervals plus the stretches between them) and score each.

    Two kinds of number are reported side by side and never merged: the predicted retention fall within the segment (TTCC), and
    the existing heuristic ``attention_risk`` / ``moment_quality`` (0-100) from measured features, aggregated over the 0.5 s
    segments that overlap it. ``rank_by_predicted_loss`` is 1 for the segment with the largest predicted fall."""
    horizon, total = len(retention) - 1, retention[0] - retention[-1]
    spans: list[tuple[int, int, str, str | None]] = []
    cursor = 0
    for iv in sorted(intervals, key=lambda i: i["start_s"]):
        if iv["start_s"] > cursor:
            spans.append((cursor, iv["start_s"], "unflagged", None))
        spans.append((iv["start_s"], iv["end_s"], iv["kind"], iv["id"]))
        cursor = iv["end_s"]
    if cursor < horizon:
        spans.append((cursor, horizon, "unflagged", None))
    segments = []
    for n, (a, b, kind, iv_id) in enumerate(spans):
        loss = retention[a] - retention[b]
        over = [h for h in (heuristic or []) if _overlap(a, b, h["start_s"], h["end_s"])]
        segments.append({
            "id": f"s{n}", "interval_id": iv_id, "kind": kind, "start_s": a, "end_s": b, "length_s": b - a,
            "retention_start": retention[a], "retention_end": retention[b], "predicted_loss": loss,
            "predicted_loss_per_s": loss / (b - a), "share_of_total_predicted_loss": loss / total if total > 0 else None,
            "heuristic": {"available": heuristic is not None, "attention_risk": _agg([h["attention_risk"] for h in over]),
                          "moment_quality": _agg([h["moment_quality"] for h in over])} if heuristic is not None
            else {"available": False, "attention_risk": _agg([]), "moment_quality": _agg([])},
            "observed": ev.observed(a, b)})
    for rank, seg in enumerate(sorted(segments, key=lambda s: -s["predicted_loss"]), 1):
        seg["rank_by_predicted_loss"] = rank
    return segments


def analyze(d: Path) -> dict | None:
    """Drop-off analysis for the video folder ``d``; None without a successful TTCC prediction."""
    art = _load(d / "ttcc_retention.json")
    if not art or art.get("status") != "success":
        return None
    retention, ts = art["retention"], art["timestamps_s"]
    if len(retention) < 3 or ts != list(range(len(retention))):
        return None
    ev = Evidence(d)
    scores = _load(d / "scores.json")
    heuristic = [{"start_s": s["start_s"], "end_s": s["end_s"], "attention_risk": s["attention_risk"], "moment_quality": s["moment_quality"]}
                 for s in (scores or {}).get("segments") or []] or None
    seconds = classify_seconds(retention)
    total = retention[0] - retention[-1]
    intervals = even_segments(retention, total)
    for n, iv in enumerate(intervals):
        iv["id"] = f"d{n}"
        iv["observed"] = ev.observed(iv["start_s"], iv["end_s"])
        iv["candidate_causes"] = ev.candidates(iv["start_s"], iv["end_s"], iv["observed"]) if iv["kind"] != "holds" else []
    table = []
    for s in seconds:
        obs = ev.observed(s["start_s"], s["end_s"])
        table.append({k: s[k] for k in ("t", "retention_end", "drop", "relative_loss", "kind")} | {"observed": obs})
    reach = {f"{int(p * 100)}pct_of_predicted_loss_by_s":  # first second by which this share of the curve's total fall has happened
             next((t for t, r in enumerate(retention) if total > 0 and retention[0] - r >= p * total), None) for p in (0.5, 0.9)}
    return {
        "schema": "hookline.ttcc_dropoffs", "version": DROPOFFS_VERSION, "video_id": art["video_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"file": "ttcc_retention.json", "cache_key": art.get("cache_key"), "horizon_s": art["horizon_s"],
                   "partial_coverage": art.get("partial_coverage"), "model": (art.get("model") or {}).get("checkpoint_repo")},
        "thresholds": THRESHOLDS, "evidence_availability": ev.availability | {"heuristic_scores": "available" if heuristic else "not run"}, "speech_language": ev.speech_language,
        "curve": {"retention_at_horizon": retention[-1], "total_predicted_loss": total, **reach},
        "intervals": intervals, "segments": score_segments(retention, intervals, ev, heuristic), "seconds": table, "confidence": None, "confidence_note": CONFIDENCE_NOTE, "note": NOTE,
    }


def write_dropoffs(d: Path) -> Path | None:
    result = analyze(d)
    if result is None:
        return None
    path = d / DROPOFFS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path
