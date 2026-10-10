"""Per-segment heuristic scores: ``moment_quality`` and ``attention_risk`` (0-100 each).

Two separate questions, deliberately not one number:
- ``moment_quality``: how much measured evidence says this stretch carries content (transcribed
  speech, readable on-screen text, a visual change).
- ``attention_risk``: how much measured evidence says a viewer might lose interest here
  (rule-based flags such as a late first word, quiet or static stretches, long shots, repeated text;
  very low motion; near-silent audio without speech).

A segment can score high on both or low on both. Every point is listed in ``contributions`` with
its source, reason and evidence id. Scores sum points and are capped at 100; 0 means "no evidence
either way", not "bad".

These are transparent editorial heuristics over measured features only (``basis`` is always
``measured_only``). They are not calibrated, not trained on outcomes, and not retention,
engagement or virality predictions. The weights below are hand-set and documented.
"""

from __future__ import annotations

from typing import Any

from .recommendations import STATIC_MOTION_MAX
from .status import AVAILABLE

SCORING_VERSION = "2"  # 2: measured evidence only (model-interpretation inputs removed)
# Points per item.
RISK_WEIGHTS = {
    "rule_recommendation": 15.0,  # measured rule flag (late first word, quiet/static stretch, long shot, ...)
    "measured_static": 5.0,  # visual available, motion below the static threshold
    "measured_quiet_no_speech": 5.0,  # mostly below -50 dBFS with speech measured and no word
}
QUALITY_WEIGHTS = {
    "measured_speech": 5.0,  # transcribed words, unflagged
    "measured_text": 5.0,  # on-screen text observed in an OCR sample inside the segment
    "measured_visual_change": 5.0,  # scene cut at the segment start, or SigLIP within-shot change >= 0.15
}
SCORE_COLUMNS = ["video_id", "segment_index", "start_s", "end_s", "attention_risk", "moment_quality", "basis",
                 "top_risk_reason", "top_quality_reason"]
NOTE = ("Heuristic 0-100 scores from documented weights over measured evidence; "
        "not calibrated, not trained on outcomes, not retention/engagement/virality predictions.")


def _overlaps(item: dict, seg: dict) -> bool:
    return item["start_s"] < seg["end_s"] and item["end_s"] > seg["start_s"]


def _add(contribs: list[dict], source: str, reason: str, points: float, ref: str | None) -> None:
    if points > 0:
        contribs.append({"source": source, "reason": reason, "points": round(points, 2), "ref": ref})


def score_segment(row: dict, recs: list[dict]) -> dict[str, Any]:
    risk: list[dict] = []
    quality: list[dict] = []
    for rec in recs:
        if rec["source"] == "rule" and _overlaps(rec, row):
            _add(risk, "rule", rec["suggestion"], RISK_WEIGHTS["rule_recommendation"], rec["id"])
    seg_ref = f"seg:{row['segment_index']}"
    motion = row.get("motion_mean")
    if row.get("visual_status") == AVAILABLE and motion is not None and motion < STATIC_MOTION_MAX:
        _add(risk, "measured", f"very little motion (motion_mean {motion:.2f})", RISK_WEIGHTS["measured_static"], seg_ref)
    low = row.get("librosa_low_energy_ratio")
    if (row.get("speech_status") == AVAILABLE and row.get("speech_present") is False and low is not None and low >= 0.8):
        _add(risk, "measured", "near-silent audio and no transcribed speech", RISK_WEIGHTS["measured_quiet_no_speech"], seg_ref)
    if row.get("speech_status") == AVAILABLE and (row.get("speech_word_count") or 0) > 0:
        _add(quality, "measured", "clear transcribed speech", QUALITY_WEIGHTS["measured_speech"], seg_ref)
    if row.get("text_status") == AVAILABLE and row.get("text_present") is True:
        _add(quality, "measured", "on-screen text observed", QUALITY_WEIGHTS["measured_text"], seg_ref)
    change = row.get("siglip_within_shot_change")
    if row.get("scene_cut_at_start") is True or (change is not None and change >= 0.15):
        _add(quality, "measured", "visual change (cut or content change)", QUALITY_WEIGHTS["measured_visual_change"], seg_ref)

    def total(c: list[dict]) -> float:
        return round(min(100.0, sum(x["points"] for x in c)), 1)

    def top(c: list[dict]) -> str | None:
        return max(c, key=lambda x: x["points"])["reason"] if c else None

    return {
        "segment_index": row["segment_index"], "start_s": row["start_s"], "end_s": row["end_s"],
        "attention_risk": total(risk), "moment_quality": total(quality),
        "risk_contributions": sorted(risk, key=lambda x: -x["points"]),
        "quality_contributions": sorted(quality, key=lambda x: -x["points"]),
        "top_risk_reason": top(risk), "top_quality_reason": top(quality),
    }


def score_video(video_id: str, rows: list[dict], recs: list[dict], top_n: int = 5) -> dict:
    basis = "measured_only"
    segs = [score_segment(r, recs) | {"basis": basis, "video_id": video_id} for r in rows]

    def ranked(key: str) -> list[dict]:
        """Top spans: adjacent segments sharing the same top reason are one entry (peak score)."""
        reason = "top_risk_reason" if key == "attention_risk" else "top_quality_reason"
        spans: list[dict] = []
        for sg in segs:
            if sg[key] <= 0:
                continue
            last = spans[-1] if spans else None
            if last and last["reason"] == sg[reason] and last["last_index"] == sg["segment_index"] - 1:
                last.update(end_s=sg["end_s"], last_index=sg["segment_index"], peak=max(last["peak"], sg[key]))
                last["segments"].append(sg["segment_index"])
            else:
                spans.append({"start_s": sg["start_s"], "end_s": sg["end_s"], "peak": sg[key], "reason": sg[reason],
                              "segments": [sg["segment_index"]], "last_index": sg["segment_index"]})
        top = sorted(spans, key=lambda sp: (-sp["peak"], sp["start_s"]))[:top_n]
        return [{"start_s": sp["start_s"], "end_s": sp["end_s"], key: sp["peak"], "segment_indices": sp["segments"],
                 "reason": sp["reason"]} for sp in top]

    return {
        "video_id": video_id, "scoring_version": SCORING_VERSION, "basis": basis, "note": NOTE,
        "weights": {"attention_risk": RISK_WEIGHTS, "moment_quality": QUALITY_WEIGHTS,
                    "cap": 100},
        "highest_attention_risk": ranked("attention_risk"),
        "highest_moment_quality": ranked("moment_quality"),
        "segments": segs,
    }
