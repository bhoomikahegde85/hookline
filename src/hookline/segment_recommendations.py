"""Structured, segment-level recommendations that combine both models' outputs (``segment_recommendations.json``).

Per segment of the predicted span: TTCC's predicted retention fall (a model prediction), the measured evidence, the rule-based proposals,
and, when Gemini ran, Gemini's observations, hypothesis and timestamped edit suggestions. The two models stay distinguishable:
TTCC predicts, Gemini explains. Nothing is merged into a single score, and every proposal is a hypothesis until validated against
real audience-retention data. Nothing here sends anything anywhere.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .gemini_edit import ARTIFACT as GEMINI_ARTIFACT
from .gemini_edit import NOTE as GEMINI_NOTE

FILE = "segment_recommendations.json"
STATUS = "hypotheses: TTCC's curve is a prediction and Gemini's explanations are interpretations; neither is validated against audience data"


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def build_segment_recommendations(d: Path) -> dict | None:
    """Combine ``ttcc_dropoffs.json`` (+ ``gemini_edit.json`` if usable). None without a drop-off analysis."""
    dr = _load(d / "ttcc_dropoffs.json")
    if not dr:
        return None
    gem = _load(d / GEMINI_ARTIFACT)
    ttcc_key = (_load(d / "ttcc_retention.json") or {}).get("cache_key")
    # Gemini's segment ids (s0, s1, ...) belong to the TTCC curve it was given. After TTCC is rerun the ids are renumbered, so a
    # kept older analysis must never be attached to the new segments.
    stale = bool(gem and gem.get("measured") and gem.get("ttcc_cache_key") != ttcc_key)
    usable = bool(gem and gem.get("measured") and gem.get("analysis")) and not stale
    gem_by_seg = {s["segment_id"]: s for s in gem["analysis"]["segments"]} if usable else {}
    scope = (gem or {}).get("analysis_scope") if usable else None
    skipped = {x["id"]: x["reason"] for x in (scope or {}).get("not_analysed") or []}
    intervals = {iv["id"]: iv for iv in dr["intervals"]}
    segments = []
    for sg in dr["segments"]:
        iv = intervals.get(sg.get("interval_id"))
        proposals = []
        for c in (iv["candidate_causes"] if iv else []):
            proposals.append({"source": "rule", "rule": c["rule"], "suggestion": c["suggestion"], "rationale": None, "expected_effect": None,
                              "confidence": None, "evidence": c.get("evidence"), "evidence_refs": [],
                              "edit": {"action": c["edit"]["action"], "start_s": c["edit"]["start_s"], "end_s": c["edit"]["end_s"]},
                              "status": "hypothesis (rule over measured features), unvalidated"})
        g = gem_by_seg.get(sg["id"])
        for s in (g or {}).get("suggestions", []):
            proposals.append({"source": "gemini", "rule": "gemini:" + s["category"], "suggestion": s["action"], "rationale": s["rationale"],
                              "expected_effect": s["expected_effect"], "confidence": s["confidence"], "evidence": None,
                              "evidence_refs": s["evidence_refs"],
                              "edit": {"action": s["edit"]["type"], "start_s": s["edit"]["start_s"], "end_s": s["edit"]["end_s"]},
                              "status": s["status"]})
        segments.append({
            "id": sg["id"], "interval_id": sg.get("interval_id"), "kind": sg["kind"], "start_s": sg["start_s"], "end_s": sg["end_s"],
            "ttcc": {"retention_start": sg["retention_start"], "retention_end": sg["retention_end"], "predicted_loss": sg["predicted_loss"],
                     "share_of_total_predicted_loss": sg["share_of_total_predicted_loss"], "rank_by_predicted_loss": sg["rank_by_predicted_loss"],
                     "meaning": "TTCC model prediction, not audience data"},
            "heuristic": sg["heuristic"], "observed": sg["observed"],
            "gemini": None if not g else {"summary": g["summary"], "observations": g["observations"], "hypothesis": g["hypothesis"],
                                          "hypothesis_confidence": g["hypothesis_confidence"]},
            "gemini_scope": None if not scope else {"analysed": sg["id"] not in skipped,
                                                    "reason": skipped.get(sg["id"], "inside the clips given to Gemini")},
            "proposals": proposals})
    return {
        "schema": "hookline.segment_recommendations", "video_id": dr["video_id"], "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": STATUS, "source": dr["source"], "evidence_availability": dr["evidence_availability"],
        "gemini": {"used": usable, "stale": stale, "status": (gem or {}).get("status"),
                   "reason": ("an earlier Gemini analysis exists but was made for a different TTCC curve, so it is not attached; rerun the Gemini step"
                              if stale else (gem or {}).get("reason")), "model": (gem or {}).get("model"),
                   "scope": None if not scope else {k: scope.get(k) for k in ("whole_predicted_span", "clips_s", "analysed_s", "max_analysed_s",
                                                                              "predicted_span_s", "not_analysed")},
                   "overall": gem["analysis"]["overall"] if usable else None, "note": GEMINI_NOTE,
                   "validation": (gem or {}).get("validation")},
        "segments": segments}


def write_segment_recommendations(d: Path) -> Path | None:
    out = build_segment_recommendations(d)
    if out is None:
        return None
    path = d / FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path
