"""Evidence-grounded editing suggestions for one video.

Two sources, never mixed:
- ``rule``: deterministic checks over measured evidence (Task 2 visual features, librosa,
  Whisper words, PaddleOCR samples). A rule only fires when the evidence it needs was
  measured: missing, failed or uncertain modalities never trigger a suggestion.
- ``gemini``: the model's own edit suggestions (``gemini.json``), passed through with their
  supporting moment ids and labelled as model interpretation.

Every recommendation names its interval, the evidence it rests on (ids that exist in the
per-video artifacts) and why the edit might help. Thresholds are heuristics chosen for
reviewability, not values validated against retention data; nothing here predicts
retention, engagement or virality.
"""

from __future__ import annotations

from typing import Any

from .status import AVAILABLE

LATE_FIRST_WORD_S = 2.0  # first spoken word at or after this
OPENING_S = 3.0  # "opening" for the no-words check
QUIET_MIN_S = 1.5  # low-energy interval at least this long (librosa, below silence_dbfs)
STATIC_MIN_S = 3.0  # same scene, low motion for at least this long
STATIC_MOTION_MAX = 1.0  # motion_mean (0-255 luma difference); ~ lowest 7 % of company-reel segments
CAVEAT = "Heuristic threshold, not validated against retention data."
EVIDENCE_TYPES = ("segment", "word", "text_event", "librosa_interval", "gemini_moment", "gemini_suggestion")


def _segs_in(rows: list[dict], start: float, end: float) -> list[dict]:
    return [r for r in rows if r["start_s"] < end and r["end_s"] > start]


def _ev(kind: str, ids: list) -> list[dict]:
    return [{"type": kind, "id": i} for i in ids]


def _opening_without_words(rows: list[dict], words: list[dict], ocr: dict | None, duration: float) -> dict | None:
    end = min(OPENING_S, duration)
    opening = _segs_in(rows, 0.0, end)
    if not opening or any(r.get("speech_status") != AVAILABLE for r in opening):
        return None  # speech not measured across the opening: no claim
    if any(w["start_s"] < end for w in words):
        return None
    if not ocr or not ocr.get("measured"):
        return None
    samples = [f for f in ocr["frames"] if f["t_s"] < end]
    if not samples or any(f["status"] != AVAILABLE or f["detections"] for f in samples):
        return None  # text seen, or not every opening sample succeeded
    return {
        "rule": "opening_without_words",
        "start_s": 0.0, "end_s": round(end, 3),
        "suggestion": "Consider stating the premise in the first seconds with a short caption or spoken line.",
        "why": (f"No speech was transcribed and no on-screen text was read in the first {end:g} s "
                f"(OCR every {ocr['sampling']['interval_s']:g} s). Words early can tell viewers what the video is about; "
                "a purely visual hook may also work, so review rather than apply blindly."),
        "evidence": _ev("segment", [r["segment_index"] for r in opening]),
    }


def _late_first_word(rows: list[dict], words: list[dict], ocr: dict | None) -> dict | None:
    if not words or words[0]["start_s"] < LATE_FIRST_WORD_S:
        return None
    first = words[0]
    before = _segs_in(rows, 0.0, first["start_s"])
    if any(r.get("speech_status") != AVAILABLE for r in before):
        return None
    text_ids = sorted({d["event_id"] for f in (ocr or {}).get("frames") or [] if f["t_s"] < first["start_s"]
                       for d in f["detections"] if "event_id" in d}) if ocr and ocr.get("measured") else []
    why = f"The first transcribed word starts at {first['start_s']:g} s."
    why += (" On-screen text is visible before it, so the opening may be text-led; check that the text carries the hook."
            if text_ids else " A shorter lead-in gets to the spoken content sooner.")
    return {
        "rule": "late_first_word",
        "start_s": 0.0, "end_s": round(first["start_s"], 3),
        "suggestion": f"Consider trimming or tightening the {first['start_s']:g} s before the first spoken word.",
        "why": why,
        "evidence": _ev("word", [first["word_index"]]) + _ev("segment", [r["segment_index"] for r in before])
        + _ev("text_event", text_ids),
    }


def _quiet_stretches(rows: list[dict], words: list[dict], librosa: dict | None, duration: float) -> list[dict]:
    if not librosa or not librosa.get("measured"):
        return []
    out = []
    for iv in librosa["low_energy_intervals"]:
        s, e = iv["start_s"], iv["end_s"]
        if e - s < QUIET_MIN_S or e >= duration - 0.25:  # a quiet tail is usually the ending, skip it
            continue
        segs = _segs_in(rows, s, e)
        if not segs or any(r.get("audio_status") != AVAILABLE for r in segs):
            continue
        if any(w["start_s"] < e and w["end_s"] > s for w in words):
            continue
        out.append({
            "rule": "quiet_stretch",
            "start_s": s, "end_s": e,
            "suggestion": f"Review the {e - s:.1f} s of near-silent audio at {s:g}-{e:g} s: trim it, or add sound/voice if it is unintended.",
            "why": ("The audio stays below the silence threshold here with no transcribed speech. Silence can be deliberate "
                    "(a pause or reveal), so this only flags it for review."),
            "evidence": _ev("librosa_interval", [iv["interval_id"]]) + _ev("segment", [r["segment_index"] for r in segs]),
        })
    return out


def _static_stretches(rows: list[dict]) -> list[dict]:
    out, run = [], []

    def close() -> None:
        if run and run[-1]["end_s"] - run[0]["start_s"] >= STATIC_MIN_S:
            s, e = run[0]["start_s"], run[-1]["end_s"]
            out.append({
                "rule": "static_stretch",
                "start_s": s, "end_s": e,
                "suggestion": f"Consider a visual change (cut, zoom, overlay or B-roll) during {s:g}-{e:g} s.",
                "why": (f"{e - s:.1f} s in one scene with mean frame-to-frame motion below {STATIC_MOTION_MAX:g}. "
                        "Static shots can be intentional (talking head, reading time for text), so check before editing."),
                "evidence": _ev("segment", [r["segment_index"] for r in run]),
            })

    for r in rows:
        m = r.get("motion_mean")
        ok = r.get("visual_status") == AVAILABLE and m is not None and m < STATIC_MOTION_MAX
        if ok and run and r.get("scene_index") == run[-1].get("scene_index"):
            run.append(r)
        else:
            close()
            run = [r] if ok else []
    close()
    return out


def _gemini(gemini: dict | None, rows: list[dict]) -> list[dict]:
    if not gemini or not gemini.get("measured"):
        return []
    out = []
    for sg in gemini["analysis"]["edit_suggestions"]:
        segs = _segs_in(rows, sg["start_s"], sg["end_s"])
        out.append({
            "rule": "gemini_suggestion",
            "start_s": sg["start_s"], "end_s": sg["end_s"],
            "suggestion": sg["action"],
            "why": f"{sg['rationale']} Might help because: {sg['expected_effect']}",
            "evidence": _ev("gemini_suggestion", [sg["id"]]) + _ev("gemini_moment", sg.get("evidence_moment_ids") or [])
            + _ev("segment", [r["segment_index"] for r in segs]),
            "model_confidence": sg["confidence"],
        })
    return out


def build_recommendations(rows: list[dict], words: list[dict], ocr: dict | None, librosa: dict | None,
                          gemini: dict | None, duration: float | None) -> list[dict]:
    duration = duration or (rows[-1]["end_s"] if rows else 0.0)
    if not rows:
        return []
    rule_recs = [r for r in (_opening_without_words(rows, words, ocr, duration), _late_first_word(rows, words, ocr)) if r]
    rule_recs += _quiet_stretches(rows, words, librosa, duration) + _static_stretches(rows)
    recs: list[dict[str, Any]] = []
    for r in sorted(rule_recs, key=lambda r: (r["start_s"], r["rule"])):
        recs.append({"id": f"r{len(recs)}", "source": "rule", "basis": "measured", **r, "caveat": CAVEAT})
    for r in _gemini(gemini, rows):
        recs.append({"id": f"r{len(recs)}", "source": "gemini", "basis": "model_interpretation", **r,
                     "caveat": "Model interpretation; verify against the video before editing."})
    return recs


def evidence_index(rows: list[dict], words: list[dict], ocr: dict | None, librosa: dict | None,
                   gemini: dict | None) -> dict[str, set]:
    analysis = (gemini or {}).get("analysis") or {}
    return {
        "segment": {r["segment_index"] for r in rows},
        "word": {w["word_index"] for w in words},
        "text_event": {e["event_id"] for e in (ocr or {}).get("events") or []},
        "librosa_interval": {iv["interval_id"] for iv in (librosa or {}).get("low_energy_intervals") or []},
        "gemini_moment": {m["id"] for m in analysis.get("moments") or []},
        "gemini_suggestion": {s["id"] for s in analysis.get("edit_suggestions") or []},
    }


def invalid_evidence(recs: list[dict], index: dict[str, set]) -> list[str]:
    """Problems with recommendation evidence: unknown types, dangling ids, empty evidence, bad intervals."""
    problems = []
    for r in recs:
        if not r.get("evidence"):
            problems.append(f"{r['id']}: no evidence")
        if not r["start_s"] < r["end_s"]:
            problems.append(f"{r['id']}: empty interval")
        for ev in r.get("evidence") or []:
            if ev["type"] not in index:
                problems.append(f"{r['id']}: unknown evidence type {ev['type']!r}")
            elif ev["id"] not in index[ev["type"]]:
                problems.append(f"{r['id']}: {ev['type']} {ev['id']!r} does not exist")
    return problems


def recommendation_ids(recs: list[dict], seg: dict) -> list[str] | None:
    return [r["id"] for r in recs if r["start_s"] < seg["end_s"] and r["end_s"] > seg["start_s"]] or None
