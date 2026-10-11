"""How strong is a video's hook, and why (``hook_report.json``).

Every factor is a value measured on the video's own first seconds (or a TTCC model prediction), placed among the reference reels
(the regular reels of the dataset, outliers excluded, the video itself excluded) as a percentile. A fixed, hand-written rule per factor
(``HOOK_RULES``: lower is better, higher is better, closest to the reference median is best, or no direction) turns the percentile into
helps / hurts / neutral. The rules are rules of thumb: nothing here is fitted, nothing is validated against audience data, and none of it
says a video will or will not go viral. Account data (follower count, handle, the video's own reach) is never read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import views_eval as ve
from .hook_score import percentile

REPORT_FILE = "hook_report.json"
MIN_REFERENCE = 20  # fewest reference values needed before a percentile is given
HELPS_AT, HURTS_AT = 67.0, 33.0  # favourability (0-100) at or above / below which a factor helps / hurts
STRONG_AT, WEAK_AT = 67, 33  # hook_scores.json percentile cut-offs for the verdict
MIN_DIRECTIONAL = 4  # factors with a direction needed for a verdict without TTCC
RULE_TAG = "Rule of thumb, not validated against audience data: "

# key -> label, unit, rule, group (where the value comes from), event (what "absent" means), why. Written by hand before any result was looked at.
HOOK_RULES: dict[str, dict] = {
    "first_word_s": {"label": "Time to first spoken word", "unit": "s", "rule": "lower_better", "group": "measured", "event": "spoken word",
                     "why": RULE_TAG + "an early voice may give viewers a reason to stay before they decide to scroll on."},
    "first_cut_s": {"label": "Time to first visible cut", "unit": "s", "rule": "lower_better", "group": "measured", "event": "cut",
                    "why": RULE_TAG + "a long wait with nothing changing on screen may let attention drift; a single continuous shot is not penalised."},
    "cuts_per_s_0_5": {"label": "Cuts per second", "unit": "cuts/s", "rule": "none", "group": "measured", "event": None,
                       "why": RULE_TAG + "shown for context, not scored: in the reference reels cuts per second had no measurable relation to outcome (Spearman about 0.0), so neither a fast nor a slow opening is treated as better or worse."},
    "motion_mean_0_5": {"label": "Mean motion", "unit": "frame-difference units", "rule": "higher_better", "group": "measured", "event": None,
                        "why": RULE_TAG + "visible movement in the opening may catch the eye more than a still frame."},
    "brightness_mean_0_5": {"label": "Brightness", "unit": "0-255", "rule": "mid_best", "group": "measured", "event": None,
                            "why": RULE_TAG + "a frame that is very dark or washed out may be harder to read at a glance; near the reference median is treated as safest."},
    "rms_dbfs_0_5": {"label": "Loudness", "unit": "dBFS", "rule": "higher_better", "group": "measured", "event": None,
                     "why": RULE_TAG + "a very quiet opening may not register with viewers who have the sound on."},
    "words_per_s_0_5": {"label": "Speaking pace", "unit": "words/s", "rule": "mid_best", "group": "measured", "event": None,
                        "why": RULE_TAG + "speech that is much slower or much faster than typical may be harder to follow; near the reference median is treated as safest."},
    "first_text_s": {"label": "Time to first on-screen text", "unit": "s", "rule": "lower_better", "group": "measured", "event": "on-screen text",
                     "why": RULE_TAG + "early text may give context to viewers watching without sound; a reel with no text is not penalised."},
    "person_max_0_3s": {"label": "Person on screen in the first 3 s", "unit": "SigLIP score", "rule": "higher_better", "group": "person", "event": None,
                        "why": RULE_TAG + "a person in the opening frames may make the video feel more personal and easier to connect with."},
    "ttcc_R3": {"label": "Predicted hold at 3 s", "unit": "fraction", "rule": "higher_better", "group": "ttcc", "event": None,
                "why": RULE_TAG + "TTCC's predicted share still watching at 3 s; a higher value may mean fewer early exits. A model prediction, not measured retention."},
    "ttcc_R5": {"label": "Predicted hold at 5 s", "unit": "fraction", "rule": "higher_better", "group": "ttcc", "event": None,
                "why": RULE_TAG + "TTCC's predicted share still watching at 5 s; a higher value may mean fewer early exits. A model prediction, not measured retention."},
    "ttcc_R10": {"label": "Predicted hold at 10 s", "unit": "fraction", "rule": "higher_better", "group": "ttcc", "event": None,
                 "why": RULE_TAG + "TTCC's predicted share still watching at 10 s; a higher value may mean the hook carries into the body. A model prediction, not measured retention."},
}
NOTE = ("This report compares the video with reference reels, using rules of thumb that are not validated against audience data. It is "
        "not a viral verdict: whether a reel spreads also depends on distribution, timing and trends that are not in the video.")


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else float("nan")


def _load(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _window(video_dir: Path) -> float:
    o = _load(video_dir / "opening_multimodal.json") or {}
    w = _num(o.get("analysed_seconds"))
    return w if w > 0 else 5.0


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - a corrupt optional artifact must never stop the report
        return None


def _reference(data_root: Path, features_root: Path, ttcc_root: Path | None, exclude_id: str) -> list[dict]:
    """Regular reels (outliers excluded, the video itself excluded) with measured / person / TTCC features attached."""
    rows = _safe(ve.load_dataset, data_root)
    if not rows:
        return []
    _safe(ve.add_labels, rows)
    rows = [r for r in rows if not r["is_outlier"] and r["video_id"] != exclude_id]
    if _safe(ve.assemble, rows, features_root, ttcc_root) is not None:
        return rows
    kept = []
    for r in rows:  # one corrupt artifact in the reference set drops that reel only
        if _safe(ve.assemble, [r], features_root, ttcc_root) is not None:
            kept.append(r)
    return kept


def _favourability(rule: str, p: float) -> float | None:
    if rule == "lower_better":
        return 100.0 - p
    if rule == "higher_better":
        return p
    if rule == "mid_best":
        return 100.0 - 2.0 * abs(p - 50.0)
    return None


def _direction(rule: str, p: float | None) -> str:
    fav = None if p is None else _favourability(rule, p)
    if p is None:
        return "unknown"
    if fav is None:
        return "neutral"
    return "helps" if fav >= HELPS_AT else ("hurts" if fav <= HURTS_AT else "neutral")


def _fmt(v: float, unit: str) -> str:
    return f"{round(v, 3):g} {unit}".strip()


def _factor(key: str, value: float, ref_vals: np.ndarray, win: float, no_event: bool) -> dict:
    spec = HOOK_RULES[key]
    rule = spec["rule"]
    p = percentile(value, ref_vals) if len(ref_vals) >= MIN_REFERENCE else float("nan")
    p = None if math.isnan(p) else round(p, 1)
    source = {"measured": f"measured over the first {win:g} s of the video (opening_multimodal.json)",
              "person": "SigLIP zero-shot score over the first 3 s (siglip.json): a model output, not a measurement",
              "ttcc": "TTCC model prediction from ttcc_retention.json: unvalidated, not measured retention"}[spec["group"]]
    if not math.isfinite(value):
        return {"key": key, "label": spec["label"], "value": None, "unit": spec["unit"], "percentile_vs_reference": None,
                "reading": "not measured for this video (the input was missing)", "direction": "unknown", "why_it_matters": spec["why"], "source": source}
    direction = "neutral" if no_event else _direction(rule, p)
    if no_event:
        reading = f"no {spec['event']} in the first {win:g} s" + (" (single continuous shot)" if key == "first_cut_s" else "") + ": not counted for or against"
    elif p is None:
        reading = f"{_fmt(value, spec['unit'])}: too few reference reels to compare"
    else:
        where = f"lower than {100 - p:.0f}% of" if rule == "lower_better" else f"higher than {p:.0f}% of"
        reading = f"{_fmt(value, spec['unit'])}: {where} the reference reels"
        if rule == "mid_best":
            reading += f" (reference median {_fmt(float(np.median(ref_vals)), spec['unit'])}; nearer the median is treated as better)"
    return {"key": key, "label": spec["label"], "value": round(value, 4), "unit": spec["unit"], "percentile_vs_reference": p, "reading": reading,
            "direction": direction, "why_it_matters": spec["why"], "source": source}


def _gemini_text(video_dir: Path) -> str | None:
    art = _load(video_dir / "gemini_edit.json")
    if not art or art.get("measured") is not True or art.get("status") not in ("available", "uncertain"):
        return None
    text = (((art.get("analysis") or {}).get("overall") or {}).get("hook_assessment"))
    return text.strip() if isinstance(text, str) and text.strip() else None


def _hook_percentile(video_dir: Path) -> float | None:
    s = _load(video_dir / "hook_scores.json")
    v = ((s or {}).get("hook_score") or {}).get("score")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 100 else None


def _verdict(video_dir: Path, factors: list[dict]) -> dict:
    hp = _hook_percentile(video_dir)
    if hp is not None:
        label = "strong" if hp >= STRONG_AT else ("weak" if hp <= WEAK_AT else "average")
        return {"label": label, "percentile": hp,
                "basis": (f"TTCC's predicted early hold (mean of R3, R5 and, for clips of 10 s or more, R10) is at the {hp:.0f}th percentile of the reference reels. This is a model "
                          "prediction of retention, unvalidated, and not audience data.")}
    known = [f for f in factors if f["direction"] != "unknown"]
    if len(known) >= MIN_DIRECTIONAL:
        helps = sum(f["direction"] == "helps" for f in known)
        hurts = sum(f["direction"] == "hurts" for f in known)
        share = helps / (helps + hurts) if helps + hurts else 0.5
        label = "strong" if share >= 0.67 and helps >= 2 else ("weak" if share <= 0.33 and hurts >= 2 else "average")
        return {"label": label, "percentile": None,
                "basis": (f"{helps} of {len(known)} factors help and {hurts} hurt, from measured opening features only, no TTCC. "
                          "Rule-of-thumb reading, not validated.")}
    return {"label": "not rated", "percentile": None,
            "basis": f"only {len(known)} factor(s) could be compared with the reference reels (at least {MIN_DIRECTIONAL} are needed) and no TTCC hook score exists"}


def _empty(video_id: str, reason: str, n: int = 0) -> dict:
    return {"schema": "hookline.hook_report", "video_id": video_id, "status": "unavailable", "reason": reason,
            "verdict": {"label": "not rated", "basis": reason, "percentile": None}, "factors": [], "strengths": [], "weaknesses": [],
            "gemini_hook_assessment": None, "reference": {"n_reels": n, "n_with_ttcc": 0, "note": "no comparison was made"}, "note": NOTE.format(n=n or "the")}


def build_hook_report(video_dir: Path, data_root: Path, features_root: Path, ttcc_root: Path | None = None) -> dict:
    """The hook report for the video whose artifacts are in ``video_dir``; never raises. ``data_root`` / ``features_root`` / ``ttcc_root`` describe the
    reference reels."""
    vid = video_dir.name
    measured = _safe(ve.measured_features, video_dir)
    if measured is None:
        return _empty(vid, "no measured opening features for this video (opening_multimodal.json missing or unreadable); run the pipeline first")
    m, _duration = measured
    person = _safe(ve.person_features, video_dir)
    ttcc = _safe(ve.ttcc_features, video_dir)
    tf = ttcc[0] if ttcc else None
    win = _window(video_dir)
    ref = _reference(data_root, features_root, ttcc_root, vid)
    n_ttcc = sum(r.get("ttcc") is not None for r in ref)

    def ref_vals(group: str, key: str) -> np.ndarray:
        src = {"measured": lambda r: r.get("measured"), "person": lambda r: r.get("person"), "ttcc": lambda r: r.get("ttcc")}[group]
        v = np.array([_num((src(r) or {}).get(key)) for r in ref], float)
        return v[np.isfinite(v)]

    values = {"measured": m, "person": person, "ttcc": tf}
    factors = []
    for key, spec in HOOK_RULES.items():
        group = spec["group"]
        if values[group] is None and group != "measured":
            continue  # SigLIP or TTCC output absent: no such factor
        value = _num((values[group] or {}).get(key))
        no_event = False
        if key in ("first_word_s", "first_text_s") and math.isfinite(value):
            no_event = value >= win
        elif key == "first_cut_s" and math.isfinite(value):
            no_event = value >= win and _num(m.get("cuts_per_s_0_5")) == 0.0
        factors.append(_factor(key, value, ref_vals(group, key), win, no_event))

    scored = {f["key"]: _favourability(HOOK_RULES[f["key"]]["rule"], f["percentile_vs_reference"]) for f in factors
              if f["direction"] in ("helps", "hurts") and f["percentile_vs_reference"] is not None}
    strengths = sorted((f["key"] for f in factors if f["direction"] == "helps"), key=lambda k: -scored[k])[:3]
    weaknesses = sorted((f["key"] for f in factors if f["direction"] == "hurts"), key=lambda k: scored[k])[:3]
    n = len(ref)
    return {"schema": "hookline.hook_report", "video_id": vid, "status": "ok", "reason": None, "verdict": _verdict(video_dir, factors), "factors": factors,
            "strengths": strengths, "weaknesses": weaknesses, "gemini_hook_assessment": _gemini_text(video_dir),
            "reference": {"n_reels": n, "n_with_ttcc": n_ttcc,
                          "note": (f"Percentile = share of the reference reels (regular reels of the dataset, outliers and this video excluded) with a lower value; "
                                   f"TTCC factors are compared with the {n_ttcc} that have a TTCC prediction.")},
            "note": NOTE.format(n=n or "the")}


def write_hook_report(video_dir: Path, report: dict) -> Path:
    path = video_dir / REPORT_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path
