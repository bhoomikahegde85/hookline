"""Strongest and weakest moments of a video, from TTCC's predicted curve plus measured features (``moments.json``).

A model prediction, not audience data. The per-second hazard h(t) = -ln(R(t) / R(t-1)) is the predicted share of the *remaining* viewers
lost in second t. A ``window_s``-second window is scored by its mean hazard: the highest are the weakest moments, the lowest the strongest.
The opening (before ``skip_opening_s``) is left out: the hook is reported separately and TTCC always predicts a steep opening. Ranking is
relative within this video, so a "strongest" moment can still be a falling stretch. Each moment lists what was measured inside it
(``dropoffs.Evidence``); the ``why`` lines only state what that evidence shows, as hypotheses, never as causes.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import dropoffs

MOMENTS_FILE = "moments.json"
NOTE = "model prediction, not audience data; moments are ranked by predicted hazard, a relative measure within this video"
MIN_RETENTION_TO_RANK = 0.02
MIN_HAZARD_SPREAD = 1e-9
EPS = 1e-6


def _hazards(retention: list[float]) -> list[float]:
    """h[t] for t = 0..T (h[0] = 0): the predicted share of the remaining viewers lost in second t, never negative."""
    out = [0.0]
    for t in range(1, len(retention)):
        prev, cur = retention[t - 1], retention[t]
        out.append(0.0 if prev <= 0 else max(0.0, -math.log(max(cur, EPS) / prev)))
    return out


def _pick(windows: list[tuple[int, float]], taken: list[tuple[int, int]], w: int, descending: bool) -> tuple[int, float] | None:
    """Best window not overlapping any in ``taken`` (ties, compared to 9 decimals, go to the earlier window); it is added to ``taken``."""
    for s, h in sorted(windows, key=lambda x: (-round(x[1], 9) if descending else round(x[1], 9), x[0])):
        if all(s + w <= a or b <= s for a, b in taken):
            taken.append((s, s + w))
            return s, h
    return None


def _heuristic(segments: list[dict] | None, start: float, end: float) -> dict | None:
    """Overlap-weighted mean of the dropoff file's heuristic attention_risk / moment_quality over [start, end)."""
    acc: dict[str, list[tuple[float, float]]] = {"attention_risk": [], "moment_quality": []}
    for seg in segments or []:
        try:
            overlap = min(end, seg["end_s"]) - max(start, seg["start_s"])
            if overlap <= 0 or not seg["heuristic"]["available"]:
                continue
            for key in acc:
                mean = seg["heuristic"][key]["mean"]
                if mean is not None:
                    acc[key].append((overlap, float(mean)))
        except (KeyError, TypeError, ValueError):
            continue
    if not any(acc.values()):
        return None
    return {key: round(sum(o * v for o, v in vals) / sum(o for o, _ in vals), 1) if vals else None for key, vals in acc.items()}


def _count(observed: dict | None, key: str) -> int | None:
    v = (observed or {}).get(key)
    return v["count"] if isinstance(v, dict) and isinstance(v.get("count"), int) else None


def _quiet(observed: dict | None) -> list:
    v = (observed or {}).get("quiet_audio")
    return v if isinstance(v, list) else []


def _text(observed: dict | None) -> list:
    v = (observed or {}).get("on_screen_text")
    return v if isinstance(v, list) else []


def _why_weak(observed: dict | None, length: float, fall: float) -> list[str]:
    words, cuts, quiet = _count(observed, "words"), _count(observed, "scene_cuts"), _quiet(observed)
    n = f"{length:g} s"
    why = []
    if words == 0 and cuts == 0:
        why.append(f"no speech and no scene cut for {n} while the curve falls (may read as dead air)")
    elif words == 0:
        why.append(f"no speech transcribed for {n} while the curve falls")
    elif cuts == 0:
        why.append(f"no scene cut for {n} while the curve falls (the shot may feel static)")
    if cuts and length > 0 and cuts / length >= dropoffs.THRESHOLDS["fast_cut_rate"]:
        why.append(f"{cuts} scene cuts in {n} (very fast cutting may be hard to follow)")
    if quiet:
        why.append("quiet audio overlaps this window (may read as a lull)")
    if words == 0 and _text(observed):
        why.append("only on-screen text is present here, no speech")
    return why or [f"predicted fall of {fall:.0%} of the starting audience is the largest in this video after the opening; "
                   "no measured feature here points to a reason"]


def _why_strong(observed: dict | None) -> list[str]:
    words, cuts = _count(observed, "words"), _count(observed, "scene_cuts")
    holds = "while predicted retention holds up better than elsewhere in this video"
    why = []
    if words and cuts:
        why.append(f"speech ({words} words) plus {cuts} scene cut{'s' if cuts != 1 else ''} {holds}")
    elif words:
        why.append(f"speech ({words} words) {holds}")
    elif cuts:
        why.append(f"{cuts} scene cut{'s' if cuts != 1 else ''} {holds}")
    texts = _text(observed)
    if texts:
        why.append(f"on-screen text is shown ({'; '.join(str(t.get('text')) for t in texts[:2])}) {holds}")
    return why or ["predicted retention holds up better here than elsewhere in this video; no measured feature in this window explains it"]


def _suggest(observed: dict | None, start: float, end: float) -> dict:
    """Same rule logic as dropoffs.candidates, via dropoffs.edit_proposal."""
    words, cuts, length = _count(observed, "words"), _count(observed, "scene_cuts"), end - start
    if _quiet(observed):
        rule, text = "quiet_stretch", "Audio is quiet here: consider trimming this stretch if it is not intended."
    elif words == 0 and length >= dropoffs.THRESHOLDS["no_speech_min_seconds"]:
        rule, text = "interval_no_speech", "No speech here: consider whether a voice line or caption should carry the message."
    elif cuts is not None and length > 0 and cuts / length >= dropoffs.THRESHOLDS["fast_cut_rate"]:
        rule, text = "interval_fast_cutting", "Very fast cutting: check that each shot is readable and the pace is intended."
    else:
        rule, text = "review", "The curve is predicted to fall most here: watch this stretch and decide whether it earns its place."
    proposal = dropoffs.edit_proposal(rule, start, end)
    return {"action": proposal["action"], "start_s": start, "end_s": end, "text": text} | ({"note": proposal["note"]} if "note" in proposal else {})


def _empty(art: dict, window_s: int, skip: int, stitched: bool, reason: str) -> dict:
    return {"schema": "hookline.moments", "video_id": art.get("video_id"), "window_s": window_s, "skip_opening_s": skip,
            "curve_is_stitched": stitched, "strongest": [], "weakest": [], "reason": reason, "note": NOTE}


def find_moments(video_dir: Path, k: int = 3, window_s: int = 3, skip_opening_s: int = 3) -> dict | None:
    """Strongest and weakest ``window_s``-second moments after the opening; None without a successful, well-formed TTCC prediction."""
    art = dropoffs._load(video_dir / "ttcc_retention.json")
    if not isinstance(art, dict) or art.get("status") != "success":
        return None
    r = art.get("retention")
    if not isinstance(r, list) or not r or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in r):
        return None
    ts = art.get("timestamps_s")
    if ts is not None and ts != list(range(len(r))):
        return None
    if k < 1 or window_s < 1 or skip_opening_s < 0:
        return None
    retention = [float(x) for x in r]
    stitched = bool(art.get("stitched"))
    horizon = len(retention) - 1
    if horizon < skip_opening_s + window_s:
        return _empty(art, window_s, skip_opening_s, stitched,
                      f"video is {horizon} s of predicted curve; at least {skip_opening_s + window_s} s are needed (opening {skip_opening_s} s is excluded)")

    h = _hazards(retention)
    # a window that starts when almost nobody is left to lose says nothing (its hazard is ~0 by construction): leave it out
    windows = [(s, sum(h[s + 1:s + window_s + 1]) / window_s) for s in range(skip_opening_s, horizon - window_s + 1) if retention[s] >= MIN_RETENTION_TO_RANK]
    if len(windows) < 2:
        return _empty(art, window_s, skip_opening_s, stitched, "the predicted curve has fallen below 2% before any moment can be ranked")
    if max(x for _, x in windows) - min(x for _, x in windows) < MIN_HAZARD_SPREAD:
        return _empty(art, window_s, skip_opening_s, stitched, "the predicted curve is flat after the opening: no moment stands out")
    median = statistics.median(h[skip_opening_s + 1:])

    taken: list[tuple[int, int]] = []
    weak: list[tuple[int, float]] = []
    strong: list[tuple[int, float]] = []
    for _ in range(k):  # alternate, so a short video still gets some of each
        for found, descending in ((weak, True), (strong, False)):
            if (pick := _pick(windows, taken, window_s, descending)) is not None:
                found.append(pick)

    try:
        ev: dropoffs.Evidence | None = dropoffs.Evidence(video_dir)
    except Exception:  # noqa: BLE001 - a malformed evidence file must not stop the moments
        ev = None
    seg_doc = dropoffs._load(video_dir / dropoffs.DROPOFFS_FILE)
    segments = seg_doc.get("segments") if isinstance(seg_doc, dict) else None
    segments = segments if isinstance(segments, list) else None

    def build(rank: int, s: int, mean_h: float, kind: str) -> dict:
        a, b = s, s + window_s
        try:
            observed: Any = ev.observed(a, b) if ev else None
        except Exception:  # noqa: BLE001
            observed = None
        return {"rank": rank, "start_s": a, "end_s": b, "mean_hazard": round(mean_h, 6), "retention_start": retention[a], "retention_end": retention[b],
                "predicted_fall": retention[a] - retention[b], "relative_to_video_median_hazard": round(mean_h / median, 3) if median > 0 else None,
                "observed": observed, "heuristic": _heuristic(segments, a, b),
                "why": _why_weak(observed, window_s, retention[a] - retention[b]) if kind == "weak" else _why_strong(observed),
                "suggested_action": _suggest(observed, a, b) if kind == "weak" else None}

    out = {"schema": "hookline.moments", "video_id": art.get("video_id"), "window_s": window_s, "skip_opening_s": skip_opening_s,
           "curve_is_stitched": stitched,
           "strongest": [build(n, s, x, "strong") for n, (s, x) in enumerate(strong, 1)],
           "weakest": [build(n, s, x, "weak") for n, (s, x) in enumerate(weak, 1)], "note": NOTE}
    if len(strong) < k or len(weak) < k:
        out["note"] += f"; the video is too short for {k} distinct non-overlapping windows in each list, so fewer are shown ({len(weak)} weakest, {len(strong)} strongest)"
    return out


def write_moments(video_dir: Path, moments: dict) -> Path:
    path = video_dir / MOMENTS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(moments, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path
