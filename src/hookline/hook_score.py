"""0-100 scores for one video, each with the measured accuracy of the model behind it (``hook_scores.json``).

A bare score is easy to produce; what makes it believable is the evidence next to it. Every number here is a *position in a reference set*
(the 100 regular reels with real Instagram views), and the file carries the accuracy of the model that produced it, measured by
``scripts/eval_views.py`` with creator-grouped cross-validation. Nothing here is audience data, and none of it says a video "will go viral".

- ``hook_score``      percentile of the video's TTCC-predicted early hold, mean of R(3), R(5), R(10), among the other videos' predictions
                      (50 = typical). A model prediction of retention, unvalidated. Needs TTCC hook predictions for other videos.
- ``breakout_score``  percentile of the predicted overperformance (log views relative to the follower band's median) among the reference
                      reels' held-out predictions. 100 = the model expects the strongest overperformance seen in other videos.
                      Shown with the AUC / interval / permutation p of the model used, and an explicit verdict when that is not above chance.
- ``retention_pct``   TTCC's predicted share still watching at 1, 3, 5, 10 s (0-100 %). A model prediction.

The scores judge the VIDEO, not the account: the model uses only features computed from the video's own content (never the follower band, the page,
its type or the video's views). The model follows a fixed rule: ``measured_plus_ttcc`` when TTCC output exists for the video and for at least 30
other videos, else ``measured``. A video that is itself in other videos is scored by a model fitted without its creator.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import views_eval as ve

SCORES_FILE = "hook_scores.json"
MIN_REFERENCE = ve.MIN_TTCC_VIDEOS
NOTE = ("Scores are positions among other videos with real views, from models whose measured accuracy is listed next to them. They are "
        "model predictions, not audience data, and not a statement that a video will or will not go viral: reach also depends on distribution, "
        "timing and trends that are not in the video.")


def percentile(value: float, reference: np.ndarray) -> float:
    ref = np.asarray([v for v in reference if np.isfinite(v)], float)
    if not len(ref) or not math.isfinite(value):
        return float("nan")
    return float(100.0 * ((ref < value).sum() + 0.5 * (ref == value).sum()) / len(ref))


def latest_run(eval_root: Path) -> Path | None:
    p = eval_root / "LATEST"
    run = eval_root / p.read_text().strip() if p.is_file() else None
    return run if run and (run / "metrics_summary.json").is_file() else None


def accuracy_block(run: Path | None, model: str) -> dict:
    """What the latest evaluation measured for this model, with a plain verdict (no spin)."""
    if run is None:
        return {"model": model, "evaluated": False, "verdict": "no evaluation run found: run scripts/eval_views.py"}
    s = json.loads((run / "metrics_summary.json").read_text())
    m = (s.get("models") or {}).get(model)
    if not m:
        return {"model": model, "evaluated": False, "run": run.name, "verdict": f"{model} was not evaluated in {run.name}"}
    ci = ((s.get("bootstrap_95ci") or {}).get(model) or {}).get("auc")
    p = ((s.get("permutation") or {}).get(model) or {}).get("p_auc")
    if ci and ci[0] > 0.5:
        verdict = f"better than chance on {s['label']['n_regular']} reels (95% interval above 0.5), still a weak ranker"
    elif m["auc"] is not None and m["auc"] < 0.5:
        verdict = "no signal found: it ranked these reels slightly worse than chance (AUC below 0.5)"
    else:
        verdict = "not shown to beat chance on these reels (the 95% interval includes 0.5)"
    return {"model": model, "evaluated": True, "run": run.name, "n_reels": m["n"], "auc": m["auc"], "auc_ci95": ci, "permutation_p_auc": p,
            "spearman": m.get("spearman"), "verdict": verdict, "metric": "AUC for 'beats its follower band's median views', fold-wise, creator-grouped"}


def _popularity(video_dir: Path, target: dict, ref: list[dict], run: Path | None, cfg: dict, tribe_root: Path | None) -> dict:
    """A separate score from TRIBE v2 alone (its predicted brain response over the opening 10 s). It never uses TTCC or the measured features,
    and is not combined with them. Percentile of the predicted over-performance among the other videos that have TRIBE output."""
    base = {"model": "tribe_only", "features_used": ["tribe"], "accuracy": accuracy_block(run, "tribe_only")}
    if tribe_root is None:
        return base | {"score": None, "reason": "no TRIBE other videos configured"}
    tf, info = ve.tribe_features(video_dir)
    pool = [r for r in ref if r["tribe"] is not None and r["handle"] != target["handle"]]
    if len(pool) < MIN_REFERENCE:
        return base | {"score": None, "reason": f"only {len(pool)} other videos have TRIBE output; {MIN_REFERENCE} are needed"}
    if tf is None:
        return base | {"score": None, "reason": f"no TRIBE output for this video ({info['reason']})"}
    tgt = dict(target, tribe=tf)
    y = np.array([r["overperformance"] for r in pool])
    X = ve.design(pool, "tribe_only")
    pred = float(ve.fit_predict(X, y, ve.design([tgt], "tribe_only"), cfg["alpha"])[0])
    ref_scores = ve.cv_scores(X, y, [r["handle"] for r in pool], cfg["folds"], 5, cfg["alpha"], cfg["seed"])[0].mean(0)
    sc = percentile(pred, ref_scores)
    return base | {"score": None if math.isnan(sc) else round(sc), "predicted_overperformance_log": round(pred, 3), "reference_reels": len(pool),
                   "meaning": "percentile of the predicted over-performance among the other videos that have TRIBE output (50 = typical); TRIBE predicts "
                              "brain response, not retention, and this score is not combined with TTCC"}


def compute_scores(video_dir: Path, data_root: Path, features_root: Path, ttcc_root: Path | None, eval_root: Path | None = None,
                   cfg: dict | None = None, tribe_root: Path | None = None) -> dict:
    """Scores for the video whose artifacts are in ``video_dir``. ``features_root``/``ttcc_root``: the other videos' artifacts."""
    cfg = ve.DEFAULTS | (cfg or {})
    rows = ve.load_dataset(data_root)
    ve.add_labels(rows)
    ve.assemble(rows, features_root, ttcc_root, tribe_root)
    ref = [r for r in rows if not r["is_outlier"]]
    vid = video_dir.name
    self_row = next((r for r in rows if r["video_id"] == vid), None)
    target = {"video_id": vid, "band": "50k-200k", "handle": self_row["handle"] if self_row else f"__new__{vid}"}  # band/page_type are never features of a score
    try:
        target["measured"], target["duration_s"] = ve.measured_features(video_dir)
    except (OSError, ValueError, KeyError) as exc:
        return {"video_id": vid, "status": "unavailable", "reason": f"no measured features for this video ({type(exc).__name__}); run scripts/run_pipeline.py first", "note": NOTE}
    tf, tinfo = ve.ttcc_features(video_dir)
    target["ttcc"], target["tribe"] = tf, None
    reference_ttcc = [r for r in ref if r["ttcc"] is not None]
    use_ttcc = tf is not None and len(reference_ttcc) >= MIN_REFERENCE
    model = "measured_plus_ttcc" if use_ttcc else "measured"
    run = latest_run(eval_root) if eval_root else None
    out: dict = {"schema": "hookline.hook_scores", "video_id": vid, "status": "ok", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "in_reference_set": self_row is not None, "note": NOTE}

    # ---- breakout: percentile of the predicted overperformance among the other videos' held-out predictions
    pool = [r for r in ref if r["handle"] != target["handle"] and (r["ttcc"] is not None or not use_ttcc)]
    y = np.array([r["overperformance"] for r in pool])
    X = ve.design(pool, model)
    pred = float(ve.fit_predict(X, y, ve.design([target], model), cfg["alpha"])[0])
    ref_scores = ve.cv_scores(X, y, [r["handle"] for r in pool], cfg["folds"], 5, cfg["alpha"], cfg["seed"])[0].mean(0)
    bp = percentile(pred, ref_scores)
    ref_dur = np.array([float(r["duration_s"]) for r in pool if r.get("duration_s") and math.isfinite(float(r["duration_s"]))])
    vdur = float(target.get("duration_s") or 0)
    outside = bool(len(ref_dur) and vdur and np.percentile(ref_dur, 95) > np.percentile(ref_dur, 5) and not (np.percentile(ref_dur, 5) <= vdur <= np.percentile(ref_dur, 95)))  # duration is a model input: far outside the reference range is extrapolation
    out["breakout_score"] = {"score": None if (math.isnan(bp) or outside) else round(bp),
                             **({"reason": f"this video ({vdur:.1f} s) is outside the 5th-95th percentile duration of the other videos "
                                           f"({np.percentile(ref_dur, 5):.0f}-{np.percentile(ref_dur, 95):.0f} s), so the model would be extrapolating"} if outside else {}), "predicted_overperformance_log": round(pred, 3),
                             "meaning": "percentile of the predicted overperformance (log views vs the follower band's median) among the other videos; "
                                        "50 = typical, 100 = strongest seen",
                             "reference_reels": len(pool), "accuracy": accuracy_block(run, model), "features_used": list(ve.MODEL_GROUPS[model])}
    # ---- hook: percentile of TTCC's early hold among the other videos' TTCC predictions
    # a clip shorter than 10 s has no R(10): use the holds the video does have (R3 and R5 required) and the same ones for every other video
    keys = [k for k in ("ttcc_R3", "ttcc_R5", "ttcc_R10") if tf is not None and math.isfinite(tf[k])]
    hold = lambda f: float(np.mean([f[k] for k in keys]))  # noqa: E731
    if tf is not None and len(reference_ttcc) >= MIN_REFERENCE and {"ttcc_R3", "ttcc_R5"} <= set(keys):
        refhold = np.array([hold(r["ttcc"]) for r in reference_ttcc if r["handle"] != target["handle"] and all(math.isfinite(r["ttcc"][k]) for k in keys)])
        hs = percentile(hold(tf), refhold)
        used = ", ".join("R(" + k[6:] + ")" for k in keys)
        out["hook_score"] = {"score": round(hs), "early_hold": round(hold(tf), 4), "reference_reels": len(refhold), "holds_used": [k[5:] for k in keys],
                             "meaning": f"percentile of TTCC's predicted early hold, mean of {used}, among the other videos' predictions (50 = typical)"
                                        + ("; this clip is shorter than 10 s, so R(10) is not part of it" if "ttcc_R10" not in keys else ""),
                             "caveat": "TTCC is a pretrained retention model; its predictions are unvalidated on these reels. Percentiles describe position, not quality."}
    else:
        if tf is None:
            why = "no TTCC hook prediction for this video"
        elif len(reference_ttcc) < MIN_REFERENCE:
            why = f"only {len(reference_ttcc)} other videos have TTCC predictions; {MIN_REFERENCE} are needed"
        else:
            why = "the video is too short for TTCC to predict R(3) and R(5)"
        out["hook_score"] = {"score": None, "reason": why + (f" ({tinfo['reason']})" if tf is None and tinfo else "")}
    out["popularity_score"] = _popularity(video_dir, target, ref, run, cfg, tribe_root)
    if tf is not None:
        h = tinfo["hook"]
        out["retention_pct"] = {f"{k}s": None if h.get(f"R{k}") is None else round(100 * h[f"R{k}"]) for k in (1, 3, 5, 10)}
        out["retention_note"] = "TTCC model prediction of the share of viewers still watching (viewers who start at 0); not audience data"
    else:
        out["retention_pct"] = None
    return out


def write_scores(video_dir: Path, scores: dict) -> Path:
    path = video_dir / SCORES_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scores, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path
