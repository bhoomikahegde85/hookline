"""Compare TTCC, TRIBE-derived features and both combined against real audience-retention labels.

TTCC predicts a retention curve R(t). TRIBE v2 predicts brain responses (fMRI) and is NOT a retention model, so its
output can only help as *input features to a model fitted on retention labels*. This module measures whether it does.

Configurations, all scored on the same videos with out-of-fold predictions (nothing is fitted on the videos it is scored on):
- ``constant``       no information (sanity floor)
- ``duration``       ridge on video duration only (a confound check: do the models beat "longer videos lose more viewers"?)
- ``ttcc_raw``       TTCC's own prediction of the target, no fitting (R(5) for ECR / r5, R(15) for r15, mean R for mean_r)
- ``ttcc_ridge``     ridge on TTCC's prediction (the fair comparator for the combined model: same fitting procedure)
- ``tribe``          ridge on TRIBE summary features only
- ``combined``       ridge on TTCC's prediction + TRIBE summary features

"Does TRIBE add value beyond TTCC" is the paired difference ``combined - ttcc_ridge`` with a bootstrap interval; if the interval
contains 0 the answer is "not shown", never "no effect". Rank metrics only (Spearman, Kendall tau-b): the public SnapUGC ECR is
rank-normalised, and absolute error would be meaningless there. With fewer than ``MIN_VIDEOS`` labelled videos nothing is compared.
The TRIBE summary features are a design choice of this module, themselves untested; a better feature set could change the answer.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MIN_VIDEOS = 30
TARGETS = {  # target name -> (index into the retention list, or "mean")
    "ecr": 5,   # SnapUGC ECR: share of viewers who watched at least 5 s, which is what R(5) predicts
    "r5": 5,
    "r15": 15,
    "mean_r": "mean",
}
TRIBE_FEATURES = ["mean", "std_over_time", "mean_abs_step_of_mean", "mean_l2_step", "early_mean_0_5s", "late_mean_after_5s", "peak_mean", "segments_per_s"]


# ---------------------------------------------------------------- metrics

def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    first = np.cumsum(cnt) - cnt
    return (first + (cnt - 1) / 2.0)[inv]


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = _rank(x) - _rank(x).mean(), _rank(y) - _rank(y).mean()
    d = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def kendall(x, y) -> float:
    """Kendall tau-b, O(n^2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan")
    dx, dy = np.sign(x[:, None] - x[None, :]), np.sign(y[:, None] - y[None, :])
    num = float((dx * dy).sum())
    den = math.sqrt(float((dx != 0).sum()) * float((dy != 0).sum()))
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------- out-of-fold ridge (standardisation fitted inside each fold)

def ridge_oof(X: np.ndarray, y: np.ndarray, folds: int = 5, repeats: int = 10, alpha: float = 10.0, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions averaged over repeated K-fold splits. Features are standardised using the training fold only."""
    n = len(y)
    X = np.asarray(X, float).reshape(n, -1)
    acc, cnt = np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        perm = rng.permutation(n)
        for k in range(folds):
            test = perm[k::folds]
            train = np.setdiff1d(perm, test)
            mu, sd = X[train].mean(0), X[train].std(0)
            sd[sd == 0] = 1.0
            Z = (X[train] - mu) / sd
            ym = y[train].mean()
            w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y[train] - ym))
            acc[test] += ((X[test] - mu) / sd) @ w + ym
            cnt[test] += 1
    return acc / np.maximum(cnt, 1)


# ---------------------------------------------------------------- features from saved artifacts

def tribe_summary(preds: np.ndarray, starts: list[float], duration: float | None) -> list[float]:
    """Eight dimension-free summaries of a (n_segments, n_vertices) TRIBE prediction. Descriptive, not a retention score."""
    g = preds.mean(1)
    step = np.diff(preds, axis=0)
    t = np.asarray(starts, float)
    early, late = g[t < 5.0], g[t >= 5.0]
    return [float(g.mean()), float(g.std()), float(np.abs(np.diff(g)).mean()) if len(g) > 1 else 0.0,
            float(np.linalg.norm(step, axis=1).mean()) if len(step) else 0.0,
            float(early.mean()) if len(early) else float("nan"), float(late.mean()) if len(late) else float("nan"),
            float(g.max()), len(g) / duration if duration else float("nan")]


def ttcc_target(retention: list[float], target: str) -> float | None:
    spec = TARGETS[target]
    if spec == "mean":
        return float(np.mean(retention))
    return float(retention[spec]) if len(retention) > spec else None


def check_alignment(video_dir: Path, tolerance_s: float = 1.0) -> dict:
    """Do TTCC's and TRIBE's outputs describe the same timeline? Reports facts; the TRIBE lag question stays open."""
    ttcc = json.loads((video_dir / "ttcc_retention.json").read_text()) if (video_dir / "ttcc_retention.json").is_file() else None
    tribe = json.loads((video_dir / "tribe.json").read_text()) if (video_dir / "tribe.json").is_file() else None
    out: dict[str, Any] = {"ttcc": bool(ttcc), "tribe": bool(tribe and tribe.get("measured")), "problems": []}
    if not (out["ttcc"] and out["tribe"]):
        out["problems"].append("both outputs are needed to check alignment")
        return out
    ts, starts, durs = ttcc["timestamps_s"], tribe["timestamps_s"], tribe["segment_durations_s"]
    dur = ttcc.get("duration_s")
    out.update({"ttcc_grid": [ts[0], ts[-1], 1.0], "tribe_first_start_s": starts[0], "tribe_last_end_s": round(starts[-1] + durs[-1], 3),
                "tribe_segment_s": sorted(set(durs)), "video_duration_s": dur, "tribe_rows": len(starts), "ttcc_points": len(ts)})
    if any(abs(d - 1.0) > 1e-6 for d in durs):
        out["problems"].append(f"TRIBE segments are not 1 s long: {sorted(set(durs))}")
    if any(b - a <= 0 for a, b in zip(starts, starts[1:])):
        out["problems"].append("TRIBE segment starts are not increasing")
    if abs(starts[0]) > tolerance_s or (dur and abs(starts[-1] + durs[-1] - dur) > tolerance_s + 1.0):
        out["problems"].append("TRIBE's time span does not match the video duration")
    if dur and abs(ts[-1] - round(dur)) > tolerance_s and not ttcc.get("partial_coverage"):
        out["problems"].append("TTCC's horizon does not match the video duration")
    out["same_second_grid"] = starts[:len(ts)] == [float(t) for t in ts[:len(starts)]] if not out["problems"] else False
    out["untested"] = ("TRIBE trains with a 5 s hemodynamic offset (data.neuro.offset = 5.0): whether a row stamped t describes the stimulus "
                       "at t, or earlier, is not verified. Hookline stores timestamps as returned and does not shift them.")
    return out


def collect(root: Path, labels_csv: Path, target: str, id_col: str = "video_id", label_col: str = "label") -> tuple[list[dict], dict]:
    """Join labels with saved outputs. Videos missing an output are counted, never imputed."""
    rows, missing = [], {"no_ttcc": [], "no_tribe": [], "bad_label": [], "target_out_of_range": []}
    with labels_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            vid = r[id_col]
            try:
                label = float(r[label_col])
            except (KeyError, ValueError):
                missing["bad_label"].append(vid)
                continue
            d = root / "features" / vid
            ttcc_p, tribe_p, npy = d / "ttcc_retention.json", d / "tribe.json", d / "tribe_preds.npy"
            ttcc = json.loads(ttcc_p.read_text()) if ttcc_p.is_file() else None
            tribe = json.loads(tribe_p.read_text()) if tribe_p.is_file() else None
            if not ttcc or ttcc.get("status") != "success":
                missing["no_ttcc"].append(vid)
            if not (tribe and tribe.get("measured") and npy.is_file()):
                missing["no_tribe"].append(vid)
            pred = ttcc_target(ttcc["retention"], target) if ttcc and ttcc.get("status") == "success" else None
            if ttcc and ttcc.get("status") == "success" and pred is None:
                missing["target_out_of_range"].append(vid)
            feats = tribe_summary(np.load(npy), tribe["timestamps_s"], ttcc.get("duration_s") if ttcc else None) \
                if tribe and tribe.get("measured") and npy.is_file() and ttcc else None
            rows.append({"video_id": vid, "label": label, "ttcc": pred, "tribe": feats, "duration": (ttcc or {}).get("duration_s")})
    return rows, missing


# ---------------------------------------------------------------- the comparison

def _boot(y, a, b, n_boot: int, seed: int) -> dict:
    """Bootstrap over videos: Spearman of each prediction and their difference (a - b)."""
    rng = np.random.default_rng(seed)
    n, ra, rb, rd = len(y), [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        sa, sb = spearman(a[i], y[i]), spearman(b[i], y[i])
        if not (math.isnan(sa) or math.isnan(sb)):
            ra.append(sa), rb.append(sb), rd.append(sa - sb)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None  # noqa: E731
    return {"a": q(ra), "b": q(rb), "diff": q(rd)}


def compare(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Score the configurations on the videos that have every output. Returns a plain dict (JSON-serialisable)."""
    use = [r for r in rows if r["ttcc"] is not None and r["tribe"] is not None and r["duration"] is not None
           and not any(math.isnan(v) for v in r["tribe"])]
    out: dict[str, Any] = {"n_labelled": len(rows), "n_compared": len(use), "min_videos": MIN_VIDEOS}
    if len(use) < MIN_VIDEOS:
        out["verdict"] = (f"insufficient data: {len(use)} videos have the label, a TTCC curve and a TRIBE output; at least {MIN_VIDEOS} are needed "
                          "before any comparison, and many more to detect a small gain")
        return out
    y = np.array([r["label"] for r in use])
    ttcc = np.array([r["ttcc"] for r in use])
    tribe = np.array([r["tribe"] for r in use])
    dur = np.array([r["duration"] for r in use])
    preds = {"constant": np.zeros(len(y)), "duration": ridge_oof(dur[:, None], y), "ttcc_raw": ttcc,
             "ttcc_ridge": ridge_oof(ttcc[:, None], y), "tribe": ridge_oof(tribe, y), "combined": ridge_oof(np.column_stack([ttcc, tribe]), y)}
    out["metrics"] = {k: {"spearman": round(spearman(p, y), 3), "kendall": round(kendall(p, y), 3)} for k, p in preds.items()}
    out["bootstrap_95ci"] = {
        "ttcc_ridge_vs_duration": _boot(y, preds["ttcc_ridge"], preds["duration"], n_boot, seed),
        "tribe_vs_duration": _boot(y, preds["tribe"], preds["duration"], n_boot, seed),
        "combined_vs_ttcc_ridge": _boot(y, preds["combined"], preds["ttcc_ridge"], n_boot, seed)}
    d = out["bootstrap_95ci"]["combined_vs_ttcc_ridge"]["diff"]
    out["verdict_tribe_adds_value_beyond_ttcc"] = (
        "not shown: the 95% interval for the Spearman gain includes 0 (this does not prove TRIBE adds nothing)" if d and d[0] <= 0 <= d[1]
        else "shown on these videos: the whole 95% interval is above 0" if d and d[0] > 0 else "TRIBE made it worse: the whole interval is below 0")
    out["note"] = ("Out-of-fold ridge, repeated 5-fold, standardisation fitted in-fold. Rank metrics only. TRIBE summary features are a design choice "
                   "of this module. Labels' domain may differ from the videos TTCC was trained on.")
    return out
