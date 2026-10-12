#!/usr/bin/env python3
"""Train a small head on SnapUGC (organic short videos) to predict real watch share (ECR), and test it honestly.

    .venv/bin/python scripts/train_snapugc_head.py

Inputs (already produced): artifacts/snapugc_feats/features/<id>/opening_multimodal.json (local visual/audio features of the first 5 s),
artifacts/snapugc_eval/features/<id>/ttcc_retention.json (TTCC native hook pass), data/snapugc/manifest.csv (ECR = share of viewers who
watched at least 5 s; rank-normalised, SnapUGC, Li et al. ECCV 2024). This is NOT fine-tuning of TTCC: it is a ridge regression on a handful of
features, validated by repeated 5-fold cross-validation, with a bootstrap interval and a permutation test. Feature sets are compared on the same
videos so a gain (or no gain) is visible: duration only | local features | TTCC only | local + TTCC.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hookline import views_eval as ve  # noqa: E402

DATA = PROJECT_ROOT / "data" / "snapugc"
LOCAL = ("cuts_per_s_0_5", "first_cut_s", "motion_mean_0_5", "brightness_mean_0_5", "rms_dbfs_0_5")
SETS = {"duration": ("log_duration",), "local": ("log_duration",) + LOCAL, "ttcc": ("ttcc_R3", "ttcc_R5", "ttcc_R10", "ttcc_drop"),
        "local+ttcc": ("log_duration",) + LOCAL + ("ttcc_R3", "ttcc_R5", "ttcc_R10", "ttcc_drop")}


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def load() -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    ids, cols, y = [], {k: [] for k in ("log_duration",) + LOCAL + ("ttcc_R3", "ttcc_R5", "ttcc_R10", "ttcc_drop")}, []
    for r in csv.DictReader((DATA / "manifest.csv").open()):
        vid = r["video_id"]
        d1 = PROJECT_ROOT / "artifacts" / "snapugc_feats" / "features" / vid
        f2 = PROJECT_ROOT / "artifacts" / "snapugc_eval" / "features" / vid / "ttcc_retention.json"
        if not (d1 / "opening_multimodal.json").is_file() or not f2.is_file() or float(r["duration_seconds"]) < 5:
            continue
        try:
            m, dur = ve.measured_features(d1)
            t = json.loads(f2.read_text())
            if t.get("status") != "success" or len(t["retention"]) < 6:
                continue
            rt = t["retention"]
            row = {"log_duration": math.log(float(r["duration_seconds"])), **{k: m[k] for k in LOCAL},
                   "ttcc_R3": rt[3], "ttcc_R5": rt[5], "ttcc_R10": rt[10] if len(rt) > 10 else float("nan"), "ttcc_drop": rt[0] - rt[1]}
        except (OSError, ValueError, KeyError):
            continue
        ids.append(vid)
        y.append(float(r["label_value"]))
        for k in cols:
            cols[k].append(row[k])
    return ids, {k: np.array(v, float) for k, v in cols.items()}, np.array(y, float)


def design(cols: dict[str, np.ndarray], names, idx) -> np.ndarray:
    X = np.column_stack([cols[k][idx] for k in names])
    return X


def fit_predict(Xtr, ytr, Xte, alpha=10.0):
    med = np.nanmedian(Xtr, axis=0)
    Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
    return B @ w + ytr.mean()


def oof(X, y, rng, reps=20, folds=5):
    P = np.zeros((reps, len(y)))
    for r in range(reps):
        order = rng.permutation(len(y))
        for k in range(folds):
            te = order[k::folds]
            tr = np.setdiff1d(order, te)
            P[r, te] = fit_predict(X[tr], y[tr], X[te])
    return P


def main() -> int:
    ids, cols, y = load()
    n = len(y)
    if n < 40:
        print(f"only {n} videos have both TTCC and local features; wait for the runs to finish")
        return 1
    ok = np.isfinite(cols["ttcc_R10"])
    print(f"{n} SnapUGC videos; ECR {y.min():.2f}-{y.max():.2f} (median {np.median(y):.2f}); {int(ok.sum())} have R(10)")
    rng = np.random.default_rng(0)
    out = {}
    idx = np.arange(n)
    for name, names in SETS.items():
        X = design(cols, names, idx)
        P = oof(X, y, rng)
        rhos = np.array([spearman(P[r], y) for r in range(P.shape[0])])
        mean_pred = P.mean(0)
        boots = []
        for _ in range(1000):
            s = rng.integers(0, n, n)
            v = spearman(mean_pred[s], y[s])
            if np.isfinite(v):
                boots.append(v)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        perm = []
        for _ in range(200):
            yp = rng.permutation(y)
            perm.append(np.mean([spearman(oof(X, yp, rng, reps=1)[0], yp)]))
        p = float((np.sum(np.array(perm) >= rhos.mean()) + 1) / (len(perm) + 1))
        out[name] = {"spearman": float(rhos.mean()), "ci95": [float(lo), float(hi)], "perm_p": p}
        print(f"{name:12s} out-of-fold Spearman with real ECR: {rhos.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  permutation p={p:.3f}")
    (PROJECT_ROOT / "artifacts" / "snapugc_head_results.json").write_text(json.dumps({"n": n, "results": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
