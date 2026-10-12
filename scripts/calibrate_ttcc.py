#!/usr/bin/env python3
"""Can real retention curves make TTCC's early predictions more accurate? Per-second recalibration of R(1..10), validated on held-out ads.

    .venv/bin/python scripts/calibrate_ttcc.py predict   # TTCC native hook pass (first 10 s) on each ad with a real curve; uses your Space
    .venv/bin/python scripts/calibrate_ttcc.py fit       # cross-validated recalibration; writes artifacts/ttcc_calibration.json (local)

Data: ads with real per-second retention (liangyuch TTCC datasets, research-use licence; kept in artifacts/ttcc_eval/, never committed).
Model: for each second t, logit(real R(t)) = a_t + b_t * logit(TTCC R(t)), fitted by least squares (2 numbers per second). Rank order is unchanged when
b_t > 0; what changes is the spread and level. Validation: 5-fold cross-validation repeated, grouped by brand so one advertiser never sits in both
the fit and the test side; metric = mean squared error of R(t) against the real curve, versus raw TTCC and versus a constant (mean real curve of the
fit ads). Caveat: ads, not organic reels; the fit may not transfer, and some of these ads could be in TTCC's own training data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "artifacts" / "ttcc_eval"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_ttcc_curves import usable  # noqa: E402

SECONDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
EPS = 1e-3


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def predict() -> int:
    for ad in usable():
        if (ROOT / "hook" / "features" / ad / "ttcc_retention.json").is_file():
            continue
        cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/run_ttcc.py", "--input", str(ROOT / "videos" / f"{ad}.mp4"), "--artifacts", str(ROOT / "hook"),
               "--features", "skip", "--enable-ttcc", "--ttcc-remote", "--ttcc-allow-remote", "--hook-s", "10"]
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        ok = (ROOT / "hook" / "features" / ad / "ttcc_retention.json").is_file()
        print(f"{ad}: {'ok' if ok else 'FAILED ' + ((p.stdout or '') + (p.stderr or ''))[-120:].replace(chr(10), ' ')}", flush=True)
    return 0


def load() -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    truth = usable()
    gt = json.loads((ROOT / "ground_truth.json").read_text())
    ads, P, T, brands = [], [], [], []
    for ad, t in truth.items():
        f = ROOT / "hook" / "features" / ad / "ttcc_retention.json"
        if not f.is_file() or len(t) <= SECONDS[-1]:
            continue
        a = json.loads(f.read_text())
        if a.get("status") != "success" or len(a["retention"]) <= SECONDS[-1]:
            continue
        ads.append(ad)
        P.append([a["retention"][s] for s in SECONDS])
        T.append([t[s] for s in SECONDS])
        brands.append(str(gt[ad].get("brand") or ad))
    return ads, np.array(P), np.array(T), brands


def fit_params(P: np.ndarray, T: np.ndarray) -> list[tuple[float, float]]:
    out = []
    for j in range(P.shape[1]):
        x, y = logit(P[:, j]), logit(T[:, j])
        b, a = np.polyfit(x, y, 1) if np.std(x) > 0 else (1.0, 0.0)
        out.append((float(a), float(b)))
    return out


def apply(P: np.ndarray, params) -> np.ndarray:
    return np.column_stack([expit(a + b * logit(P[:, j])) for j, (a, b) in enumerate(params)])


def fit() -> int:
    ads, P, T, brands = load()
    n = len(ads)
    if n < 30:
        print(f"only {n} ads with predictions; run 'predict' first")
        return 1
    rng = np.random.default_rng(0)
    ub = sorted(set(brands))
    res = {"raw": [], "calibrated": [], "constant": []}
    per_t = {"raw": np.zeros(len(SECONDS)), "calibrated": np.zeros(len(SECONDS)), "constant": np.zeros(len(SECONDS))}
    reps = 20
    for _ in range(reps):
        order = rng.permutation(len(ub))
        fold_of = {ub[i]: k % 5 for k, i in enumerate(order)}
        fold = np.array([fold_of[b] for b in brands])
        pred_cal, pred_const = np.zeros_like(T), np.zeros_like(T)
        for k in range(5):
            tr, te = fold != k, fold == k
            if tr.sum() < 10 or te.sum() == 0:
                continue
            pred_cal[te] = apply(P[te], fit_params(P[tr], T[tr]))
            pred_const[te] = T[tr].mean(axis=0)
        for name, pr in (("raw", P), ("calibrated", pred_cal), ("constant", pred_const)):
            res[name].append(float(np.mean((pr - T) ** 2)))
            per_t[name] += np.mean((pr - T) ** 2, axis=0) / reps
    print(f"{n} ads with real curves and TTCC hook predictions ({len(ub)} brands); grouped 5-fold CV x{reps}")
    for name in ("constant", "raw", "calibrated"):
        a = np.array(res[name])
        print(f"mean squared error of R(1..10): {name:10s} {a.mean():.4f}  (CV spread {a.std():.4f})")
    for j, s in enumerate(SECONDS):
        if s in (1, 3, 5, 10):
            print(f"  R({s:2d}): constant {per_t['constant'][j]:.4f} | raw {per_t['raw'][j]:.4f} | calibrated {per_t['calibrated'][j]:.4f}")
    params = fit_params(P, T)
    print("fitted (a_t, b_t) at t=1,3,5,10:", [(s, round(params[SECONDS.index(s)][0], 2), round(params[SECONDS.index(s)][1], 2)) for s in (1, 3, 5, 10)])
    sp = lambda x, y: float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])  # noqa: E731
    print("Spearman(raw TTCC, real) at t=3,5,10:", [round(sp(P[:, SECONDS.index(s)], T[:, SECONDS.index(s)]), 2) for s in (3, 5, 10)], " (calibration with b>0 does not change rank)")
    out = {"n_ads": n, "n_brands": len(ub), "seconds": list(SECONDS), "params": params, "cv_mse": {k: float(np.mean(v)) for k, v in res.items()},
           "cv_mse_per_second": {k: [float(x) for x in v] for k, v in per_t.items()}, "model": "logit(real R(t)) = a_t + b_t * logit(TTCC R(t))",
           "caveat": "fitted on TikTok ads with real retention; may not transfer to organic reels; research-use data"}
    (PROJECT_ROOT / "artifacts" / "ttcc_calibration.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit({"predict": predict, "fit": fit}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__) or 2)())
