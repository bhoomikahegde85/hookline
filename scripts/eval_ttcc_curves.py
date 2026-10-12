#!/usr/bin/env python3
"""Does the deployed TTCC pipeline match REAL per-second retention? Ads with ground-truth curves (liangyuch/ttcc-test-20), predicted on your Space.

    .venv/bin/python scripts/eval_ttcc_curves.py predict     # runs TTCC (upload-path settings: 10 s windows, whole video) on each ad; uses your Space
    .venv/bin/python scripts/eval_ttcc_curves.py score       # compares predictions with the real curves (local, no Space)

Inputs (made by hand from the dataset, research-use licence, never committed): artifacts/ttcc_eval/videos/<ad_id>.mp4 and ground_truth.json (percent curves).
Metrics: integrated Brier score (IBS, mean squared error of the curve) against the real curve, compared with a constant baseline (the mean real
curve of the OTHER ads, leave-one-out), and Spearman rank agreement of the hook values R(3), R(5), R(10) and the final value across ads.
Tiny sample: read the numbers as a check, not a benchmark.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "artifacts" / "ttcc_eval"
MAX_DURATION_S = 75.0


def usable() -> dict:
    gt = json.loads((ROOT / "ground_truth.json").read_text())
    out = {}
    for ad, v in gt.items():
        r = np.asarray(v["retention"], float)
        if not (5.0 <= v["duration"] <= MAX_DURATION_S) or len(r) < 5 or r[0] < 99 or r.max() <= 0 or abs(len(r) - (math.ceil(v["duration"]) + 1)) > 1:
            continue
        out[ad] = r / 100.0
    return out


def predict() -> int:
    for ad in usable():
        done = ROOT / "run" / "features" / ad / "ttcc_retention.json"
        if done.is_file():
            print(f"{ad}: cached")
            continue
        cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/run_ttcc.py", "--input", str(ROOT / "videos" / f"{ad}.mp4"), "--artifacts", str(ROOT / "run"),
               "--features", "skip", "--enable-ttcc", "--ttcc-remote", "--ttcc-allow-remote", "--hook-s", "10", "--long-videos", "windows", "--window-mode", "fixed",
               "--window-s", "10", "--stride-s", "5", "--warmup-s", "2.5"]
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        print(f"{ad}: {'ok' if done.is_file() else 'FAILED ' + ((p.stdout or '') + (p.stderr or ''))[-200:]}", flush=True)
    return 0


def at(r: np.ndarray, t: int) -> float:
    return float(r[t]) if t < len(r) else float("nan")


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def score() -> int:
    truth = usable()
    pred = {}
    for ad in truth:
        f = ROOT / "run" / "features" / ad / "ttcc_retention.json"
        if f.is_file():
            a = json.loads(f.read_text())
            if a.get("status") == "success":
                pred[ad] = np.asarray(a["retention"], float)
    ads = sorted(pred)
    if len(ads) < 5:
        print(f"only {len(ads)} predictions; run 'predict' first")
        return 1
    rows = []
    for ad in ads:
        t, p = truth[ad], pred[ad]
        n = min(len(t), len(p))
        others = [truth[o] for o in ads if o != ad]
        m = max(len(x) for x in others + [t])
        pad = lambda x: np.concatenate([x, np.full(m - len(x), x[-1])])  # noqa: E731  (a finished video keeps its last value)
        base = np.mean([pad(x) for x in others], axis=0)
        ibs_model = float(np.mean((p[:n] - t[:n]) ** 2))
        ibs_base = float(np.mean((base[:n] - t[:n]) ** 2))  # same seconds as the model, so the two are comparable
        rows.append({"ad": ad, "n": n, "ibs_model": ibs_model, "ibs_constant": ibs_base,
                     **{f"true_R{s}": at(t, s) for s in (3, 5, 10)}, **{f"pred_R{s}": at(p, s) for s in (3, 5, 10)}, "true_end": float(t[n - 1]), "pred_end": float(p[n - 1])})
    mi = np.mean([r["ibs_model"] for r in rows])
    bi = np.mean([r["ibs_constant"] for r in rows])
    wins = sum(r["ibs_model"] < r["ibs_constant"] for r in rows)
    print(f"{len(rows)} ads with real curves and TTCC predictions")
    print(f"mean IBS: TTCC {mi:.4f} | constant (mean of other ads) {bi:.4f} | TTCC better on {wins} of {len(rows)} ads")
    for s in (3, 5, 10):
        print(f"R({s}s): Spearman(pred, true) = {spearman([r[f'pred_R{s}'] for r in rows], [r[f'true_R{s}'] for r in rows]):+.2f}   "
              f"true range {np.nanmin([r[f'true_R{s}'] for r in rows]):.2f}-{np.nanmax([r[f'true_R{s}'] for r in rows]):.2f}, predicted range {np.nanmin([r[f'pred_R{s}'] for r in rows]):.2f}-{np.nanmax([r[f'pred_R{s}'] for r in rows]):.2f}")
    print(f"end of video: Spearman = {spearman([r['pred_end'] for r in rows], [r['true_end'] for r in rows]):+.2f}")
    (ROOT / "results.json").write_text(json.dumps({"rows": rows, "mean_ibs_model": mi, "mean_ibs_constant": bi}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit({"predict": predict, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__) or 2)())
