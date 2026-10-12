#!/usr/bin/env python3
"""Does TTCC's predicted hold rank real videos by their real 5-second watch share (SnapUGC ECR)? Predictions run on your Space.

    .venv/bin/python scripts/eval_snapugc.py predict    # TTCC native hook pass (first 10 s) per video, on your Space
    .venv/bin/python scripts/eval_snapugc.py score      # Spearman of R(3), R(5), R(10) vs ECR with a bootstrap interval (local)

ECR = share of viewers who watched at least 5 s (rank-normalised, SnapUGC, Li et al. ECCV 2024): the closest thing here to real retention. The hosted
videos cover only ECR 0 to about 0.52, so the range is narrow. Videos are research-use only: stay inside data/ and artifacts/ (never committed).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "snapugc"
OUT = PROJECT_ROOT / "artifacts" / "snapugc_eval"


def rows() -> list[dict]:
    return [r for r in csv.DictReader((DATA / "manifest.csv").open()) if r["label_name"] == "ECR" and float(r["duration_seconds"]) >= 5.0]


def predict() -> int:
    for r in rows():
        vid = r["video_id"]
        if (OUT / "features" / vid / "ttcc_retention.json").is_file():
            continue
        cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/run_ttcc.py", "--input", str(DATA / r["file_path"]), "--artifacts", str(OUT),
               "--features", "skip", "--enable-ttcc", "--ttcc-remote", "--ttcc-allow-remote", "--hook-s", "10"]
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        ok = (OUT / "features" / vid / "ttcc_retention.json").is_file()
        print(f"{vid}: {'ok' if ok else 'FAILED ' + ((p.stdout or '') + (p.stderr or ''))[-160:].replace(chr(10), ' ')}", flush=True)
    return 0


def rank(a):
    return np.argsort(np.argsort(a))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def score() -> int:
    data = []
    for r in rows():
        f = OUT / "features" / r["video_id"] / "ttcc_retention.json"
        if not f.is_file():
            continue
        a = json.loads(f.read_text())
        if a.get("status") != "success" or len(a["retention"]) < 6:
            continue
        rt = a["retention"]
        data.append({"id": r["video_id"], "ecr": float(r["label_value"]), "dur": float(r["duration_seconds"]), "R3": rt[3], "R5": rt[5], "R10": rt[10] if len(rt) > 10 else float("nan")})
    if len(data) < 20:
        print(f"only {len(data)} predictions so far; run 'predict' first")
        return 1
    y = np.array([d["ecr"] for d in data])
    rng = np.random.default_rng(0)
    print(f"{len(data)} videos; ECR range {y.min():.2f}-{y.max():.2f}, median {np.median(y):.2f}")
    res = {}
    for k in ("R3", "R5", "R10"):
        x = np.array([d[k] for d in data])
        ok = np.isfinite(x)
        rho = spearman(x[ok], y[ok])
        boots = []
        idx = np.where(ok)[0]
        for _ in range(2000):
            s = rng.choice(idx, len(idx))
            v = spearman(x[s], y[s])
            if np.isfinite(v):
                boots.append(v)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[k] = {"n": int(ok.sum()), "spearman": rho, "ci95": [float(lo), float(hi)]}
        print(f"predicted {k} vs ECR: Spearman {rho:+.2f}  95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]  n={int(ok.sum())}")
    dur = spearman([d["dur"] for d in data], y)
    print(f"for comparison, video duration vs ECR: Spearman {dur:+.2f}")
    (OUT / "results.json").write_text(json.dumps({"n": len(data), "results": res, "duration_vs_ecr": dur}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit({"predict": predict, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__) or 2)())
