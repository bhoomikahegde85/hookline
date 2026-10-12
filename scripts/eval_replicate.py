#!/usr/bin/env python3
"""Does the Replicate TRIBE "virality" score separate reels that beat their page's usual views? Compared with the models in the latest views evaluation.

    python scripts/eval_replicate.py                       # reads artifacts/replicate/features/<id>/replicate.json (scripts/run_replicate_tribe.py)

The wrapper's score is a fixed formula with nothing fitted here, so its AUC is computed directly on the 100 regular reels (label: views above the
follower-band median, as in scripts/eval_views.py). The CI is a creator-clustered bootstrap and the p-value a permutation test that shuffles the label
within follower bands. The fitted models in the latest 01-Hookline/evaluation/<run> use cross-validation instead, so the comparison is indicative only.
Reads saved files only: no upload, no cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.views_eval import add_labels, auc, load_dataset  # noqa: E402


def spearman(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(np.asarray(x, float))).astype(float) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(score: np.ndarray, label: np.ndarray, over: np.ndarray, handles: list[str], bands: list[str], n_boot: int, n_perm: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs = auc(score, label)
    uniq = sorted(set(handles))
    members = {h: np.array([i for i, x in enumerate(handles) if x == h]) for h in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[h] for h in rng.choice(uniq, len(uniq))])
        a = auc(score[idx], label[idx])
        if a == a:
            boots.append(a)
    band_idx = [np.array([i for i, x in enumerate(bands) if x == b]) for b in sorted(set(bands))]
    hits = 0
    for _ in range(n_perm):
        yp = label.copy()
        for ix in band_idx:
            yp[ix] = rng.permutation(label[ix])
        hits += auc(score, yp) >= obs
    return {"n": int(len(score)), "auc": obs, "auc_95ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "perm_p_one_sided": (hits + 1) / (n_perm + 1), "spearman_vs_overperformance": spearman(score, over)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "replicate")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args(argv)
    rows = load_dataset(args.data)
    add_labels(rows)
    reg = [r for r in rows if not r["is_outlier"]]
    have, missing = [], []
    for r in reg:
        f = args.artifacts / "features" / r["video_id"] / "replicate.json"
        s = json.loads(f.read_text()).get("score") if f.exists() else None
        (have if s is not None else missing).append((r, s))
    if len(have) < 30:
        print(f"only {len(have)} of {len(reg)} regular reels have a score (need 30): run scripts/run_replicate_tribe.py first")
        return 1
    rs = [r for r, _ in have]
    score = np.array([s for _, s in have], float)
    label = np.array([r["above_band_median"] for r in rs])
    over = np.array([r["overperformance"] for r in rs])
    res = evaluate(score, label, over, [r["handle"] for r in rs], [r["band"] for r in rs], args.bootstrap, args.permutations)
    res["missing"] = len(missing)
    res["score_range"] = [float(score.min()), float(score.max())]
    runs = sorted(p for p in (args.data / "evaluation").iterdir() if (p / "metrics_summary.json").exists())
    ref = {}
    if runs:
        m = json.loads((runs[-1] / "metrics_summary.json").read_text()).get("models", {})
        ref = {"run": runs[-1].name, **{k: round(m[k]["auc"], 3) for k in ("ttcc_only", "tribe_only", "measured", "duration", "page_type") if k in m}}
    res["reference_models_cv_auc"] = ref
    out = args.artifacts / "replicate_eval.json"
    out.write_text(json.dumps(res, indent=2))
    ci = res["auc_95ci"]
    print(f"Replicate TRIBE score on {res['n']} regular reels: AUC {res['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  perm p {res['perm_p_one_sided']:.4f}  "
          f"Spearman {res['spearman_vs_overperformance']:+.3f}  (missing {res['missing']})")
    print(f"for comparison (cross-validated, latest run): {ref}")
    print("Not above chance unless the CI excludes 0.5 and p is small. Views measure reach, not retention. Written to " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
