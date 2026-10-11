#!/usr/bin/env python3
"""Evaluate whether Hookline's features predict which reels beat their page's usual views (real Instagram views, 01-Hookline/*/metadata.csv).

    python scripts/eval_views.py                                   # measured-feature baselines (+ TTCC when hook predictions exist)
    python scripts/eval_views.py --ttcc-artifacts artifacts/hook   # TTCC hook predictions from scripts/run_ttcc.py --long-videos hook
    python scripts/eval_views.py --quick                           # 200 bootstrap / 100 permutations, for a fast look (not for reporting)

Writes a NEW timestamped folder under 01-Hookline/evaluation/ (earlier runs are never overwritten): metrics_summary.json,
baseline_comparison.csv, outlier_ranking.csv, bootstrap_confidence_intervals.json, permutation_test.json, prediction_results.csv,
run_metadata.json. Reads saved files only: no model, no Space, no upload, no cost. Views are never used as retention labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.views_eval import (  # noqa: E402
    DEFAULTS, add_labels, assemble, evaluate, load_dataset, new_run_dir, now_stamp, run_metadata, write_outputs,
)



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "01-Hookline", help="dataset root with face/, faceless/, outliers/")
    ap.add_argument("--features", type=Path, default=PROJECT_ROOT / "artifacts" / "company" / "features", help="measured features per video")
    ap.add_argument("--ttcc-artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "hook",
                    help="artifact root whose features/<id>/ttcc_retention.json are TTCC hook predictions")
    ap.add_argument("--tribe-artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "tribe",
                    help="artifact root whose features/<id>/tribe.json + tribe_preds.npy are TRIBE outputs (first 10 s are used)")
    ap.add_argument("--out", type=Path, default=None, help="output root (default <data>/evaluation)")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--folds", type=int, default=DEFAULTS["folds"])
    ap.add_argument("--repeats", type=int, default=DEFAULTS["repeats"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--bootstrap", type=int, default=DEFAULTS["n_boot"])
    ap.add_argument("--permutations", type=int, default=DEFAULTS["n_perm"])
    ap.add_argument("--quick", action="store_true", help="200 bootstrap resamples and 100 permutations")
    args = ap.parse_args(argv)
    cfg = {"seed": args.seed, "folds": args.folds, "repeats": args.repeats, "alpha": args.alpha,
           "n_boot": 200 if args.quick else args.bootstrap, "n_perm": 100 if args.quick else args.permutations}
    stamp = now_stamp()
    rows = load_dataset(args.data)
    medians = add_labels(rows)
    ttcc_root = args.ttcc_artifacts / "features" if args.ttcc_artifacts and (args.ttcc_artifacts / "features").is_dir() else None
    tribe_root = args.tribe_artifacts / "features" if args.tribe_artifacts and (args.tribe_artifacts / "features").is_dir() else None
    coverage = assemble(rows, args.features, ttcc_root, tribe_root)
    print(f"{len(rows)} reels ({sum(not r['is_outlier'] for r in rows)} regular, {sum(r['is_outlier'] for r in rows)} outliers); "
          f"TTCC hook predictions for {len(coverage['ttcc_available'])}" + ("" if ttcc_root else f" (no {args.ttcc_artifacts}/features)")
          + f"; TRIBE outputs for {len(coverage['tribe_available'])}" + ("" if tribe_root else f" (no {args.tribe_artifacts}/features)"))
    results = evaluate(rows, cfg)
    run_dir = new_run_dir(args.out or (args.data / "evaluation"), stamp)
    meta = run_metadata(cfg, args.data, args.features, ttcc_root, rows, stamp)
    meta["inputs"]["tribe_root"] = str(tribe_root) if tribe_root else None
    paths = write_outputs(run_dir, rows, medians, coverage, results, meta)
    (run_dir.parent / "LATEST").write_text(run_dir.name + "\n")
    print(f"\n{'model':<32}{'n':>4}  {'AUC':>6}  {'95% CI':>15}  {'Spearman':>9}  {'perm p(AUC)':>11}  status")
    for model, m in results["models"].items():
        c = (results["bootstrap_95ci"].get(model) or {}).get("auc")
        p = results["permutation"].get(model, {}).get("p_auc")
        sp = m["spearman"]
        print(f"{model:<32}{m['n']:>4}  {m['auc']:>6.3f}  {('[%.3f, %.3f]' % tuple(c)) if c else '':>15}  {('%.3f' % sp) if sp is not None else '-':>9}  "
              f"{('%.4f' % p) if p is not None else '-':>11}  {results['status'][model]}")
    for model, why in results["status"].items():
        if model not in results["models"]:
            print(f"{model:<32}  {why}")
    print("\noutliers (scored by models that never saw their creator): " + json.dumps(
        {m: {"auc_vs_regular": round(v["auc_outlier_vs_regular"], 3) if v["auc_outlier_vs_regular"] is not None else None,
             "median_rank": v["median_outlier_rank"], "of": v["n_ranked"]} for m, v in results["outliers"]["per_model"].items()}))
    print(f"\nwritten to {run_dir}: " + ", ".join(paths.values()))
    print("Verified = these metrics on real views. TTCC values are unvalidated model predictions used as features; views are not retention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
