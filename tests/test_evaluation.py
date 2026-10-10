"""The evaluation harness, tested on SYNTHETIC data only (these are not results about any real model or video)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hookline.evaluation import (
    MIN_VIDEOS, check_alignment, collect, compare, kendall, ridge_oof, spearman, tribe_summary, ttcc_target,
)


def rows_from(y, ttcc, tribe, dur):
    return [{"video_id": f"v{i}", "label": float(y[i]), "ttcc": float(ttcc[i]), "tribe": list(tribe[i]), "duration": float(dur[i])}
            for i in range(len(y))]


def synthetic(n=120, tribe_matters=False, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    tribe = rng.normal(size=(n, 8))
    ttcc = latent + 0.6 * rng.normal(size=n)  # TTCC sees the latent signal with noise
    dur = rng.uniform(5, 60, n)
    y = latent + 0.5 * rng.normal(size=n)
    if tribe_matters:
        extra = rng.normal(size=n)
        tribe[:, 0] = extra + 0.2 * rng.normal(size=n)  # a TRIBE feature carries information TTCC does not
        y = y + 1.2 * extra
    return rows_from(y, ttcc, tribe, dur)


def test_metrics_match_hand_calculations() -> None:
    x = np.array([1, 2, 3, 4, 5.0])
    assert spearman(x, x) == 1.0 and spearman(x, -x) == -1.0 and kendall(x, x) == 1.0
    assert round(spearman([1, 1, 2, 3], [1, 2, 2, 3]), 4) == 0.8333  # tie-aware
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))  # constant input has no rank correlation


def test_refuses_to_compare_too_few_videos() -> None:
    out = compare(synthetic(n=MIN_VIDEOS - 1))
    assert out["verdict"].startswith("insufficient data") and "metrics" not in out
    assert compare([])["n_compared"] == 0


def test_no_leakage_a_shuffled_label_scores_near_zero() -> None:
    rows = synthetic(n=150)
    rng = np.random.default_rng(1)
    shuffled = rng.permutation([r["label"] for r in rows])
    X = np.array([r["tribe"] for r in rows])
    scores = [spearman(ridge_oof(X, shuffled, seed=s), shuffled) for s in range(5)]
    assert abs(float(np.mean(scores))) < 0.15  # a leaky pipeline would score far above this on random labels


def test_noise_features_do_not_beat_ttcc_alone() -> None:
    out = compare(synthetic(tribe_matters=False), n_boot=300)
    assert out["metrics"]["ttcc_ridge"]["spearman"] > 0.5 and out["metrics"]["ttcc_raw"]["spearman"] > 0.5
    assert out["metrics"]["tribe"]["spearman"] < 0.3  # noise features alone predict little
    assert out["verdict_tribe_adds_value_beyond_ttcc"].startswith("not shown")
    lo, hi = out["bootstrap_95ci"]["combined_vs_ttcc_ridge"]["diff"]
    assert lo <= 0 <= hi


def test_a_genuinely_informative_tribe_feature_is_detected() -> None:
    out = compare(synthetic(n=200, tribe_matters=True), n_boot=300)
    assert out["metrics"]["combined"]["spearman"] > out["metrics"]["ttcc_ridge"]["spearman"] + 0.05
    assert out["verdict_tribe_adds_value_beyond_ttcc"].startswith("shown")
    assert out["bootstrap_95ci"]["combined_vs_ttcc_ridge"]["diff"][0] > 0


def test_duration_confound_is_a_baseline() -> None:
    rows = synthetic(n=150)
    dur = np.array([r["duration"] for r in rows])
    for r, d in zip(rows, dur):
        r["label"] = -d / 60 + 0.05 * np.random.default_rng(int(d * 1000)).normal()  # label is almost pure duration
    out = compare(rows, n_boot=200)
    assert out["metrics"]["duration"]["spearman"] > 0.9  # the baseline exposes it, so models are judged against it


def test_tribe_summary_and_ttcc_target() -> None:
    preds = np.arange(60, dtype=float).reshape(6, 10)
    f = tribe_summary(preds, [0, 1, 2, 3, 4, 5], 6.0)
    assert len(f) == 8 and f[7] == 1.0 and f[0] == preds.mean()
    r = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
    assert ttcc_target(r, "ecr") == 0.3 and ttcc_target(r, "r15") is None and abs(ttcc_target(r, "mean_r") - np.mean(r)) < 1e-12


def make_artifacts(root: Path, vid: str, retention, with_tribe=True, duration=6.0, starts=None, seg=1.0):
    d = root / "features" / vid
    d.mkdir(parents=True)
    (d / "ttcc_retention.json").write_text(json.dumps({"status": "success", "video_id": vid, "retention": retention, "duration_s": duration,
        "timestamps_s": list(range(len(retention))), "partial_coverage": False}))
    if with_tribe:
        starts = starts if starts is not None else [float(i) for i in range(7)]
        (d / "tribe.json").write_text(json.dumps({"measured": True, "timestamps_s": starts, "segment_durations_s": [seg] * len(starts)}))
        np.save(d / "tribe_preds.npy", np.random.default_rng(0).normal(size=(len(starts), 20)).astype(np.float32))
    return d


def test_collect_counts_missing_outputs_instead_of_imputing(tmp_path) -> None:
    make_artifacts(tmp_path, "a", [1, .5, .4, .3, .2, .1, .05])
    make_artifacts(tmp_path, "b", [1, .5, .4, .3, .2, .1, .05], with_tribe=False)
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label"])
        w.writerows([["a", 0.4], ["b", 0.3], ["c", 0.2], ["d", "oops"]])
    rows, missing = collect(tmp_path, labels, "ecr")
    assert [r["video_id"] for r in rows] == ["a", "b", "c"] and rows[0]["tribe"] is not None and rows[1]["tribe"] is None
    assert missing["no_tribe"] == ["b", "c"] and missing["no_ttcc"] == ["c"] and missing["bad_label"] == ["d"]
    assert compare(rows)["n_compared"] == 1


def test_alignment_check_reports_grid_and_flags_mismatches(tmp_path) -> None:
    ok = make_artifacts(tmp_path / "ok", "v", [1, .5, .4, .3, .2, .1, .05])
    rep = check_alignment(ok)
    assert rep["problems"] == [] and rep["same_second_grid"] is True and rep["tribe_segment_s"] == [1.0] and "5.0" in rep["untested"]
    bad = make_artifacts(tmp_path / "bad", "v", [1, .5, .4, .3, .2, .1, .05], seg=2.0)
    assert any("not 1 s long" in p for p in check_alignment(bad)["problems"])
    shifted = make_artifacts(tmp_path / "shift", "v", [1, .5, .4, .3, .2, .1, .05], starts=[float(i) + 30 for i in range(7)])
    assert any("does not match the video duration" in p for p in check_alignment(shifted)["problems"])
    only = make_artifacts(tmp_path / "only", "v", [1, .5, .4], with_tribe=False)
    assert "both outputs are needed" in check_alignment(only)["problems"][0]
