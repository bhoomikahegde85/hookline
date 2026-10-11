"""Views-based evaluation: labels, grouped cross-validation, metrics, bootstrap, permutation, outliers and the written outputs.
Synthetic data with a known answer; the real run is scripts/eval_views.py."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from hookline import views_eval as ve
from hookline.views_eval import BANDS, MEASURED_FEATURES, TTCC_FEATURES

QUICK = {"seed": 0, "folds": 5, "repeats": 3, "alpha": 10.0, "n_boot": 200, "n_perm": 60}


def synthetic(n_handles=90, signal=2.0, seed=1, ttcc_signal=0.0, ttcc_frac=1.0, outliers=10):
    """Reels with a known relation between one measured feature and overperformance; some creators have two reels."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_handles):
        band = BANDS[h % 4]
        for k in range(2 if h % 9 == 0 else 1):
            x = rng.normal()
            t = rng.normal()
            views = int(math.exp(10 + h % 4 + signal * 0.5 * x + ttcc_signal * 0.5 * t + rng.normal()))
            m = {f: rng.normal() for f in MEASURED_FEATURES}
            m["first_word_s"] = x                      # the informative feature
            rows.append({"video_id": f"V{h:03d}_{k}", "handle": f"h{h}", "band": band, "views": max(views, 1), "group": "face", "is_outlier": False, "page_type": "face" if (h // 4) % 2 else "faceless",  # independent of the band (h % 4)
                         "measured": m, "duration_s": float(rng.uniform(6, 90)),
                         "ttcc": ({f: rng.normal() for f in TTCC_FEATURES} | {"ttcc_R3": t}) if rng.random() < ttcc_frac else None})
    for j in range(outliers):
        x = rng.normal() + 2.0
        rows.append({"video_id": f"OL{j:02d}", "handle": "h0" if j == 0 else f"o{j}", "band": BANDS[j % 4], "views": int(1e9), "group": "outliers",
                     "is_outlier": True, "page_type": "face", "measured": {f: rng.normal() for f in MEASURED_FEATURES} | {"first_word_s": x}, "duration_s": 30.0,
                     "ttcc": {f: rng.normal() for f in TTCC_FEATURES}})
    ve.add_labels(rows)
    return rows


def test_labels_use_regular_reels_only() -> None:
    rows = synthetic()
    medians = ve.add_labels(rows)
    for b in BANDS:
        reg = [r["views"] for r in rows if r["band"] == b and not r["is_outlier"]]
        assert medians[b] == float(np.median(reg))                          # outliers (1e9 views) never move a median
    r = rows[0]
    assert r["overperformance"] == pytest.approx(math.log(r["views"]) - math.log(medians[r["band"]]))
    assert all(r["above_band_median"] == int(r["overperformance"] > 0) for r in rows)


def test_folds_never_split_a_creator() -> None:
    groups = [f"h{i // 2}" for i in range(40)] + [f"s{i}" for i in range(30)]
    for seed in range(5):
        folds = ve.grouped_folds(groups, 5, np.random.default_rng(seed))
        assert sorted(np.concatenate(folds).tolist()) == list(range(70))
        for f in folds:
            inside = {groups[i] for i in f}
            assert all(groups[j] not in inside for g in folds if g is not f for j in g)


def test_auc_matches_the_mann_whitney_definition() -> None:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200).round(1), rng.integers(0, 2, 200)            # rounding makes ties
    assert ve.auc(s, y) == pytest.approx(roc_auc_score(y, s))
    assert ve.auc([1, 2, 3], [0, 0, 0]) != ve.auc([1, 2, 3], [0, 0, 0])      # undefined (nan) with one class


def test_fit_predict_learns_nothing_from_the_test_rows() -> None:
    rng = np.random.default_rng(0)
    Xtr, ytr = rng.normal(size=(50, 3)), rng.normal(size=50)
    Xte = rng.normal(size=(5, 3))
    a = ve.fit_predict(Xtr, ytr, Xte, 10.0)
    Xte2 = Xte.copy()
    Xte2[:, 0] = np.nan                                                      # imputed with the TRAINING median
    b = ve.fit_predict(Xtr, ytr, Xte2, 10.0)
    assert np.allclose(ve.fit_predict(Xtr, ytr, np.vstack([Xte, Xte * 100]), 10.0)[:5], a)  # test rows do not change each other
    assert not np.allclose(a, b) and np.isfinite(b).all()


def test_a_real_signal_is_found_and_significant() -> None:
    res = ve.evaluate(synthetic(signal=2.0), QUICK, log=lambda *_: None)
    m = res["models"]
    assert m["measured"]["auc"] > 0.7 and m["measured"]["spearman"] > 0.35
    assert res["permutation"]["measured"]["p_auc"] < 0.05
    lo, hi = res["bootstrap_95ci"]["measured"]["auc"]
    assert lo < m["measured"]["auc"] < hi and lo > 0.5
    assert res["bootstrap_95ci"]["measured_minus_constant"]["auc"][0] > 0
    assert m["constant"]["auc"] == 0.5 and m["constant"]["spearman"] is None


def test_no_signal_stays_near_chance_and_is_not_significant() -> None:
    res = ve.evaluate(synthetic(signal=0.0, seed=3), QUICK, log=lambda *_: None)
    lo, hi = res["bootstrap_95ci"]["measured"]["auc"]
    assert lo < 0.5 < hi and res["permutation"]["measured"]["p_auc"] > 0.05


def test_ttcc_models_need_enough_predictions_and_compare_on_the_same_reels() -> None:
    few = ve.evaluate(synthetic(ttcc_frac=0.2), QUICK, log=lambda *_: None)
    assert "measured_plus_ttcc" not in few["models"] and "not run" in few["status"]["measured_plus_ttcc"]
    part = ve.evaluate(synthetic(ttcc_frac=0.7, ttcc_signal=2.0), QUICK, log=lambda *_: None)
    n = part["models"]["measured_plus_ttcc"]["n"]
    assert part["models"]["measured_on_ttcc_subset"]["n"] == n == part["models"]["ttcc_only"]["n"] < part["models"]["measured"]["n"]
    assert part["models"]["ttcc_only"]["auc"] > 0.65 and "measured_plus_ttcc_minus_measured_on_ttcc_subset" in part["bootstrap_95ci"]


def test_outliers_are_scored_without_their_creator(monkeypatch) -> None:
    rows = synthetic()
    seen = []
    real = ve.fit_predict

    def spy(Xtr, ytr, Xte, alpha):
        seen.append(len(Xtr))
        return real(Xtr, ytr, Xte, alpha)

    res = ve.evaluate(rows, QUICK | {"n_perm": 0}, log=lambda *_: None)
    monkeypatch.setattr(ve, "fit_predict", spy)
    out = ve.score_outliers(rows, [r for r in rows if not r["is_outlier"]], [r for r in rows if r["is_outlier"]], res, QUICK)
    n_reg = sum(not r["is_outlier"] for r in rows)
    shared = sum(1 for r in rows if not r["is_outlier"] and r["handle"] == "h0")
    assert n_reg - shared in seen and n_reg in seen                         # OL00's creator (h0) is held out of its model
    pm = out["per_model"]["measured"]
    assert pm["n_ranked"] == len(rows) and pm["auc_outlier_vs_regular"] > 0.7  # outliers were built to score high
    assert set(pm["outlier_ranks"]) == {r["video_id"] for r in rows if r["is_outlier"]}


def _write_fake_data(tmp_path, rows):
    data, feats = tmp_path / "data", tmp_path / "features"
    for g in ("face", "faceless", "outliers"):
        (data / g).mkdir(parents=True)
        with (data / g / "metadata.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "instagram_handle", "follower_band", "views"])
            for r in rows:
                home = "outliers" if r["is_outlier"] else ("faceless" if r["video_id"].endswith("_1") else "face")
                if home == g:
                    w.writerow([r["video_id"], r["handle"], r["band"], r["views"]])
    for r in rows:
        d = feats / r["video_id"]
        d.mkdir(parents=True)
        m = r["measured"]
        (d / "opening_multimodal.json").write_text(json.dumps({
            "analysed_seconds": 5.0, "duration_s": r["duration_s"],
            "visual": {"stream_status": "available", "scene_cuts": abs(m["cuts_per_s_0_5"]) * 5, "first_scene_cut_s": None, "motion_mean": m["motion_mean_0_5"],
                       "brightness_mean": m["brightness_mean_0_5"]},
            "audio": {"stream_status": "available", "rms_dbfs": m["rms_dbfs_0_5"]},
            "speech": {"status": "available", "speech_detected": True, "first_word_s": m["first_word_s"], "words_per_s": m["words_per_s_0_5"]},
            "text": {"status": "unavailable"}}))
    return data, feats


def test_cli_writes_every_output_into_a_new_timestamped_folder(tmp_path) -> None:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("eval_views", Path(__file__).resolve().parents[1] / "scripts" / "eval_views.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = synthetic(signal=2.0)
    data, feats = _write_fake_data(tmp_path, rows)
    out = tmp_path / "evaluation"
    args = ["--data", str(data), "--features", str(feats), "--ttcc-artifacts", str(tmp_path / "none"), "--out", str(out), "--repeats", "2",
            "--bootstrap", "100", "--permutations", "20"]
    assert mod.main(args) == 0 and mod.main(args) == 0
    runs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(runs) == 2 and (out / "LATEST").read_text().strip() == runs[-1].name   # the earlier run is kept, not overwritten
    for name in ("metrics_summary.json", "baseline_comparison.csv", "outlier_ranking.csv", "bootstrap_confidence_intervals.json",
                 "permutation_test.json", "prediction_results.csv", "run_metadata.json"):
        assert (runs[0] / name).is_file(), name
    summary = json.loads((runs[0] / "metrics_summary.json").read_text())
    meta = json.loads((runs[0] / "run_metadata.json").read_text())
    assert summary["models"]["measured"]["auc"] > 0.6 and "Unvalidated" in summary["note"] and summary["label"]["n_regular"] == sum(not r["is_outlier"] for r in rows)
    assert meta["seed"] == 0 and meta["dataset"]["n_total"] == len(rows) and meta["config"]["n_perm"] == 20
    table = list(csv.DictReader((runs[0] / "baseline_comparison.csv").open()))
    assert [t["model"] for t in table][:5] == ["constant", "follower_band", "duration", "page_type", "measured"]
    status = {t["model"]: t["status"] for t in table}
    assert status["measured_plus_ttcc"].startswith("not run") and status["tribe_only"].startswith("not run")  # no TTCC / TRIBE output: said, not faked
    ranking = list(csv.DictReader((runs[0] / "outlier_ranking.csv").open()))
    assert len(ranking) == len(rows) and sum(r["is_outlier"] == "1" for r in ranking) == 10
    for p in runs[0].iterdir():
        text = p.read_text()
        assert "hf_" not in text and "AIza" not in text and "GEMINI_API_KEY" not in text and "HF_TOKEN" not in text


def test_page_type_is_a_metadata_feature_and_finds_its_signal() -> None:
    rows = synthetic(signal=0.0, seed=5)
    for r in rows:                                         # make face reels overperform: a known relation to recover
        if not r["is_outlier"] and r["page_type"] == "face":
            r["views"] = int(r["views"] * 4)
    ve.add_labels(rows)
    res = ve.evaluate(rows, QUICK, log=lambda *_: None)
    assert res["models"]["page_type"]["auc"] > 0.55 and res["models"]["page_type"]["groups"] == ["page_type"]
    assert ve.design(rows[:3], "page_type").shape == (3, 1) and ve.design(rows[:3], "measured_plus_page_type").shape == (3, len(MEASURED_FEATURES) + 1)
    assert "measured_plus_page_type_minus_measured" in res["bootstrap_95ci"]


def test_tribe_features_use_the_first_10_seconds_only(tmp_path) -> None:
    d = tmp_path / "v"
    d.mkdir()
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(30, 50))
    preds[15:] += 100.0                                   # a huge change after 10 s must not reach the features
    np.save(d / "tribe_preds.npy", preds)
    (d / "tribe.json").write_text(json.dumps({"measured": True, "timestamps_s": list(range(30))}))
    f, info = ve.tribe_features(d)
    assert set(f) == set(ve.TRIBE_FEATURE_COLS) and info["segments_used"] == 10 and abs(f["tribe_mean"]) < 1
    assert ve.tribe_features(tmp_path / "none")[0] is None
    (d / "tribe.json").write_text(json.dumps({"measured": False, "timestamps_s": list(range(30))}))
    assert ve.tribe_features(d)[0] is None


def test_ttcc_plus_tribe_is_compared_with_ttcc_alone_on_the_same_reels() -> None:
    rows = synthetic(ttcc_frac=1.0, ttcc_signal=2.0, seed=7)
    rng = np.random.default_rng(3)
    for i, r in enumerate(rows):                           # TRIBE for 70 % of reels, only a noise column carries information
        r["tribe"] = {k: float(rng.normal()) for k in ve.TRIBE_FEATURE_COLS} if rng.random() < 0.7 else None
    res = ve.evaluate(rows, QUICK, log=lambda *_: None)
    n = res["models"]["ttcc_plus_tribe"]["n"]
    assert n == res["models"]["tribe_only"]["n"] < res["models"]["ttcc_only"]["n"]
    assert res["models"]["ttcc_only_on_ttcc_tribe_subset"]["n"] == n                                  # the smaller model re-run on the same reels
    assert "ttcc_plus_tribe_minus_ttcc_only_on_ttcc_tribe_subset" in res["bootstrap_95ci"]
    gain = res["bootstrap_95ci"]["ttcc_plus_tribe_minus_ttcc_only_on_ttcc_tribe_subset"]["auc"]
    assert gain[0] <= 0 <= gain[1] or gain[1] < 0.15                                                  # pure-noise features show no clear gain
    few = ve.evaluate(synthetic(ttcc_frac=1.0), QUICK, log=lambda *_: None)
    assert "not run" in few["status"]["tribe_only"] and "ttcc_plus_tribe" not in few["models"]


def test_foldwise_metrics_are_not_biased_below_chance_for_a_mean_only_or_categorical_model() -> None:
    """Pooling held-out scores across folds gives a category-only model an AUC below its true value (the fold's own reels are left out of its
    training mean). Fold-wise metrics recover the real separation."""
    rng = np.random.default_rng(0)
    n = 100
    page = np.array([i % 2 for i in range(n)], float)
    y = rng.normal(size=n) + 1.2 * page                       # category 1 truly overperforms
    b = (y > np.median(y)).astype(int)
    groups = [f"g{i}" for i in range(n)]
    P, F = ve.cv_scores(page[:, None], y, groups, 5, 10, 10.0, 0)
    pooled = ve.auc(P.mean(0), b)
    fold = ve.foldwise_metrics(P, F, y, b)
    raw = ve.auc(page, b)
    assert raw > 0.65 and fold["auc"] > pooled and abs(fold["auc"] - raw) < 0.06 and fold["spearman"] > 0.25
    P0, F0 = ve.cv_scores(np.zeros((n, 0)), y, groups, 5, 10, 10.0, 0)
    assert ve.foldwise_metrics(np.zeros_like(P0), F0, y, b)["auc"] == 0.5          # a constant has no ranking information


# ---------------------------------------------------------------- 0-100 scores

def _score_fixture(tmp_path, ttcc_for=()):
    rows = synthetic(signal=2.0, seed=11, ttcc_frac=0.0)
    data, feats = _write_fake_data(tmp_path, rows)
    ttcc_root = tmp_path / "ttcc"
    for r in rows:
        if r["video_id"] in ttcc_for or (ttcc_for == "all" and not r["is_outlier"]):
            d = ttcc_root / r["video_id"]
            d.mkdir(parents=True)
            ret = [1.0] + [max(0.05, 0.9 - 0.04 * t - 0.02 * r["ttcc"]["ttcc_R3"]) for t in range(1, 31)] if r["ttcc"] else [1.0] + [max(0.05, 0.9 - 0.04 * t) for t in range(1, 31)]
            (d / "ttcc_retention.json").write_text(json.dumps({"status": "success", "retention": ret, "horizon_s": 30, "inference": {"window_end_s": 30.0}, "policy": "hook"}))
    return rows, data, feats, ttcc_root


def test_scores_are_positions_with_their_accuracy_and_never_a_verdict(tmp_path) -> None:
    from hookline.hook_score import compute_scores, write_scores
    rows, data, feats, ttcc_root = _score_fixture(tmp_path)
    target = next(r for r in rows if not r["is_outlier"] and 25 < r["duration_s"] < 70)  # inside the reference duration range, else the breakout score is withheld
    sc = compute_scores(feats / target["video_id"], data, feats, None, None)
    assert sc["status"] == "ok" and sc["in_reference_set"] is True and 0 <= sc["breakout_score"]["score"] <= 100
    far = max((r for r in rows if not r["is_outlier"]), key=lambda r: r["duration_s"])  # the longest reel sits beyond the 95th percentile: extrapolation, no score
    assert compute_scores(feats / far["video_id"], data, feats, None, None)["breakout_score"]["score"] is None
    assert sc["hook_score"]["score"] is None and "TTCC" in sc["hook_score"]["reason"] and sc["retention_pct"] is None
    assert sc["breakout_score"]["accuracy"]["evaluated"] is False and "no evaluation run" in sc["breakout_score"]["accuracy"]["verdict"]
    assert "not a statement that a video will" in sc["note"] and "viral" not in json.dumps(sc["breakout_score"]["meaning"]).lower()
    assert sc["breakout_score"]["reference_reels"] < sum(not r["is_outlier"] for r in rows)       # the video's own creator is held out of the model
    assert write_scores(feats / target["video_id"], sc).name == "hook_scores.json"
    inside = [r for r in rows if not r["is_outlier"] and 20 < r["duration_s"] < 75]
    best = max(inside, key=lambda r: r["measured"]["first_word_s"])   # the informative feature: a high value overperforms
    worst = min(inside, key=lambda r: r["measured"]["first_word_s"])
    hi, lo = (compute_scores(feats / x["video_id"], data, feats, None, None)["breakout_score"]["score"] for x in (best, worst))
    assert hi > lo


def test_accuracy_block_states_whether_the_model_beats_chance(tmp_path) -> None:
    from hookline.hook_score import accuracy_block
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "LATEST").write_text("run\n")
    summary = {"label": {"n_regular": 100}, "models": {"measured": {"n": 100, "auc": 0.6, "spearman": 0.1}, "page_type": {"n": 100, "auc": 0.64, "spearman": 0.2}},
               "bootstrap_95ci": {"measured": {"auc": [0.46, 0.68]}, "page_type": {"auc": [0.55, 0.73]}}, "permutation": {"measured": {"p_auc": 0.06}}}
    (run / "metrics_summary.json").write_text(json.dumps(summary))
    assert "not shown to beat chance" in accuracy_block(run, "measured")["verdict"]
    assert "better than chance" in accuracy_block(run, "page_type")["verdict"]
    assert accuracy_block(run, "ttcc_only")["evaluated"] is False and accuracy_block(None, "measured")["evaluated"] is False


def test_hook_score_needs_ttcc_on_the_reference_set_and_ranks_by_early_hold(tmp_path) -> None:
    from hookline.hook_score import compute_scores
    rows, data, feats, ttcc_root = _score_fixture(tmp_path)
    for r in rows:                                                                              # TTCC hook curves for all regular reels; a new video above them
        if not r["is_outlier"]:
            d = ttcc_root / r["video_id"]
            d.mkdir(parents=True, exist_ok=True)
            h = 0.5 + 0.2 * (hash(r["video_id"]) % 100) / 100
            (d / "ttcc_retention.json").write_text(json.dumps({"status": "success", "retention": [1.0] + [h * 0.97 ** t for t in range(1, 31)], "horizon_s": 30,
                                                               "inference": {"window_end_s": 30.0}, "policy": "hook"}))
    new = tmp_path / "newfeat" / "NEW"
    new.mkdir(parents=True)
    src = feats / "V001_0"
    (new / "opening_multimodal.json").write_text((src / "opening_multimodal.json").read_text())
    for label, h, expect in (("high", 1.0, lambda v: v >= 95), ("low", 0.4, lambda v: v <= 5)):
        (new / "ttcc_retention.json").write_text(json.dumps({"status": "success", "retention": [1.0] + [h * 0.97 ** t for t in range(1, 31)], "horizon_s": 30,
                                                             "inference": {"window_end_s": 30.0}, "policy": "hook"}))
        sc = compute_scores(new, data, feats, ttcc_root, None)
        assert sc["status"] == "ok" and sc["in_reference_set"] is False and expect(sc["hook_score"]["score"]), (label, sc["hook_score"])
        assert sc["retention_pct"]["5s"] == round(100 * h * 0.97 ** 5) and sc["breakout_score"]["features_used"] == ["measured", "ttcc"]
    stitched = json.loads((new / "ttcc_retention.json").read_text()) | {"stitched": True}
    (new / "ttcc_retention.json").write_text(json.dumps(stitched))                              # a stitched curve is not a hook prediction: not used
    assert compute_scores(new, data, feats, ttcc_root, None)["hook_score"]["score"] is None
    assert compute_scores(tmp_path / "empty", data, feats, None, None)["status"] == "unavailable"


def test_scores_never_use_account_data(tmp_path) -> None:
    """The product judges the video, not the account: page type, follower band and views are not features of any score."""
    from hookline.hook_score import compute_scores
    rows, data, feats, ttcc_root = _score_fixture(tmp_path)
    sc = compute_scores(feats / "V010_0", data, feats, None, None)
    assert sc["breakout_score"]["features_used"] == ["measured"]
    text = json.dumps(sc).lower()
    assert "page_type" not in text and "faceless" not in text and not {"band", "page_type"} & set(sc["breakout_score"]["features_used"])
    assert "follower band's median" in text                  # the follower band appears only as the definition of the LABEL, never as a feature
    import inspect
    assert "page_type" not in inspect.signature(compute_scores).parameters
