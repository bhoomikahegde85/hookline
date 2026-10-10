"""Does a video beat its page's usual views? Evaluation against real Instagram views (``01-Hookline/*/metadata.csv``).

Labels (views are NOT retention labels and are never used as such):
- ``overperformance`` = log(views) - log(median views of the reel's follower band), band medians taken over the 100 regular reels
  (``face/`` and ``faceless/``); the 10 ``outliers/`` reels were selected for unusually high views, so they never set a median and are
  never used for training. They are scored separately by models fitted on the regular reels.
- ``above_band_median`` (primary binary label) = overperformance > 0.

Models, all fitted the same way (ridge regression on the continuous label, alpha fixed in advance, features standardised and missing values
imputed with the training fold's median, inside each fold) and scored out-of-fold with cross-validation grouped by creator handle (a creator's
reels are never split between training and test):
- ``constant``            no information (AUC 0.5 by definition)
- ``follower_band``       one-hot follower band (the label is band-centred, so this should be near chance; a leakage check)
- ``duration``            log duration
- ``measured``            a small fixed set of measured opening features (first 5 s), chosen before looking at any result
- ``measured_plus_ttcc``  the same plus TTCC's hook predictions R(3), R(5), R(10) and the steepest early drop (unvalidated model outputs)
- ``ttcc_only``           TTCC's hook predictions alone (extra row, for context)

Metrics: AUC of the predicted score for ``above_band_median``, Spearman with ``overperformance``, 95% bootstrap intervals (resampling
creators), paired bootstrap differences between models, and permutation tests that re-run the whole cross-validation on shuffled labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import spearman

REGULAR_GROUPS = ("face", "faceless")
OUTLIER_GROUP = "outliers"
BANDS = ("10k-50k", "50k-200k", "200k-1M", "1M+")
MEASURED_FEATURES = ("cuts_per_s_0_5", "first_cut_s", "motion_mean_0_5", "brightness_mean_0_5", "rms_dbfs_0_5", "first_word_s",
                     "words_per_s_0_5", "first_text_s")
TTCC_FEATURES = ("ttcc_R3", "ttcc_R5", "ttcc_R10", "ttcc_steepest_early_drop")
MODELS = ("constant", "follower_band", "duration", "measured", "measured_plus_ttcc", "ttcc_only")
DEFAULTS = {"seed": 0, "folds": 5, "repeats": 20, "alpha": 10.0, "n_boot": 2000, "n_perm": 1000}
MIN_TTCC_VIDEOS = 30
VERIFIED_NOTE = ("Verified: the metrics in this file were computed by this run on real Instagram views of these reels (held-out, grouped by "
                 "creator). Unvalidated: TTCC's R(t) values used as features are model predictions of retention, never measured retention; "
                 "views measure reach, which also depends on distribution, timing and trends that are not in the video.")


# ---------------------------------------------------------------- data

def load_dataset(data_root: Path) -> list[dict]:
    rows = []
    for group in (*REGULAR_GROUPS, OUTLIER_GROUP):
        meta = data_root / group / "metadata.csv"
        with meta.open(newline="") as f:
            for r in csv.DictReader(f):
                rows.append({"video_id": r["video_id"], "handle": r["instagram_handle"], "band": r["follower_band"], "views": int(r["views"]),
                             "group": group, "is_outlier": group == OUTLIER_GROUP, "video_path": str(data_root / group / "videos" / f"{r['video_id']}.mp4")})
    unknown = sorted({r["band"] for r in rows} - set(BANDS))
    if unknown:
        raise ValueError(f"unknown follower band(s) {unknown}")
    return rows


def add_labels(rows: list[dict]) -> dict[str, float]:
    """Band medians over the regular reels; adds ``overperformance`` and ``above_band_median`` to every row. Returns the medians."""
    medians = {}
    for band in BANDS:
        v = [r["views"] for r in rows if r["band"] == band and not r["is_outlier"]]
        if not v:
            raise ValueError(f"no regular reels in band {band}: cannot define its median")
        medians[band] = float(np.median(v))
    for r in rows:
        r["overperformance"] = math.log(r["views"]) - math.log(medians[r["band"]])
        r["above_band_median"] = int(r["overperformance"] > 0)
    return medians


def _num(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else float("nan")


def measured_features(d: Path) -> tuple[dict, float]:
    """The fixed opening features (first 5 s) and the duration, from ``opening_multimodal.json``. Absent events become the window length
    (no cut / word / text seen in the window), missing streams NaN (imputed in-fold)."""
    o = json.loads((d / "opening_multimodal.json").read_text())
    win = _num(o.get("analysed_seconds")) or 5.0
    vis, aud, sp, tx = o.get("visual") or {}, o.get("audio") or {}, o.get("speech") or {}, o.get("text") or {}
    vis_ok, aud_ok = vis.get("stream_status") == "available", aud.get("stream_status") == "available"
    sp_ok = sp.get("status") in ("available", "uncertain")
    tx_ok = tx.get("status") in ("available", "uncertain")
    f = {
        "cuts_per_s_0_5": _num(vis.get("scene_cuts")) / win if vis_ok else float("nan"),
        "first_cut_s": (_num(vis.get("first_scene_cut_s")) if vis.get("first_scene_cut_s") is not None else win) if vis_ok else float("nan"),
        "motion_mean_0_5": _num(vis.get("motion_mean")) if vis_ok else float("nan"),
        "brightness_mean_0_5": _num(vis.get("brightness_mean")) if vis_ok else float("nan"),
        "rms_dbfs_0_5": _num(aud.get("rms_dbfs")) if aud_ok else float("nan"),
        "first_word_s": (_num(sp.get("first_word_s")) if sp.get("first_word_s") is not None else win) if sp_ok else float("nan"),
        "words_per_s_0_5": (_num(sp.get("words_per_s")) if sp.get("speech_detected") else 0.0) if sp_ok else float("nan"),
        "first_text_s": (_num(tx.get("first_text_s")) if tx.get("first_text_s") is not None else win) if tx_ok else float("nan"),
    }
    return f, _num(o.get("duration_s"))


def ttcc_features(d: Path) -> tuple[dict | None, dict | None]:
    """TTCC hook features from a single-pass, start-anchored prediction (``ttcc_retention.json``); (None, why) otherwise. Stitched curves are refused."""
    from .ttcc import hook_metrics

    p = d / "ttcc_retention.json"
    if not p.is_file():
        return None, {"reason": "no TTCC prediction"}
    art = json.loads(p.read_text())
    if art.get("status") != "success":
        return None, {"reason": "TTCC prediction not successful"}
    if art.get("stitched"):
        return None, {"reason": "stitched windowed curve: not a start-anchored hook prediction"}
    shown = (art.get("inference") or {}).get("window_end_s")
    h = art.get("hook") or hook_metrics(art["retention"], shown)
    info = {"prompt_T": art.get("horizon_s"), "seconds_shown": h.get("seconds_shown"), "policy": art.get("policy"), "cache_key": art.get("cache_key")}
    if any(d_ for d_ in (art.get("inference") or {}).get("deviations_from_upstream_eval") or [] if "sensitivity run" in d_):
        return None, {"reason": "sensitivity run (model told a different length than it saw): not used for evaluation"} | info
    steep = h.get("steepest_early_drop") or {}
    f = {"ttcc_R3": _num(h.get("R3")), "ttcc_R5": _num(h.get("R5")), "ttcc_R10": _num(h.get("R10")), "ttcc_steepest_early_drop": _num(steep.get("drop"))}
    return f, info


def assemble(rows: list[dict], features_root: Path, ttcc_root: Path | None) -> dict:
    """Attach features. Returns coverage facts; nothing is invented for a missing file (the row is marked and imputed in-fold, or excluded)."""
    cov = {"measured_missing": [], "ttcc_available": [], "ttcc_missing": {}, "ttcc_configs": {}}
    for r in rows:
        d = features_root / r["video_id"]
        try:
            r["measured"], r["duration_s"] = measured_features(d)
        except (OSError, ValueError, KeyError) as exc:
            r["measured"], r["duration_s"] = {k: float("nan") for k in MEASURED_FEATURES}, float("nan")
            cov["measured_missing"].append(f"{r['video_id']}: {type(exc).__name__}")
        r["ttcc"] = None
        if ttcc_root is not None:
            f, info = ttcc_features(ttcc_root / r["video_id"])
            if f is None:
                cov["ttcc_missing"][r["video_id"]] = info["reason"]
            else:
                r["ttcc"] = f
                cov["ttcc_available"].append(r["video_id"])
                key = f"prompt_T={info['prompt_T']},shown={info['seconds_shown']}"
                cov["ttcc_configs"][key] = cov["ttcc_configs"].get(key, 0) + 1
    return cov


def design(rows: list[dict], model: str) -> np.ndarray:
    n = len(rows)
    if model == "constant":
        return np.zeros((n, 0))
    if model == "follower_band":
        return np.array([[1.0 if r["band"] == b else 0.0 for b in BANDS] for r in rows])
    if model == "duration":
        return np.array([[math.log(r["duration_s"]) if r["duration_s"] and r["duration_s"] > 0 else float("nan")] for r in rows])
    cols = []
    if model in ("measured", "measured_plus_ttcc"):
        cols.append(np.array([[r["measured"][k] for k in MEASURED_FEATURES] for r in rows]))
    if model in ("measured_plus_ttcc", "ttcc_only"):
        cols.append(np.array([[(r["ttcc"] or {}).get(k, float("nan")) for k in TTCC_FEATURES] for r in rows]))
    return np.hstack(cols)


# ---------------------------------------------------------------- model and cross-validation

def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge with in-fold median imputation and standardisation (statistics from the training rows only)."""
    ym = float(ytr.mean())
    if Xtr.shape[1] == 0:
        return np.full(len(Xte), ym)
    with warnings.catch_warnings():  # a feature missing in every training row: imputed with 0 (it then carries no information)
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ytr - ym))
    return ((Xte - mu) / sd) @ w + ym


def grouped_folds(groups: list[str], folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Test-index arrays; every group (creator) is in exactly one fold."""
    uniq = sorted(set(groups))
    order = rng.permutation(len(uniq))
    fold_of = {uniq[g]: i % folds for i, g in enumerate(order)}
    idx = np.array([fold_of[g] for g in groups])
    return [np.where(idx == k)[0] for k in range(folds)]


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: list[str], folds: int, repeats: int, alpha: float, seed: int) -> np.ndarray:
    """Out-of-fold predictions averaged over repeated grouped K-fold splits."""
    n = len(y)
    acc, cnt = np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        for test in grouped_folds(groups, folds, rng):
            if len(test) == 0:
                continue
            train = np.setdiff1d(np.arange(n), test)
            acc[test] += fit_predict(X[train], y[train], X[test], alpha)
            cnt[test] += 1
    return acc / np.maximum(cnt, 1)


def auc(score, label) -> float:
    """Area under the ROC curve (Mann-Whitney; ties count half)."""
    s, l_ = np.asarray(score, float), np.asarray(label, int)
    pos, neg = s[l_ == 1], s[l_ == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def _metrics(score: np.ndarray, y: np.ndarray, b: np.ndarray) -> dict:
    sp = spearman(score, y)
    return {"auc": auc(score, b), "spearman": None if math.isnan(sp) else sp}


def cluster_bootstrap(scores: dict[str, np.ndarray], y: np.ndarray, b: np.ndarray, groups: list[str], n_boot: int, seed: int,
                      pairs: list[tuple[str, str]]) -> dict:
    """Percentile 95% intervals, resampling creators (all of a creator's reels together); paired differences on the same resamples."""
    rng = np.random.default_rng(seed + 1)
    uniq = sorted(set(groups))
    members = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in uniq}
    draws: dict[str, dict[str, list]] = {k: {"auc": [], "spearman": []} for k in [*scores, *(f"{a}_minus_{c}" for a, c in pairs)]}
    for _ in range(n_boot):
        idx = np.concatenate([members[uniq[j]] for j in rng.integers(0, len(uniq), len(uniq))])
        yy, bb = y[idx], b[idx]
        m = {k: _metrics(s[idx], yy, bb) for k, s in scores.items()}
        for k, v in m.items():
            for met in ("auc", "spearman"):
                if v[met] is not None and not math.isnan(v[met]):
                    draws[k][met].append(v[met])
        for a, c in pairs:
            for met in ("auc", "spearman"):
                if all(m[x][met] is not None and not math.isnan(m[x][met]) for x in (a, c)):
                    draws[f"{a}_minus_{c}"][met].append(m[a][met] - m[c][met])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= max(20, n_boot // 10) else None  # noqa: E731
    return {k: {met: ci(v) for met, v in d.items()} for k, d in draws.items()}


def permutation_test(X: np.ndarray, y: np.ndarray, groups: list[str], observed: dict, cfg: dict) -> dict:
    """Shuffle the continuous label across reels (the binary label follows it) and re-run the whole grouped cross-validation."""
    rng = np.random.default_rng(cfg["seed"] + 2)
    hits = {"auc": 0, "spearman": 0}
    done = 0
    for _ in range(cfg["n_perm"]):
        yp = rng.permutation(y)
        s = oof_predictions(X, yp, groups, cfg["folds"], cfg["repeats"], cfg["alpha"], cfg["seed"])
        m = _metrics(s, yp, (yp > 0).astype(int))
        done += 1
        if m["auc"] >= observed["auc"]:
            hits["auc"] += 1
        if observed["spearman"] is not None and m["spearman"] is not None and m["spearman"] >= observed["spearman"]:
            hits["spearman"] += 1
    return {"n_permutations": done, "p_auc": (hits["auc"] + 1) / (done + 1),
            "p_spearman": (hits["spearman"] + 1) / (done + 1) if observed["spearman"] is not None else None,
            "alternative": "one-sided: the model scores at least as well as on shuffled labels"}


# ---------------------------------------------------------------- the evaluation

def evaluate(rows: list[dict], cfg: dict | None = None, log=print) -> dict:
    cfg = DEFAULTS | (cfg or {})
    regular = [r for r in rows if not r["is_outlier"]]
    outliers = [r for r in rows if r["is_outlier"]]
    groups = [r["handle"] for r in regular]
    y = np.array([r["overperformance"] for r in regular])
    b = np.array([r["above_band_median"] for r in regular])
    ttcc_idx = np.array([i for i, r in enumerate(regular) if r["ttcc"] is not None], dtype=int)
    results: dict[str, Any] = {"models": {}, "scores": {}, "status": {}}
    run_models = list(MODELS)
    ttcc_ok = len(ttcc_idx) >= MIN_TTCC_VIDEOS
    for model in run_models:
        if model in ("measured_plus_ttcc", "ttcc_only") and not ttcc_ok:
            results["status"][model] = (f"not run: {len(ttcc_idx)} of {len(regular)} regular reels have a TTCC hook prediction; at least "
                                        f"{MIN_TTCC_VIDEOS} are needed (run scripts/run_ttcc.py --long-videos hook first)")
            continue
        sub = ttcc_idx if model in ("measured_plus_ttcc", "ttcc_only") else np.arange(len(regular))
        X = design([regular[i] for i in sub], model)
        s = np.zeros(len(sub)) if model == "constant" else oof_predictions(X, y[sub], [groups[i] for i in sub], cfg["folds"], cfg["repeats"], cfg["alpha"], cfg["seed"])
        full = np.full(len(regular), np.nan)
        full[sub] = s
        results["scores"][model] = full
        results["models"][model] = {"n": int(len(sub)), "n_features": int(X.shape[1]), **_metrics(s, y[sub], b[sub])}
        results["status"][model] = "ok" if len(sub) == len(regular) else f"ok on the {len(sub)} reels with a TTCC prediction"
        log(f"views-eval: {model}: AUC {results['models'][model]['auc']:.3f}")
    # measured on the TTCC subset, so the TTCC comparison is on identical reels
    if ttcc_ok and len(ttcc_idx) < len(regular):
        X = design([regular[i] for i in ttcc_idx], "measured")
        s = oof_predictions(X, y[ttcc_idx], [groups[i] for i in ttcc_idx], cfg["folds"], cfg["repeats"], cfg["alpha"], cfg["seed"])
        full = np.full(len(regular), np.nan)
        full[ttcc_idx] = s
        results["scores"]["measured_on_ttcc_subset"] = full
        results["models"]["measured_on_ttcc_subset"] = {"n": int(len(ttcc_idx)), "n_features": int(X.shape[1]), **_metrics(s, y[ttcc_idx], b[ttcc_idx])}
        results["status"]["measured_on_ttcc_subset"] = "ok"

    # bootstrap: all-reel models together, TTCC models on their subset
    full_models = [m for m in results["scores"] if not np.isnan(results["scores"][m]).any()]
    pairs = [(a, c) for a, c in (("measured", "constant"), ("measured", "duration"), ("measured", "follower_band"), ("duration", "constant"),
                                         ("measured_plus_ttcc", "measured"), ("ttcc_only", "constant"))
             if a in full_models and c in full_models]
    ci = cluster_bootstrap({m: results["scores"][m] for m in full_models}, y, b, groups, cfg["n_boot"], cfg["seed"], pairs)
    sub_models = [m for m in results["scores"] if m not in full_models]
    if sub_models:
        base = "measured" if "measured_on_ttcc_subset" not in results["scores"] else "measured_on_ttcc_subset"
        names = sorted(set(sub_models) | {base})
        sub_pairs = [("measured_plus_ttcc", base)] if "measured_plus_ttcc" in names else []
        ci |= cluster_bootstrap({m: results["scores"][m][ttcc_idx] for m in names}, y[ttcc_idx], b[ttcc_idx], [groups[i] for i in ttcc_idx],
                                cfg["n_boot"], cfg["seed"], sub_pairs)
    results["bootstrap_95ci"] = ci

    perm = {}
    for model in results["models"]:
        if model == "constant":
            perm[model] = {"n_permutations": 0, "p_auc": None, "p_spearman": None, "note": "no fitted model: AUC is 0.5 by definition"}
            continue
        sub = ttcc_idx if model in ("measured_plus_ttcc", "ttcc_only", "measured_on_ttcc_subset") else np.arange(len(regular))
        base_model = "measured" if model == "measured_on_ttcc_subset" else model
        log(f"views-eval: permutation test for {model} ({cfg['n_perm']} shuffles)")
        perm[model] = permutation_test(design([regular[i] for i in sub], base_model), y[sub], [groups[i] for i in sub], results["models"][model], cfg)
    results["permutation"] = perm

    results["outliers"] = score_outliers(rows, regular, outliers, results, cfg)
    results["cfg"] = cfg
    return results


def score_outliers(rows: list[dict], regular: list[dict], outliers: list[dict], results: dict, cfg: dict) -> dict:
    """Each outlier is scored by a model fitted on the regular reels of OTHER creators (some outlier creators also have a regular reel).
    Regular reels keep their out-of-fold scores, so all 110 can be ranked on comparable held-out scores."""
    ranking: dict[str, Any] = {"per_model": {}, "rows": []}
    y = np.array([r["overperformance"] for r in regular])
    for model in results["models"]:
        if model in ("constant", "measured_on_ttcc_subset"):
            continue
        need_ttcc = model in ("measured_plus_ttcc", "ttcc_only")
        reg_idx = [i for i, r in enumerate(regular) if (r["ttcc"] is not None or not need_ttcc)]
        outs = [o for o in outliers if (o["ttcc"] is not None or not need_ttcc)]
        o_scores = {}
        for o in outs:
            train = [i for i in reg_idx if regular[i]["handle"] != o["handle"]]
            Xtr = design([regular[i] for i in train], model)
            o_scores[o["video_id"]] = float(fit_predict(Xtr, y[train], design([o], model), cfg["alpha"])[0])
        reg_scores = {regular[i]["video_id"]: float(results["scores"][model][i]) for i in reg_idx}
        allv = {**reg_scores, **o_scores}
        order = sorted(allv, key=lambda k: -allv[k])
        rank = {k: i + 1 for i, k in enumerate(order)}
        lab = np.array([1 if k in o_scores else 0 for k in allv])
        sc = np.array(list(allv.values()))
        ranking["per_model"][model] = {"n_ranked": len(allv), "n_outliers_scored": len(o_scores), "auc_outlier_vs_regular": auc(sc, lab) if o_scores else None,
                                       "outlier_ranks": {k: rank[k] for k in o_scores}, "median_outlier_rank": float(np.median([rank[k] for k in o_scores])) if o_scores else None}
        for k in allv:
            ranking.setdefault("_scores", {}).setdefault(k, {})[model] = (allv[k], rank[k], len(allv))
    by_id = {r["video_id"]: r for r in rows}
    for vid, per in ranking.pop("_scores", {}).items():
        r = by_id[vid]
        ranking["rows"].append({"video_id": vid, "is_outlier": r["is_outlier"], "handle": r["handle"], "band": r["band"], "views": r["views"],
                                "overperformance": r["overperformance"], "scores": {m: {"score": s, "rank": k, "of": n} for m, (s, k, n) in per.items()}})
    ranking["note"] = ("Outlier scores come from models that never saw that creator. Regular reels' scores are out-of-fold. With 10 outliers, any AUC "
                       "here has a wide uncertainty; treat it as a case study.")
    return ranking


# ---------------------------------------------------------------- writing

def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


def new_run_dir(out_root: Path, stamp: str) -> Path:
    """A fresh timestamped directory; an existing one is never reused or overwritten."""
    out_root.mkdir(parents=True, exist_ok=True)
    d, k = out_root / stamp, 1
    while d.exists():
        k += 1
        d = out_root / f"{stamp}_{k}"
    d.mkdir()
    return d


def write_outputs(run_dir: Path, rows: list[dict], medians: dict, coverage: dict, results: dict, meta: dict) -> dict[str, str]:
    regular = [r for r in rows if not r["is_outlier"]]
    ci, perm = results["bootstrap_95ci"], results["permutation"]
    paths = {}

    table = []
    for model in [*MODELS, "measured_on_ttcc_subset"]:
        m = results["models"].get(model)
        c = ci.get(model) or {}
        p = perm.get(model) or {}
        table.append({"model": model, "status": results["status"].get(model, "not run"), "n": (m or {}).get("n"), "n_features": (m or {}).get("n_features"),
                      "auc": (m or {}).get("auc"), "auc_ci_low": (c.get("auc") or [None, None])[0], "auc_ci_high": (c.get("auc") or [None, None])[1],
                      "spearman": (m or {}).get("spearman"), "spearman_ci_low": (c.get("spearman") or [None, None])[0],
                      "spearman_ci_high": (c.get("spearman") or [None, None])[1], "perm_p_auc": p.get("p_auc"), "perm_p_spearman": p.get("p_spearman")})
    with (run_dir / "baseline_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        for t in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else ("" if v is None or (isinstance(v, float) and math.isnan(v)) else v))
                        for k, v in t.items()})
    paths["baseline_comparison"] = "baseline_comparison.csv"

    with (run_dir / "prediction_results.csv").open("w", newline="") as f:
        cols = ["video_id", "handle", "follower_band", "views", "overperformance", "above_band_median", *[f"oof_{m}" for m in results["scores"]],
                *MEASURED_FEATURES, *TTCC_FEATURES, "duration_s"]
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(regular):
            sc = [("" if math.isnan(results["scores"][m][i]) else round(float(results["scores"][m][i]), 6)) for m in results["scores"]]
            feats = [("" if math.isnan(r["measured"][k]) else round(r["measured"][k], 6)) for k in MEASURED_FEATURES]
            tt = [("" if not r["ttcc"] or math.isnan(r["ttcc"][k]) else round(r["ttcc"][k], 6)) for k in TTCC_FEATURES]
            w.writerow([r["video_id"], r["handle"], r["band"], r["views"], round(r["overperformance"], 6), r["above_band_median"], *sc, *feats, *tt,
                        "" if math.isnan(r["duration_s"]) else round(r["duration_s"], 3)])
    paths["prediction_results"] = "prediction_results.csv"

    out_models = [m for m in results["outliers"]["per_model"]]
    with (run_dir / "outlier_ranking.csv").open("w", newline="") as f:
        cols = ["video_id", "is_outlier", "handle", "follower_band", "views", "overperformance", *[c for m in out_models for c in (f"score_{m}", f"rank_{m}", f"of_{m}")]]
        w = csv.writer(f)
        w.writerow(cols)
        key = "measured" if "measured" in out_models else (out_models[0] if out_models else None)
        rows_sorted = sorted(results["outliers"]["rows"], key=lambda r: (r["scores"].get(key) or {}).get("rank", 10**9))
        for r in rows_sorted:
            vals = []
            for m in out_models:
                s = r["scores"].get(m)
                vals += [round(s["score"], 6), s["rank"], s["of"]] if s else ["", "", ""]
            w.writerow([r["video_id"], int(r["is_outlier"]), r["handle"], r["band"], r["views"], round(r["overperformance"], 6), *vals])
    paths["outlier_ranking"] = "outlier_ranking.csv"

    (run_dir / "bootstrap_confidence_intervals.json").write_text(json.dumps(_jsonable({
        "method": "percentile 95% intervals; resampling creators (handles) with replacement; out-of-fold scores held fixed",
        "n_boot": results["cfg"]["n_boot"], "intervals": ci}), indent=2))
    paths["bootstrap"] = "bootstrap_confidence_intervals.json"
    (run_dir / "permutation_test.json").write_text(json.dumps(_jsonable({
        "method": "labels shuffled across reels, whole grouped cross-validation re-run each time; p = (hits + 1) / (n + 1)", "results": perm}), indent=2))
    paths["permutation"] = "permutation_test.json"

    summary = {"kind": "verified evaluation result on real views (see note)", "note": VERIFIED_NOTE,
               "label": {"continuous": "overperformance = log(views) - log(median views of the follower band over the 100 regular reels)",
                         "binary_primary": "above_band_median = overperformance > 0", "band_medians": medians,
                         "positives": int(sum(r["above_band_median"] for r in regular)), "n_regular": len(regular)},
               "models": results["models"], "status": results["status"], "bootstrap_95ci": ci,
               "permutation": {m: {k: v for k, v in p.items() if k in ("p_auc", "p_spearman", "n_permutations")} for m, p in perm.items()},
               "outliers": {"per_model": results["outliers"]["per_model"], "note": results["outliers"]["note"]},
               "coverage": {"measured_missing": coverage["measured_missing"], "ttcc_available": len(coverage["ttcc_available"]),
                            "ttcc_missing": len(coverage["ttcc_missing"]), "ttcc_configs": coverage["ttcc_configs"]},
               "files": paths}
    (run_dir / "metrics_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    paths["metrics_summary"] = "metrics_summary.json"
    (run_dir / "run_metadata.json").write_text(json.dumps(_jsonable(meta), indent=2))
    paths["run_metadata"] = "run_metadata.json"
    return paths


def run_metadata(cfg: dict, data_root: Path, features_root: Path, ttcc_root: Path | None, rows: list[dict], stamp: str) -> dict:
    files = {}
    for g in (*REGULAR_GROUPS, OUTLIER_GROUP):
        p = data_root / g / "metadata.csv"
        files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"timestamp_utc": stamp, "config": cfg, "seed": cfg["seed"],
            "dataset": {"n_total": len(rows), "n_regular": sum(not r["is_outlier"] for r in rows), "n_outliers": sum(r["is_outlier"] for r in rows),
                        "n_handles": len({r["handle"] for r in rows}), "metadata_sha256": files},
            "inputs": {"data_root": str(data_root), "features_root": str(features_root), "ttcc_root": str(ttcc_root) if ttcc_root else None},
            "model_settings": {"estimator": "ridge regression on the continuous label (numpy), alpha fixed, no tuning",
                               "preprocessing": "median imputation and standardisation fitted on the training fold only",
                               "cv": f"grouped by creator handle, {cfg['folds']} folds x {cfg['repeats']} repeats, out-of-fold scores averaged",
                               "measured_features": list(MEASURED_FEATURES), "ttcc_features": list(TTCC_FEATURES),
                               "feature_choice": "fixed before any result was seen; not selected on this data"},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(terse=True)},
            "secrets": "none recorded"}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
