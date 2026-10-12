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

Metrics: AUC of the predicted score for ``above_band_median`` and Spearman with ``overperformance``, each computed INSIDE every held-out fold and
averaged over folds and repeats (see ``foldwise_metrics``: pooling scores across folds biases weak models below 0.5), 95% bootstrap intervals
(resampling creators), paired bootstrap differences between models, and permutation tests that re-run the whole cross-validation and metric on
shuffled labels.
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

from .brain_networks import HOOK_FEATURES, NETWORKS, network_features
from .evaluation import TRIBE_FEATURES, spearman

REGULAR_GROUPS = ("face", "faceless")
OUTLIER_GROUP = "outliers"
BANDS = ("10k-50k", "50k-200k", "200k-1M", "1M+")
MEASURED_FEATURES = ("cuts_per_s_0_5", "first_cut_s", "motion_mean_0_5", "brightness_mean_0_5", "rms_dbfs_0_5", "first_word_s",
                     "words_per_s_0_5", "first_text_s")
TTCC_FEATURES = ("ttcc_R3", "ttcc_R5", "ttcc_R10", "ttcc_steepest_early_drop")
PERSON_FEATURES = ("person_max_0_3s", "talking_to_camera_mean_0_3s")  # SigLIP zero-shot scores over the first 3 s: a person / someone speaking to the camera
TRIBE_FEATURE_COLS = tuple(f"tribe_{n}" for n in TRIBE_FEATURES)
TRIBENET_FEATURE_COLS = tuple(f"tribenet_{n}" for n in (*(f"net_{x}" for x in NETWORKS), *HOOK_FEATURES))  # Yeo-7 network profile + attention-curve peak
TRIBE_WINDOW_S = 10.0  # TRIBE summaries use the first 10 s only, the same opening the hook analysis looks at
# Feature groups per model. "page_type" is dataset metadata (the folder the reel came from); the rest are computed from the video.
MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "constant": (), "follower_band": ("band",), "duration": ("duration",), "page_type": ("page_type",), "measured": ("measured",),
    "measured_plus_page_type": ("measured", "page_type"), "person_only": ("person",), "measured_plus_person": ("measured", "person"), "ttcc_only": ("ttcc",), "measured_plus_ttcc": ("measured", "ttcc"),
    "tribe_only": ("tribe",), "ttcc_plus_tribe": ("ttcc", "tribe"), "measured_plus_ttcc_plus_tribe": ("measured", "ttcc", "tribe"),
    "tribenet_only": ("tribenet",), "measured_plus_tribenet": ("measured", "tribenet"), "page_type_plus_tribenet": ("page_type", "tribenet")}
MODELS = tuple(MODEL_GROUPS)
SUBSET_GROUPS = ("ttcc", "tribe", "tribenet")  # groups that only some reels have: such a model runs on the reels that have all of them
# (bigger model, smaller model): the paired bootstrap difference tells whether the extra feature group adds anything
PAIRS = (("measured", "constant"), ("measured", "duration"), ("measured", "follower_band"), ("measured", "page_type"), ("duration", "constant"),
         ("page_type", "constant"), ("measured_plus_page_type", "measured"), ("person_only", "constant"), ("measured_plus_person", "measured"), ("measured_plus_ttcc", "measured"), ("ttcc_only", "constant"),
         ("tribe_only", "constant"), ("ttcc_plus_tribe", "ttcc_only"), ("ttcc_plus_tribe", "tribe_only"),
         ("measured_plus_ttcc_plus_tribe", "measured_plus_ttcc"),
         ("tribenet_only", "constant"), ("tribenet_only", "tribe_only"), ("measured_plus_tribenet", "measured"),
         ("page_type_plus_tribenet", "page_type"), ("tribenet_only", "page_type"))
DEFAULTS = {"seed": 0, "folds": 5, "repeats": 20, "alpha": 10.0, "n_boot": 2000, "n_perm": 1000}
MIN_TTCC_VIDEOS = 30  # also the minimum for any model that needs TRIBE
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
                             "group": group, "is_outlier": group == OUTLIER_GROUP, "page_type": r.get("page_type") or group,
                             "video_path": str(data_root / group / "videos" / f"{r['video_id']}.mp4")})
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
    shown = (art.get("inference") or {}).get("window_end_s")
    if art.get("stitched") and not (art.get("hook") or {}).get("native"):
        return None, {"reason": "stitched windowed curve without a native first-window hook: not a start-anchored hook prediction"}
    h = art.get("hook") or hook_metrics(art["retention"], shown)  # a stitched curve's hook is its first window, read before blending
    info = {"prompt_T": art.get("horizon_s"), "seconds_shown": h.get("seconds_shown"), "policy": art.get("policy"), "cache_key": art.get("cache_key"), "hook": h}
    if any(d_ for d_ in (art.get("inference") or {}).get("deviations_from_upstream_eval") or [] if "sensitivity run" in d_):
        return None, {"reason": "sensitivity run (model told a different length than it saw): not used for evaluation"} | info
    steep = h.get("steepest_early_drop") or {}
    f = {"ttcc_R3": _num(h.get("R3")), "ttcc_R5": _num(h.get("R5")), "ttcc_R10": _num(h.get("R10")), "ttcc_steepest_early_drop": _num(steep.get("drop"))}
    return f, info


def person_features(d: Path) -> dict | None:
    """Is a person (or someone speaking to the camera) on screen in the first 3 s? From SigLIP's per-frame zero-shot scores (``siglip.json``); None when
    SigLIP was not run. A video-computed stand-in for the dataset's face / faceless label, which a new upload does not come with."""
    p = d / "siglip.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text())
    if s.get("status") != "available":
        return None
    fr = [f for f in s.get("frames") or [] if f.get("t_s") is not None and f["t_s"] < 3.0 and f.get("scores")]
    if not fr:
        return None
    return {"person_max_0_3s": max(max(f["scores"].get("person_closeup", 0.0), f["scores"].get("talking_to_camera", 0.0)) for f in fr),
            "talking_to_camera_mean_0_3s": float(np.mean([f["scores"].get("talking_to_camera", 0.0) for f in fr]))}


def tribe_features(d: Path) -> tuple[dict | None, dict | None]:
    """Eight summaries (``evaluation.tribe_summary``) of TRIBE's predicted brain response over the first 10 s; (None, why) when absent.
    TRIBE predicts fMRI-like responses, NOT retention: these are descriptive features whose usefulness is what the evaluation tests."""
    from .evaluation import tribe_summary

    tj, npy = d / "tribe.json", d / "tribe_preds.npy"
    if not (tj.is_file() and npy.is_file()):
        return None, {"reason": "no TRIBE output"}
    t = json.loads(tj.read_text())
    if not t.get("measured"):
        return None, {"reason": "TRIBE output not measured"}
    preds, starts = np.load(npy), np.asarray(t["timestamps_s"], float)
    keep = starts < TRIBE_WINDOW_S
    if keep.sum() < 3:
        return None, {"reason": "fewer than 3 TRIBE segments in the first 10 s"}
    vals = tribe_summary(preds[keep], list(starts[keep]), TRIBE_WINDOW_S)
    if not all(math.isfinite(v) for v in vals):
        return None, {"reason": "TRIBE summary has missing values"}
    return dict(zip(TRIBE_FEATURE_COLS, vals)), {"segments_used": int(keep.sum())}


def tribenet_features(d: Path) -> tuple[dict | None, dict | None]:
    """Yeo-7 network profile and attention-curve peak of TRIBE's predicted response over the first 10 s (``brain_networks``); (None, why) when absent."""
    tj, npy = d / "tribe.json", d / "tribe_preds.npy"
    if not (tj.is_file() and npy.is_file()):
        return None, {"reason": "no TRIBE output"}
    t = json.loads(tj.read_text())
    if not t.get("measured"):
        return None, {"reason": "TRIBE output not measured"}
    f = network_features(np.load(npy), t["timestamps_s"], TRIBE_WINDOW_S)
    if f is None:
        return None, {"reason": "fewer than 4 TRIBE segments in the first 10 s"}
    if not all(math.isfinite(v) for v in f.values()):
        return None, {"reason": "network features have missing values"}
    return {f"tribenet_{k}": v for k, v in f.items()}, {}


def assemble(rows: list[dict], features_root: Path, ttcc_root: Path | None, tribe_root: Path | None = None) -> dict:
    """Attach features. Returns coverage facts; nothing is invented for a missing file (the row is marked and imputed in-fold, or excluded)."""
    cov = {"measured_missing": [], "ttcc_available": [], "ttcc_missing": {}, "ttcc_configs": {}, "tribe_available": [], "tribe_missing": {}, "tribenet_available": []}
    for r in rows:
        d = features_root / r["video_id"]
        try:
            r["measured"], r["duration_s"] = measured_features(d)
        except (OSError, ValueError, KeyError) as exc:
            r["measured"], r["duration_s"] = {k: float("nan") for k in MEASURED_FEATURES}, float("nan")
            cov["measured_missing"].append(f"{r['video_id']}: {type(exc).__name__}")
        r["person"] = person_features(d)
        r["ttcc"] = r["tribe"] = r["tribenet"] = None
        if ttcc_root is not None:
            f, info = ttcc_features(ttcc_root / r["video_id"])
            if f is None:
                cov["ttcc_missing"][r["video_id"]] = info["reason"]
            else:
                r["ttcc"] = f
                cov["ttcc_available"].append(r["video_id"])
                key = f"prompt_T={info['prompt_T']},shown={info['seconds_shown']}"
                cov["ttcc_configs"][key] = cov["ttcc_configs"].get(key, 0) + 1
        if tribe_root is not None:
            f, info = tribe_features(tribe_root / r["video_id"])
            if f is None:
                cov["tribe_missing"][r["video_id"]] = info["reason"]
            else:
                r["tribe"] = f
                cov["tribe_available"].append(r["video_id"])
            nf, _ = tribenet_features(tribe_root / r["video_id"])
            if nf is not None:
                r["tribenet"] = nf
                cov["tribenet_available"].append(r["video_id"])
    return cov


def design(rows: list[dict], model: str) -> np.ndarray:
    """Feature matrix of a model: the columns of each of its feature groups."""
    n = len(rows)
    cols = []
    for g in MODEL_GROUPS[model]:
        if g == "band":
            cols.append(np.array([[1.0 if r["band"] == b else 0.0 for b in BANDS] for r in rows]))
        elif g == "duration":
            cols.append(np.array([[math.log(r["duration_s"]) if r["duration_s"] and r["duration_s"] > 0 else float("nan")] for r in rows]))
        elif g == "page_type":
            cols.append(np.array([[1.0 if r.get("page_type") == "face" else 0.0] for r in rows]))
        elif g == "measured":
            cols.append(np.array([[r["measured"][k] for k in MEASURED_FEATURES] for r in rows]))
        elif g == "ttcc":
            cols.append(np.array([[(r["ttcc"] or {}).get(k, float("nan")) for k in TTCC_FEATURES] for r in rows]))
        elif g == "person":
            cols.append(np.array([[(r.get("person") or {}).get(k, float("nan")) for k in PERSON_FEATURES] for r in rows]))
        elif g == "tribe":
            cols.append(np.array([[(r.get("tribe") or {}).get(k, float("nan")) for k in TRIBE_FEATURE_COLS] for r in rows]))
        elif g == "tribenet":
            cols.append(np.array([[(r.get("tribenet") or {}).get(k, float("nan")) for k in TRIBENET_FEATURE_COLS] for r in rows]))
    return np.hstack(cols) if cols else np.zeros((n, 0))


def needs(model: str) -> tuple[str, ...]:
    return tuple(g for g in MODEL_GROUPS[model] if g in SUBSET_GROUPS)


def has(row: dict, groups: tuple[str, ...]) -> bool:
    return all(row.get(g) is not None for g in groups)


def subset_key(model: str) -> str:
    return "+".join(needs(model)) or "all"


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


def cv_scores(X: np.ndarray, y: np.ndarray, groups: list[str], folds: int, repeats: int, alpha: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Held-out score of every reel in every repeat of a grouped K-fold: (P, F), both (repeats, n); F is the fold id the reel was held out in."""
    n = len(y)
    P, F = np.zeros((repeats, n)), np.zeros((repeats, n), dtype=int)
    rng = np.random.default_rng(seed)
    for r in range(repeats):
        for k, test in enumerate(grouped_folds(groups, folds, rng)):
            if len(test) == 0:
                continue
            train = np.setdiff1d(np.arange(n), test)
            P[r, test] = fit_predict(X[train], y[train], X[test], alpha)
            F[r, test] = k
    return P, F


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: list[str], folds: int, repeats: int, alpha: float, seed: int) -> np.ndarray:
    """Held-out scores averaged over the repeats (one score per reel; used for rankings and files, not for the headline metrics)."""
    return cv_scores(X, y, groups, folds, repeats, alpha, seed)[0].mean(0)


def auc(score, label) -> float:
    """Area under the ROC curve (Mann-Whitney; ties count half)."""
    s, l_ = np.asarray(score, float), np.asarray(label, int)
    pos, neg = s[l_ == 1], s[l_ == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


MIN_FOLD_REELS = 6


def foldwise_metrics(P: np.ndarray, F: np.ndarray, y: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None, max_repeats: int | None = None) -> dict:
    """AUC (for ``b``) and Spearman (with ``y``) computed inside each held-out fold, then averaged over folds and repeats.

    Why not pool all held-out scores: every fold's model is fitted on different reels, so its intercept and group means differ slightly, and a fold
    whose reels do well is trained on the others, which do worse. Pooling then ranks across folds and biases AUC downward for weak models
    (a mean-only model scores well below 0.5). Inside one fold all reels share the same model, so ties and ranks are meaningful.
    ``idx`` selects reels (duplicates allowed: a bootstrap resample)."""
    idx = np.arange(len(y)) if idx is None else idx
    aucs, sps = [], []
    for r in range(P.shape[0] if max_repeats is None else min(max_repeats, P.shape[0])):
        fr = F[r][idx]
        for k in np.unique(fr):
            sel = idx[fr == k]
            if len(sel) < MIN_FOLD_REELS:
                continue
            a = auc(P[r][sel], b[sel])
            if not math.isnan(a):
                aucs.append(a)
            sp = spearman(P[r][sel], y[sel])
            if not math.isnan(sp):
                sps.append(sp)
    return {"auc": float(np.mean(aucs)) if aucs else float("nan"), "spearman": float(np.mean(sps)) if sps else None}


BOOT_REPEATS = 4  # cross-validation repeats used inside each bootstrap resample (the point estimate uses all of them)


def cluster_bootstrap(cvs: dict[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray, b: np.ndarray, groups: list[str], n_boot: int, seed: int,
                      pairs: list[tuple[str, str]]) -> dict:
    """Percentile 95% intervals, resampling creators (all of a creator's reels together); paired differences on the same resamples.
    The held-out predictions are fixed; each resample recomputes the fold-wise metrics on the resampled reels."""
    rng = np.random.default_rng(seed + 1)
    uniq = sorted(set(groups))
    members = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in uniq}
    draws: dict[str, dict[str, list]] = {k: {"auc": [], "spearman": []} for k in [*cvs, *(f"{a}_minus_{c}" for a, c in pairs)]}
    for _ in range(n_boot):
        idx = np.concatenate([members[uniq[j]] for j in rng.integers(0, len(uniq), len(uniq))])
        m = {k: foldwise_metrics(P, F, y, b, idx, BOOT_REPEATS) for k, (P, F) in cvs.items()}
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
    """Shuffle the continuous label across reels (the binary label follows it) and re-run the whole grouped cross-validation and metric."""
    rng = np.random.default_rng(cfg["seed"] + 2)
    hits = {"auc": 0, "spearman": 0}
    done = 0
    for _ in range(cfg["n_perm"]):
        yp = rng.permutation(y)
        P, F = cv_scores(X, yp, groups, cfg["folds"], cfg["repeats"], cfg["alpha"], cfg["seed"])
        m = foldwise_metrics(P, F, yp, (yp > 0).astype(int))
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
    results: dict[str, Any] = {"models": {}, "scores": {}, "cv": {}, "status": {}, "subset_idx": {}, "base": {}}
    counts = {g: sum(r.get(g) is not None for r in regular) for g in SUBSET_GROUPS}

    def idx_for(key: str) -> np.ndarray:
        gs = () if key == "all" else tuple(key.split("+"))
        return np.array([i for i, r in enumerate(regular) if has(r, gs)], dtype=int)

    def run(name: str, model: str, idx: np.ndarray, key: str) -> None:
        X = design([regular[i] for i in idx], model)
        P, F = cv_scores(X, y[idx], [groups[i] for i in idx], cfg["folds"], cfg["repeats"], cfg["alpha"], cfg["seed"])
        if model == "constant":
            P = np.zeros_like(P)
        full = np.full(len(regular), np.nan)
        full[idx] = P.mean(0)
        results["scores"][name] = full
        results["cv"][name] = (P, F)
        results["models"][name] = {"n": int(len(idx)), "n_features": int(X.shape[1]), "groups": list(MODEL_GROUPS[model]),
                                   **foldwise_metrics(P, F, y[idx], b[idx])}
        results["subset_idx"][name], results["base"][name] = key, model
        results["status"][name] = "ok" if len(idx) == len(regular) else f"ok on the {len(idx)} reels that have {key.replace('+', ' and ')} output"
        log(f"views-eval: {name}: AUC {results['models'][name]['auc']:.3f}")

    for model in MODEL_GROUPS:
        key = subset_key(model)
        idx = idx_for(key)
        if key != "all" and len(idx) < MIN_TTCC_VIDEOS:
            have = ", ".join(f"{counts[g]} have {g}" for g in needs(model))
            results["status"][model] = (f"not run: {len(idx)} of {len(regular)} regular reels have every output this model needs ({have}); at least "
                                        f"{MIN_TTCC_VIDEOS} are needed")
            continue
        run(model, model, idx, key)
    # a bigger model is compared with its smaller sibling on the SAME reels: run the sibling on that subset when it ran on others
    refs: dict[tuple[str, str], str] = {}
    for a_, b_ in PAIRS:
        if a_ not in results["models"] or subset_key(a_) == "all":
            continue
        key = subset_key(a_)
        if subset_key(b_) == key:
            refs[(a_, b_)] = b_
            continue
        name = f"{b_}_on_{key.replace('+', '_')}_subset"
        if name not in results["models"]:
            run(name, b_, idx_for(key), key)
        refs[(a_, b_)] = name

    ci: dict[str, Any] = {}
    for key in sorted({k for k in results["subset_idx"].values()}):
        idx = idx_for(key)
        names = [m for m, k in results["subset_idx"].items() if k == key]
        pairs = [(a_, refs.get((a_, b_), b_)) for a_, b_ in PAIRS if a_ in names and refs.get((a_, b_), b_) in names]
        ci |= cluster_bootstrap({m: results["cv"][m] for m in names}, y[idx], b[idx], [groups[i] for i in idx], cfg["n_boot"], cfg["seed"], pairs)
    results["bootstrap_95ci"] = ci

    perm = {}
    for name in results["models"]:
        if results["base"][name] == "constant":
            perm[name] = {"n_permutations": 0, "p_auc": None, "p_spearman": None, "note": "no fitted model: AUC is 0.5 by definition"}
            continue
        idx = idx_for(results["subset_idx"][name])
        log(f"views-eval: permutation test for {name} ({cfg['n_perm']} shuffles)")
        perm[name] = permutation_test(design([regular[i] for i in idx], results["base"][name]), y[idx], [groups[i] for i in idx], results["models"][name], cfg)
    results["permutation"] = perm
    results["pairs"] = {f"{a_}_minus_{refs.get((a_, b_), b_)}": [a_, refs.get((a_, b_), b_)] for a_, b_ in PAIRS if a_ in results["models"]}
    results["outliers"] = score_outliers(rows, regular, outliers, results, cfg)
    results["cfg"] = cfg
    results.pop("cv")  # large arrays: needed only inside this function
    return results


def score_outliers(rows: list[dict], regular: list[dict], outliers: list[dict], results: dict, cfg: dict) -> dict:
    """Each outlier is scored by a model fitted on the regular reels of OTHER creators (some outlier creators also have a regular reel).
    Regular reels keep their out-of-fold scores, so all 110 can be ranked on comparable held-out scores."""
    ranking: dict[str, Any] = {"per_model": {}, "rows": []}
    y = np.array([r["overperformance"] for r in regular])
    for model in results["models"]:
        base = results["base"][model]
        if base == "constant" or "_on_" in model:
            continue
        gs = needs(base)
        reg_idx = [i for i, r in enumerate(regular) if has(r, gs)]
        outs = [o for o in outliers if has(o, gs)]
        o_scores = {}
        for o in outs:
            train = [i for i in reg_idx if regular[i]["handle"] != o["handle"]]
            Xtr = design([regular[i] for i in train], base)
            o_scores[o["video_id"]] = float(fit_predict(Xtr, y[train], design([o], base), cfg["alpha"])[0])
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
    for model in [*MODELS, *[m for m in results["models"] if m not in MODELS]]:
        m = results["models"].get(model)
        c = ci.get(model) or {}
        p = perm.get(model) or {}
        table.append({"model": model, "status": results["status"].get(model, "not run"), "n": (m or {}).get("n"), "n_features": (m or {}).get("n_features"),
                      "groups": "+".join((m or {}).get("groups") or []), "auc": (m or {}).get("auc"), "auc_ci_low": (c.get("auc") or [None, None])[0], "auc_ci_high": (c.get("auc") or [None, None])[1],
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
        cols = ["video_id", "handle", "follower_band", "page_type", "views", "overperformance", "above_band_median", *[f"oof_{m}" for m in results["scores"]],
                *MEASURED_FEATURES, *PERSON_FEATURES, *TTCC_FEATURES, *TRIBE_FEATURE_COLS, *TRIBENET_FEATURE_COLS, "duration_s"]
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(regular):
            sc = [("" if math.isnan(results["scores"][m][i]) else round(float(results["scores"][m][i]), 6)) for m in results["scores"]]
            feats = [("" if math.isnan(r["measured"][k]) else round(r["measured"][k], 6)) for k in MEASURED_FEATURES]
            tt = [("" if not r["ttcc"] or math.isnan(r["ttcc"][k]) else round(r["ttcc"][k], 6)) for k in TTCC_FEATURES]
            tb = [("" if not r.get("tribe") else round(r["tribe"][k], 6)) for k in TRIBE_FEATURE_COLS]
            tn = [("" if not r.get("tribenet") else round(r["tribenet"][k], 6)) for k in TRIBENET_FEATURE_COLS]
            pr = [("" if not r.get("person") else round(r["person"][k], 6)) for k in PERSON_FEATURES]
            w.writerow([r["video_id"], r["handle"], r["band"], r.get("page_type", ""), r["views"], round(r["overperformance"], 6), r["above_band_median"], *sc, *feats, *pr, *tt, *tb, *tn,
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
                            "ttcc_missing": len(coverage["ttcc_missing"]), "ttcc_configs": coverage["ttcc_configs"],
                            "tribe_available": len(coverage["tribe_available"]), "tribenet_available": len(coverage["tribenet_available"]), "tribe_missing": len(coverage["tribe_missing"])},
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
                               "measured_features": list(MEASURED_FEATURES), "person_features": list(PERSON_FEATURES), "person_note": "added after the first two runs, exploratory", "ttcc_features": list(TTCC_FEATURES), "tribe_features": list(TRIBE_FEATURE_COLS),
                               "page_type": "dataset metadata (the folder a reel came from), not computed from the video",
                               "feature_choice": "fixed before any result was seen; not selected on this data"},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(terse=True)},
            "secrets": "none recorded"}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
