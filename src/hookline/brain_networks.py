"""Yeo-7 network view of TRIBE v2's predicted brain response (exploratory; not retention, not a validated engagement score).

TRIBE v2 predicts an average subject's fMRI response on the fsaverage5 surface (20484 vertices, left hemisphere assumed first). Averaging those
vertices inside the Yeo 2011 7-network atlas gives one curve per network. The "attention-network" curve and its peaks are a descriptive summary of
those curves, not something Meta or the atlas authors define; whether they relate to real viewing behaviour is what scripts/eval_views.py tests.
"""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

ATLAS_DIR = Path(__file__).resolve().parents[2] / "data" / "atlas" / "yeo2011_7networks_fsaverage5"
N_PER_HEMI = 10242
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
NETWORK_LABELS = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attention", "SalVentAttn": "Ventral attention",
                  "Limbic": "Limbic", "Cont": "Frontoparietal", "Default": "Default mode"}
ATTENTION_NETWORKS = ("Vis", "DorsAttn")  # the attention-capture curve is the mean of these two network curves
HOOK_FEATURES = ("peak_time_s", "peak_height", "opening_gap")
INTERPRETIVE_NETWORKS = ("Limbic", "Cont", "Default")


def read_annot(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    """FreeSurfer .annot -> (label value per vertex, {label value: name}). Small reader so no neuroimaging package is needed."""
    b = path.read_bytes()
    n = struct.unpack(">i", b[:4])[0]
    pairs = np.frombuffer(b[4:4 + 8 * n], ">i4").reshape(n, 2)
    o = 4 + 8 * n
    if struct.unpack(">i", b[o:o + 4])[0] != 1:
        raise ValueError(f"{path.name}: no colour table")
    o += 4
    if struct.unpack(">i", b[o:o + 4])[0] != -2:
        raise ValueError(f"{path.name}: unsupported colour-table version")
    o += 8  # version, number of entries
    o += 4 + struct.unpack(">i", b[o:o + 4])[0]  # original file name
    nread = struct.unpack(">i", b[o:o + 4])[0]
    o += 4
    names = {}
    for _ in range(nread):
        ln = struct.unpack(">i", b[o + 4:o + 8])[0]
        o += 8
        name = b[o:o + ln].rstrip(b"\0").decode()
        o += ln
        r, g, bl, t = struct.unpack(">iiii", b[o:o + 16])
        o += 16
        names[r + (g << 8) + (bl << 16) + (t << 24)] = name
    labels = np.full(n, -1, dtype=np.int64)
    labels[pairs[:, 0]] = pairs[:, 1]
    return labels, names


@lru_cache(maxsize=1)
def vertex_networks() -> np.ndarray:
    """Network index into NETWORKS for each of the 20484 vertices; -1 for the medial wall."""
    out = []
    for hemi in ("lh", "rh"):
        labels, names = read_annot(ATLAS_DIR / f"{hemi}.annot")
        if len(labels) != N_PER_HEMI:
            raise ValueError(f"{hemi}.annot has {len(labels)} vertices, expected {N_PER_HEMI}")
        net_of = {}
        for value, name in names.items():
            part = name.split("_")  # e.g. 7Networks_LH_DorsAttn_Post
            net_of[value] = NETWORKS.index(part[2]) if len(part) > 2 and part[2] in NETWORKS else -1
        out.append(np.array([net_of.get(int(v), -1) for v in labels]))
    return np.concatenate(out)


def network_curves(preds: np.ndarray) -> dict[str, np.ndarray]:
    """(n_segments, 20484) predicted response -> one mean curve per Yeo-7 network."""
    preds = np.asarray(preds, float)
    nets = vertex_networks()
    if preds.ndim != 2 or preds.shape[1] != len(nets):
        raise ValueError(f"expected (n_segments, {len(nets)}) predictions, got {preds.shape}")
    return {n: preds[:, nets == i].mean(1) for i, n in enumerate(NETWORKS)}


def attention_curve(curves: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([curves[n] for n in ATTENTION_NETWORKS], axis=0)


def network_activation_shares(curves: dict[str, np.ndarray], starts: list[float], window_s: float = 10.0) -> list[dict]:
    """Share of absolute predicted response magnitude per network in the opening window.

    The denominator is the sum over Yeo-7 network curves, so the percentages are a descriptive profile of TRIBE's predicted fMRI response,
    not a probability of attention, emotion or purchase intent.
    """
    t = np.asarray(starts, float)
    keep = t < window_s
    if not len(t) or not keep.any():
        keep = np.ones(len(t), dtype=bool)
    mags = {n: float(np.mean(np.abs(np.asarray(curves[n], float)[keep]))) for n in NETWORKS}
    total = sum(mags.values()) or 1.0
    return [{"network": n, "label": NETWORK_LABELS[n], "share": mags[n] / total, "pct": round(100.0 * mags[n] / total)} for n in NETWORKS]


def network_reading(shares: list[dict]) -> dict:
    """Plain-language reading of the dominant network profile, with explicit uncertainty."""
    ordered = sorted(shares, key=lambda x: x["share"], reverse=True)
    top = ordered[:2]
    top_names = {x["network"] for x in top}
    limbic = next((x for x in shares if x["network"] == "Limbic"), {"share": 0.0, "pct": 0})
    if limbic["share"] >= 0.18:
        emotional = {"level": "strong", "pct": limbic["pct"], "text": "stronger emotional-salience signal"}
    elif limbic["share"] >= 0.10:
        emotional = {"level": "moderate", "pct": limbic["pct"], "text": "some emotional-salience signal"}
    else:
        emotional = {"level": "low", "pct": limbic["pct"], "text": "low emotional-salience signal"}
    low_interpretive = all(next((x["share"] for x in shares if x["network"] == n), 0.0) < 0.10 for n in INTERPRETIVE_NETWORKS)
    if top_names == set(ATTENTION_NETWORKS) and sum(x["share"] for x in top) >= 0.55 and low_interpretive:
        summary = ("Visual + dorsal-attention dominance: the opening looks like focused visual tracking. Limbic, frontoparietal and default-mode "
                   "responses are near baseline, so Hookline reads this as watching without much emotion, narrative or decision-state signal.")
        recommendation = "Before a CTA, add a deliberate transition: verbal prompt, text overlay or pacing shift from demonstration into evaluation."
    elif "Limbic" in top_names:
        summary = "Limbic is among the dominant predicted networks, so the opening may be carrying a stronger emotional signal."
        recommendation = "Keep the emotional beat tied to the product or story so the hook does not become detached from the decision moment."
    elif "Cont" in top_names:
        summary = "Frontoparietal/control is among the dominant predicted networks, suggesting a more evaluation or decision-oriented moment."
        recommendation = "Make the next action concrete while the viewer is already in an evaluation state."
    elif "Default" in top_names:
        summary = "Default-mode is among the dominant predicted networks, suggesting more narrative or self-referential processing."
        recommendation = "Clarify the story payoff and connect it back to the offer before attention drifts."
    else:
        summary = "No single interpretive profile dominates; use the time curve and peaks as exploratory hook/surprise candidates."
        recommendation = "Inspect the peak moments alongside the video, transcript and OCR before making an edit."
    return {"top": top, "emotional_signal": emotional, "summary": summary, "recommendation": recommendation,
            "caveat": "Exploratory brain-response signal, not measured emotion or purchase intent."}


def hook_peaks(curve: np.ndarray, starts: list[float], top: int = 3) -> list[dict]:
    """Local maxima of the attention-network curve, highest first. ``height`` is in standard deviations of the curve above its mean."""
    c = np.asarray(curve, float)
    if len(c) < 3 or float(c.std()) == 0.0:
        return []
    z = (c - c.mean()) / c.std()
    idx = [i for i in range(1, len(c) - 1) if c[i] > c[i - 1] and c[i] >= c[i + 1]]
    idx.sort(key=lambda i: -z[i])
    return [{"t_s": float(starts[i]), "height": float(z[i])} for i in idx[:top]]


def network_features(preds: np.ndarray, starts: list[float], window_s: float = 10.0) -> dict[str, float] | None:
    """Features of the first ``window_s`` seconds: the seven network means relative to the whole-cortex mean (a profile), plus where the
    attention-network curve peaks, how high, and how much the first 3 s differ from the rest. None when fewer than 4 segments fall in the window."""
    t = np.asarray(starts, float)
    keep = t < window_s
    if keep.sum() < 4:
        return None
    p, ts = np.asarray(preds, float)[keep], t[keep]
    curves = network_curves(p)
    overall = p.mean(1)
    feats = {f"net_{n}": float((curves[n] - overall).mean()) for n in NETWORKS}
    a = attention_curve(curves)
    peaks = hook_peaks(a, list(ts), top=1)
    sd = float(a.std()) or 1.0
    early, late = a[ts < 3.0], a[ts >= 3.0]
    feats["peak_time_s"] = peaks[0]["t_s"] if peaks else float("nan")
    feats["peak_height"] = peaks[0]["height"] if peaks else float("nan")
    feats["opening_gap"] = float((early.mean() - late.mean()) / sd) if len(early) and len(late) else float("nan")
    return feats


BRAIN_NOTE = "TRIBE-predicted brain response over time: emotional signal, dominant networks, and hook/surprise peaks."


def brain_view(video_dir: Path) -> dict | None:
    """Per-video payload for the review page's brain-response panel, from the saved TRIBE output (``tribe.json`` + ``tribe_preds.npy``)."""
    import json

    tj, npy = video_dir / "tribe.json", video_dir / "tribe_preds.npy"
    try:
        t = json.loads(tj.read_text())
        if not t.get("measured") or not npy.is_file():
            return None
        preds, starts = np.load(npy), [float(x) for x in t["timestamps_s"]]
        curves = network_curves(preds)
    except (OSError, ValueError, KeyError):
        return None
    if len(starts) < 3:
        return None
    a = attention_curve(curves)
    shares = network_activation_shares(curves, starts)
    r = lambda x: [round(float(v), 4) for v in x]  # noqa: E731
    return {"status": "ok", "timestamps_s": starts, "networks": {n: {"label": NETWORK_LABELS[n], "curve": r(curves[n])} for n in NETWORKS},
            "attention_curve": r(a), "attention_networks": [NETWORK_LABELS[n] for n in ATTENTION_NETWORKS], "peaks": hook_peaks(a, starts, 3),
            "dominance": shares, "reading": network_reading(shares),
            "note": BRAIN_NOTE}
