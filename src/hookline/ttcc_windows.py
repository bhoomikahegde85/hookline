"""Overlapping windows for TTCC and a merge into one continuous predicted retention curve.

Why windows. The model's limit is not 30 s: its head emits 60 values (T_MAX = 60), training used 5 to 60 s horizons, and the
backbone has a 32,768-token context (about 54 s of video at ~600 tokens/s; longer ads were dropped from training). The practical limit
here is GPU memory with ``sdpa`` attention (measured: 25 s fits an L40S, 37 s does not), and nothing is trained beyond 60 s. So long
videos are predicted in windows of ``window_s`` seconds (configurable, up to 60) that overlap by ``window_s - stride_s``.

How windows are merged. Each window is a *standalone* prediction R_k(tau), tau = 0..T_k, R_k(0) = 1 by convention: the model treats it as an
ad that starts at the window's start. Chaining windows as conditional survival would repeat the steep "opening drop" of an ad at every
window start, an artifact. Instead the merge works in hazard space, lambda(t) = -ln(R(t) / R(t-1)) (the share of the remaining viewers
lost in second t):
- the first window (it starts at the real start of the video) counts fully;
- every later window's first ``warmup_s`` seconds are down-weighted (a quadratic ramp, so the first seconds count very little where another window covers them), because they describe a fresh start, not the middle of a video;
- the merged hazard at second t is the weighted mean of the hazards of the windows covering t;
- the curve is rebuilt as R(t) = exp(-sum of merged hazards), so it is continuous, monotone non-increasing and has no jumps at boundaries.

What this is not. A stitched curve is an approximation, not a native single-pass prediction: window k's hazards describe viewers who start at its
start, while the merged curve describes viewers who started at 0. It is unvalidated against audience data, like the single-window curve.
``diagnostics`` reports how much overlapping windows disagree, so the stitching can be judged.
"""

from __future__ import annotations

import math

T_MIN, T_MAX = 5, 60
MIN_WEIGHT = 0.02  # keeps a window's very first seconds usable when no other window covers them (e.g. after a failed neighbour)


# Auto window size by video length (seconds of video -> window length), used when window_mode="auto". Up to the single-pass
# limit the whole video is one window; longer videos get shorter windows (finer local reading, less memory per pass), never under
# the model's 5 s minimum. Chosen from the model's structure, not tuned on real reels.
AUTO_TIERS = ((45, 20), (90, 15), (10**9, 10))


def auto_window_s(n: int, max_window_s: int, min_window_s: int = T_MIN) -> int:
    """Window length for an ``n``-second video: ``n`` itself when it fits ``max_window_s``, else the tier's length within [min_window_s, max_window_s]."""
    if n <= max_window_s:
        return n
    size = next(w for limit, w in AUTO_TIERS if n <= limit)
    return max(min_window_s, min(size, max_window_s))


def window_params(n: int, window_s: int, stride_s: int, warmup_s: float, mode: str, size: int | None = None) -> tuple[int, int, float]:
    """(window, stride, warm-up) in seconds. ``window_s`` / ``stride_s`` are the configured maximum window and its stride; ``size`` overrides
    the window (an out-of-memory retry). The stride keeps the configured overlap ratio; in auto mode the warm-up is at most a quarter of a window."""
    W = int(size) if size else (auto_window_s(n, window_s) if mode == "auto" else window_s)
    stride = max(1, min(W, round(W * stride_s / window_s)))
    return W, stride, (min(warmup_s, max(1.0, W / 4)) if mode == "auto" else warmup_s)


def plan_windows(n: int, window_s: int, stride_s: int) -> list[tuple[int, int]]:
    """Window (start, end) pairs in whole seconds covering [0, n].

    A video of at most ``window_s`` seconds is one window ``(0, n)`` (a short final chunk is never produced). Otherwise windows are
    ``window_s`` long and start every ``stride_s`` seconds; the last window is shifted back so it ends exactly at ``n`` and is a full
    window rather than a short remainder."""
    if n < T_MIN:
        raise ValueError(f"video of {n} s is shorter than the model's {T_MIN} s minimum")
    if not (T_MIN <= window_s <= T_MAX):
        raise ValueError(f"window_s must be {T_MIN}..{T_MAX} s")
    if not (1 <= stride_s <= window_s):
        raise ValueError("stride_s must be between 1 s and window_s")
    if n <= window_s:
        return [(0, n)]
    starts = [0]
    while starts[-1] + window_s < n:
        starts.append(min(starts[-1] + stride_s, n - window_s))
    return [(s, s + window_s) for s in starts]


def hazards(retention: list[float]) -> list[float]:
    """Per-second hazards lambda(tau), tau = 1..T, from R(0..T) (R(0) = 1)."""
    out = []
    for prev, cur in zip(retention, retention[1:]):
        if not (prev > 0 and cur > 0):
            raise ValueError("retention values must be positive to take hazards")
        out.append(max(0.0, -math.log(cur / prev)))
    return out


def weight(tau: int, window_index: int, warmup_s: float) -> float:
    """Weight of window ``window_index`` at local second ``tau`` (1-based). Window 0 always counts fully."""
    if window_index == 0 or warmup_s <= 0:
        return 1.0
    return max(MIN_WEIGHT, min(1.0, tau / warmup_s) ** 2)


def merge_windows(windows: list[dict], n: int, warmup_s: float = 5.0) -> dict:
    """Merge window predictions into one curve R(0..M) on the original timeline.

    ``windows``: dicts with ``start`` and ``end`` (whole seconds on the original timeline) and ``retention`` = R_k(0..end-start).
    Window order is by start; the first window must start at 0. ``M`` (``covered_until``) is the last second reachable without a gap:
    seconds nobody covers (a failed window) end the curve; nothing is interpolated across a gap."""
    ws = sorted(windows, key=lambda w: w["start"])
    if not ws or ws[0]["start"] != 0:
        return {"retention": [1.0], "covered_until": 0, "hazard": [], "coverage": [], "diagnostics": {"note": "no window starts at 0: nothing to anchor R(0) = 1"}}
    hz = [hazards(w["retention"]) for w in ws]
    for w, h in zip(ws, hz):
        if len(h) != w["end"] - w["start"]:
            raise ValueError(f"window {w['start']}-{w['end']} has {len(h)} hazards for {w['end'] - w['start']} s")
    merged: list[float] = []
    coverage: list[list[int]] = []
    for t in range(1, n + 1):  # second t covers (t-1, t]
        num = den = 0.0
        covering = []
        for k, (w, h) in enumerate(zip(ws, hz)):
            if w["start"] < t <= w["end"]:
                tau = t - w["start"]
                wt = weight(tau, k, warmup_s)
                num += wt * h[tau - 1]
                den += wt
                covering.append(k)
        if not covering:
            break
        merged.append(num / den)
        coverage.append(covering)
    retention = [1.0]
    for lam in merged:
        retention.append(retention[-1] * math.exp(-lam))
    return {"retention": retention, "covered_until": len(merged), "hazard": merged, "coverage": coverage,
            "diagnostics": _diagnostics(ws, hz, warmup_s)}


def _diagnostics(ws: list[dict], hz: list[list[float]], warmup_s: float) -> dict:
    """How much do overlapping windows disagree (hazard difference per second, after the later window's warm-up)?"""
    pairs = []
    for a in range(len(ws)):
        for b in range(a + 1, len(ws)):
            lo, hi = ws[b]["start"], min(ws[a]["end"], ws[b]["end"])
            diffs = [abs(hz[a][t - 1 - ws[a]["start"]] - hz[b][t - 1 - ws[b]["start"]])
                     for t in range(lo + 1, hi + 1) if (t - ws[b]["start"]) > warmup_s]
            if diffs:
                pairs.append({"windows": [a, b], "overlap_s": [lo, hi], "seconds_compared": len(diffs),
                              "mean_abs_hazard_difference": sum(diffs) / len(diffs), "max_abs_hazard_difference": max(diffs)})
    means = [p["mean_abs_hazard_difference"] for p in pairs]
    return {"overlap_pairs": pairs, "mean_abs_hazard_difference": sum(means) / len(means) if means else None,
            "note": "hazard = share of remaining viewers lost per second; differences are between standalone window predictions after the later window's warm-up"}


def summarize_curve(retention: list[float]) -> dict:
    """The summary stored with a curve (same fields as a single-pass prediction)."""
    T = len(retention) - 1
    drops = [(t, retention[t - 1] - retention[t]) for t in range(1, T + 1)]
    steepest = max(drops, key=lambda d: d[1])
    return {"retention_at_horizon": retention[-1],
            "steepest_one_second_drop": {"from_s": steepest[0] - 1, "to_s": steepest[0], "drop": steepest[1]},
            "retention_at_s": {str(t): retention[t] for t in (1, 3, 5, 10, 15, 30, 60) if t <= T}}
