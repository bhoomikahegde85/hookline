"""'What if I make this edit?': apply accepted trims to a video and compare TTCC's predicted curves of the original and the edited video.

Pure and local: this module never calls TTCC or any network service. Running TTCC on the edited file is the caller's job; the comparison here
only sets two predicted retention curves (R(0..T) on a 1 s grid, a pretrained model's prediction, not measured retention) side by side.

The verdict is deliberately conservative: it is decided only on the predicted hold at 3, 5 and 10 s (each on its own timeline from the start of
each video) and only when the difference is clearly above the model's re-encoding noise. TTCC was shown to respond to overall visual look more
than to the order of events, so a small difference from trimming is expected to be within noise. Nothing here is audience data.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from .media import find_tool

SCHEMA = "hookline.whatif"
NOISE_FLOOR = 0.0033        # re-encoding noise of TTCC's R(t), as measured in the sanity check
VERDICT_NOISE_FACTOR = 3    # a verdict needs |delta| above this many noise floors on most hold metrics
HOLD_SECONDS = (3, 5, 10)
EPS = 0.02                  # ranges closer than this are merged / shorter than this are dropped
FFMPEG_TIMEOUT_S = 900
KEYS = (("start_s", "end_s"), ("start", "end"), ("a", "b"), ("from", "to"))

CAVEAT_PREDICTION = "Model prediction, not audience data: nothing here was measured on real viewers."
CAVEAT_SENSITIVITY = ("TTCC was shown to respond to overall visual look more than to the order of events (a measured sanity check: black frames "
                      "moved R(3) about 12x re-encoding noise, shuffled 1-second chunks only about 2x), so small differences from trimming "
                      "may be within noise.")
CAVEAT_STITCHED = "Stitched curves are approximate: windows of the video were predicted separately and joined, so the joins are not exact."
CAVEAT_SCOPE = "The removal was applied to this video only, and nothing is known about how real audiences would react to the edit."
CAVEAT_END = "Retention at the end is not like-for-like when the videos differ in length: the edited video is shorter, so fewer seconds have passed."
CAVEAT_SHORT = "No hold metric (3, 5 or 10 s) was available on both curves, so no verdict was made."


def _pair(t: dict) -> tuple[float, float] | None:
    for ka, kb in KEYS:
        if ka in t and kb in t:
            try:
                return float(t[ka]), float(t[kb])
            except (TypeError, ValueError):
                return None
    return None


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def normalise_trims(trims: list[dict], duration_s: float, min_keep_s: float = 5.0, max_remove_fraction: float = 0.8
                    ) -> tuple[list[tuple[float, float]], list[str]]:
    """Sorted, clipped, merged ranges to remove, plus warnings. Raises ValueError when nothing valid remains or the result is unsafe."""
    if not duration_s or duration_s <= 0:
        raise ValueError("the video duration is unknown or not positive")
    warnings: list[str] = []
    ranges: list[tuple[float, float]] = []
    for i, t in enumerate(trims or []):
        p = _pair(t) if isinstance(t, dict) else None
        if p is None or not all(x == x and abs(x) != float("inf") for x in p):
            warnings.append(f"trim {i + 1} ignored: needs numeric start_s and end_s")
            continue
        a, b = p
        if a > b:
            a, b = b, a
            warnings.append(f"trim {i + 1}: start was after end, swapped")
        ca, cb = max(0.0, a), min(float(duration_s), b)
        if (ca, cb) != (a, b):
            warnings.append(f"trim {i + 1} ({a:.2f}-{b:.2f} s) clipped to the video: {ca:.2f}-{cb:.2f} s" if cb - ca >= EPS
                            else f"trim {i + 1} ({a:.2f}-{b:.2f} s) is outside the video and was ignored")
        if cb - ca < EPS:
            if (ca, cb) == (a, b):
                warnings.append(f"trim {i + 1} ({a:.2f}-{b:.2f} s) has zero length and was ignored")
            continue
        ranges.append((ca, cb))
    if not ranges:
        raise ValueError("no valid trim remains: every trim was malformed, empty or outside the video")
    merged = _merge(ranges)
    if len(merged) < len(ranges):
        warnings.append("overlapping or adjacent trims were merged")
    removed = sum(b - a for a, b in merged)
    if removed > max_remove_fraction * duration_s:
        raise ValueError(f"these trims remove {removed:.1f} s of {duration_s:.1f} s ({100 * removed / duration_s:.0f} %), above the "
                         f"{100 * max_remove_fraction:.0f} % limit")
    if duration_s - removed < min_keep_s:
        raise ValueError(f"only {duration_s - removed:.1f} s would remain; at least {min_keep_s:g} s must be kept")
    return merged, warnings


def keep_segments(duration_s: float, trims: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The ranges that stay: the complement of ``trims`` within [0, duration_s]."""
    keep, pos = [], 0.0
    for a, b in _merge([(max(0.0, a), min(duration_s, b)) for a, b in trims if b > a]):
        if a - pos >= EPS:
            keep.append((pos, a))
        pos = max(pos, b)
    if duration_s - pos >= EPS:
        keep.append((pos, float(duration_s)))
    return keep


def edited_time(t: float, trims: list[tuple[float, float]]) -> float | None:
    """Original time -> time in the edited video; None if ``t`` falls inside a removed range [a, b)."""
    if t < 0:
        return None
    shift = 0.0
    for a, b in _merge([(a, b) for a, b in trims if b > a]):
        if a <= t < b:
            return None
        if b <= t:
            shift += b - a
    return t - shift


def _fail(what: str, proc) -> RuntimeError:
    lines = [ln for ln in ((getattr(proc, "stderr", "") or "").strip().splitlines()) if ln.strip()]
    return RuntimeError(f"{what} failed (exit code {getattr(proc, 'returncode', '?')}): " + (" | ".join(lines[-5:]) or "no output"))


def _probe(src: Path, run: Callable) -> tuple[float, bool]:
    cmd = [find_tool("ffprobe") or "ffprobe", "-v", "error", "-print_format", "json", "-show_entries", "format=duration:stream=codec_type", str(src)]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"ffprobe could not run: {e}") from e
    if proc.returncode != 0:
        raise _fail("ffprobe", proc)
    try:
        info = json.loads(proc.stdout or "{}")
        duration = float((info.get("format") or {}).get("duration"))
    except (ValueError, TypeError) as e:
        raise RuntimeError("ffprobe did not report a duration for this video") from e
    return duration, any(s.get("codec_type") == "audio" for s in info.get("streams") or [])


def apply_trims(src: Path, trims: list[tuple[float, float]], out: Path, run=subprocess.run) -> Path:
    """Write ``src`` without the ``trims`` ranges to ``out`` (H.264 yuv420p, AAC only if the source has audio). Atomic: tmp file, then replace."""
    src, out = Path(src), Path(out)
    duration, has_audio = _probe(src, run)
    keep = keep_segments(duration, trims)
    if not keep:
        raise ValueError("nothing would remain after the trims")
    parts, labels = [], ""
    for i, (a, b) in enumerate(keep):
        span = f"start={a:.3f}" + ("" if b >= duration - EPS else f":end={b:.3f}")
        parts.append(f"[0:v:0]trim={span},setpts=PTS-STARTPTS[v{i}]")
        labels += f"[v{i}]"
        if has_audio:
            parts.append(f"[0:a:0]atrim={span},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[a{i}]"
    parts.append(f"{labels}concat=n={len(keep)}:v=1:a={int(has_audio)}[vc]" + ("[a]" if has_audio else ""))
    parts.append("[vc]scale=trunc(iw/2)*2:trunc(ih/2)*2[v]")
    tmp = out.with_name(f"{out.stem}.part{out.suffix}")
    cmd = [find_tool("ffmpeg") or "ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(src), "-filter_complex", ";".join(parts), "-map", "[v]"]
    cmd += ["-map", "[a]", "-c:a", "aac"] if has_audio else ["-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(tmp)]
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            proc = run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"ffmpeg could not run: {e}") from e
        if proc.returncode != 0:
            raise _fail("ffmpeg", proc)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("ffmpeg reported success but wrote no output")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _at(curve: list[float], s: int) -> float | None:
    return float(curve[s]) if len(curve) > s else None


def _hazard(curve: list[float]) -> float | None:
    """Mean share of the remaining viewers predicted lost per second."""
    h = [1.0 - curve[t] / curve[t - 1] for t in range(1, len(curve)) if curve[t - 1] > 0]
    return sum(h) / len(h) if h else None


def compare_curves(original: list[float], edited: list[float], trims: list[tuple[float, float]], noise_floor: float = NOISE_FLOOR,
                   original_is_stitched: bool = False, edited_is_stitched: bool = False) -> dict:
    """Original vs edited predicted retention (each R(0..T), start-anchored). A model prediction; see CAVEATS in the output."""
    original, edited = [float(v) for v in original], [float(v) for v in edited]
    metrics: list[dict] = []

    def add(label: str, o: float | None, e: float | None, hold: bool = False) -> None:
        if o is None or e is None:
            return
        d = e - o
        metrics.append({"label": label, "original": o, "edited": e, "delta": d, "within_noise": abs(d) <= noise_floor * VERDICT_NOISE_FACTOR, "_hold": hold})  # same bar as the verdict

    for s in HOLD_SECONDS:
        add(f"predicted hold at {s} s", _at(original, s), _at(edited, s), hold=True)
    if original and edited:
        add("predicted retention at the end of each video", original[-1], edited[-1])
    add("mean predicted hazard per second", _hazard(original), _hazard(edited))

    holds = [m for m in metrics if m["_hold"]]
    bar = noise_floor * VERDICT_NOISE_FACTOR
    up, down = sum(m["delta"] > bar for m in holds), sum(m["delta"] < -bar for m in holds)
    verdict = ("edited curve holds higher" if up * 2 > len(holds) else "edited curve holds lower" if down * 2 > len(holds)
               else "no clear difference")
    for m in metrics:
        del m["_hold"]

    removed = sum(b - a for a, b in trims)
    caveats = [CAVEAT_PREDICTION, CAVEAT_SENSITIVITY]
    if original_is_stitched or edited_is_stitched:
        caveats.append(CAVEAT_STITCHED)
    caveats.append(CAVEAT_SCOPE)
    if not holds:
        caveats.append(CAVEAT_SHORT)
    if removed > 0:
        caveats.append(CAVEAT_END)
    return {"schema": SCHEMA, "trims": [[a, b] for a, b in trims], "removed_s": removed,
            "original_duration_s": max(len(original) - 1, 0), "edited_duration_s": max(len(edited) - 1, 0),
            "metrics": metrics, "verdict": verdict, "caveats": caveats}
