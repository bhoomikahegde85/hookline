"""Per-frame visual measurements on the proxy and per-segment aggregation.

Measurements are descriptive. More cuts, more motion or higher sharpness are
not treated as better or worse content.

Per frame (proxy resolution, grayscale unless noted):
- brightness: mean luma, 0-255
- contrast: luma standard deviation, 0-255
- sharpness: variance of the Laplacian (scale-dependent; comparable across proxies only;
  inflated by noise)
- noise: Immerkaer fast noise-sigma estimate, 0-255 scale (fine texture and small text also raise it)
- motion: mean absolute luma difference to the previous frame after a light blur
  (pixel change: conflates object/camera motion with lighting changes and noise)
- hist_change: Bhattacharyya distance between HSV colour histograms of consecutive frames, 0-1
- struct_change: 1 - max(NCC, 0) between consecutive grayscale thumbnails, 0-1
  (catches layout changes with the same colours, e.g. text cards on one background)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .media import find_tool
from .status import AVAILABLE, NOT_APPLICABLE, UNCERTAIN, expects_data, window_status

NAN = float("nan")


@dataclass
class FrameSeries:
    times: np.ndarray
    brightness: np.ndarray
    contrast: np.ndarray
    sharpness: np.ndarray
    noise: np.ndarray
    motion: np.ndarray  # NaN for the first frame
    hist_change: np.ndarray  # NaN for the first frame
    struct_change: np.ndarray  # NaN for the first frame
    width: int
    height: int
    fps: int
    errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def change(self) -> np.ndarray:
        """Combined frame-to-frame dissimilarity: max of colour and structural change."""
        return np.fmax(self.hist_change, self.struct_change)


class FrameReader:
    """Iterate BGR frames from the proxy through an FFmpeg pipe (constant memory).

    Decoder messages are collected in ``errors`` once iteration finishes.
    """

    def __init__(self, proxy: Path, width: int, height: int):
        self.proxy, self.width, self.height = proxy, width, height
        self.errors: list[str] = []

    def __iter__(self):
        ffmpeg = find_tool("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found")
        cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-i", str(self.proxy), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        frame_bytes = self.width * self.height * 3
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
            proc.stdout.close()
            err = proc.stderr.read().decode(errors="replace").strip()
            if proc.wait() != 0 or err:
                self.errors.append(f"decoder errors while reading proxy: {err[-300:] or proc.returncode}")


def _hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])


def _thumb(gray: np.ndarray, width: int) -> np.ndarray:
    h, w = gray.shape
    t = cv2.resize(gray, (width, max(1, round(width * h / w))), interpolation=cv2.INTER_AREA).astype(np.float64)
    return t - t.mean()


_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _noise_sigma(gray: np.ndarray) -> float:
    h, w = gray.shape
    if h < 3 or w < 3:
        return NAN
    conv = cv2.filter2D(gray.astype(np.float64), -1, _NOISE_KERNEL)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def _struct_change(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-6:  # at least one flat frame: no structure to compare (colour change is caught by histograms)
        return 0.0
    return float(1.0 - max((a * b).sum() / denom, 0.0))


def analyze_proxy(proxy: str | Path, frame_times: list[float], width: int, height: int, cfg: PipelineConfig) -> FrameSeries:
    k = cfg.motion_blur_kernel | 1
    rows: list[tuple[float, ...]] = []
    prev = None
    reader = FrameReader(Path(proxy), width, height)
    for bgr in reader:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cur = (cv2.GaussianBlur(gray, (k, k), 0).astype(np.int16), _hist(bgr), _thumb(gray, cfg.struct_thumb_width))
        if prev is None:
            motion = hist_change = struct_change = NAN
        else:
            motion = float(np.abs(cur[0] - prev[0]).mean())
            hist_change = float(cv2.compareHist(prev[1], cur[1], cv2.HISTCMP_BHATTACHARYYA))
            struct_change = _struct_change(prev[2], cur[2])
        rows.append((
            float(gray.mean()), float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_64F).var()), _noise_sigma(gray),
            motion, hist_change, struct_change,
        ))
        prev = cur

    errors = list(reader.errors)
    n = min(len(rows), len(frame_times))
    if len(rows) != len(frame_times):
        errors.append(f"decoded {len(rows)} frames but proxy reports {len(frame_times)} timestamps; using {n}")
    arr = np.array(rows[:n], dtype=float).reshape(-1, 7)
    return FrameSeries(
        times=np.array(frame_times[:n], dtype=float),
        brightness=arr[:, 0],
        contrast=arr[:, 1],
        sharpness=arr[:, 2],
        noise=arr[:, 3],
        motion=arr[:, 4],
        hist_change=arr[:, 5],
        struct_change=arr[:, 6],
        width=width,
        height=height,
        fps=cfg.proxy_fps,
        errors=errors,
    )


def detect_cuts(series: FrameSeries | None, cfg: PipelineConfig) -> list[float]:
    """Times of frames that start a new shot.

    A frame is a cut candidate if its colour histogram jumps (``scene_cut_threshold``)
    or its structural change is high and spikes above the local median (so
    sustained camera motion or noise is not counted). Runs of adjacent
    candidates (dip-to-black, flash or whip transitions) count as one cut at the
    frame with the largest change. Slow dissolves are not detected.
    """
    if series is None or len(series) < 2:
        return []
    hist = np.nan_to_num(series.hist_change, nan=0.0)
    struct = np.nan_to_num(series.struct_change, nan=0.0)
    change = np.fmax(hist, struct)
    c = cfg.cut_context_frames
    candidates = []
    for i in range(1, len(series)):
        neighbours = np.r_[struct[max(1, i - c) : i], struct[i + 1 : i + 1 + c]]
        baseline = float(np.median(neighbours)) if neighbours.size else 0.0
        structural = struct[i] >= cfg.struct_cut_threshold and struct[i] - baseline >= cfg.struct_cut_margin
        if hist[i] >= cfg.scene_cut_threshold or structural:
            candidates.append(i)
    runs: list[list[int]] = []
    for i in candidates:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return [round(float(series.times[max(run, key=lambda j: change[j])]), 6) for run in runs]


def _mean(a: np.ndarray) -> float | None:
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else None


def _max(a: np.ndarray) -> float | None:
    a = a[~np.isnan(a)]
    return float(a.max()) if a.size else None


def frame_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times >= start - 1e-6) & (times < end - 1e-6)


def aggregate_visual(
    series: FrameSeries | None,
    start: float,
    end: float,
    stream: str,
    expected_span: list[float] | None,
    cfg: PipelineConfig,
    cut_times: list[float] = (),
) -> dict:
    """Visual values for one segment. Values stay None unless measured.

    Frames detected as cuts are excluded from motion (a cut is not movement).
    """
    out: dict = {
        "visual_status": stream,
        "visual_coverage": None,
        "visual_frames": None,
        "brightness_mean": None,
        "contrast_std": None,
        "sharpness_lapvar": None,
        "noise_sigma": None,
        "motion_mean": None,
        "motion_max": None,
        "scene_change_max": None,
        "visual_warnings": [],
        "feature_status": {},
    }
    if series is None or stream != AVAILABLE:
        return out

    mask = frame_mask(series.times, start, end)
    n = int(mask.sum())
    expected = (end - start) * series.fps
    coverage = min(1.0, n / expected) if expected > 0 else 0.0
    status = window_status(stream, coverage, expects_data(start, end, expected_span, cfg.min_coverage), cfg.min_coverage)
    out.update(visual_status=status, visual_coverage=round(coverage, 4), visual_frames=n)
    if n == 0:
        return out

    is_cut = np.isin(np.round(series.times, 6), np.round(np.asarray(cut_times, dtype=float), 6))
    motion = series.motion[mask & ~is_cut]
    out["brightness_mean"] = _mean(series.brightness[mask])
    out["contrast_std"] = _mean(series.contrast[mask])
    out["sharpness_lapvar"] = _mean(series.sharpness[mask])
    out["noise_sigma"] = _mean(series.noise[mask])
    out["motion_mean"] = _mean(motion)
    out["motion_max"] = _max(motion)
    out["scene_change_max"] = _max(series.change[mask])
    if out["motion_mean"] is None:
        reason = "only cut frames in window" if (mask & is_cut).any() else "no previous frame"
        out["feature_status"]["motion_mean"] = out["feature_status"]["motion_max"] = f"{NOT_APPLICABLE}: {reason}"
    if out["scene_change_max"] is None:
        out["feature_status"]["scene_change_max"] = f"{NOT_APPLICABLE}: no previous frame"

    flags = out["visual_warnings"]
    b, c, s = out["brightness_mean"], out["contrast_std"], out["sharpness_lapvar"]
    if b < cfg.dark_brightness:
        flags.append("dark")
    elif b > cfg.bright_brightness:
        flags.append("overexposed")
    if c < cfg.low_contrast_std:
        flags.append("low_contrast")
    elif s < cfg.low_sharpness_lapvar:
        flags.append("low_sharpness")  # only meaningful when there is contrast to be sharp
    noise = out["noise_sigma"]
    if noise is not None and noise > cfg.noise_sigma_warn:
        flags.append("possible_noise")
        out["feature_status"]["sharpness_lapvar"] = "uncertain: noise inflates Laplacian variance"
    if status == UNCERTAIN:
        flags.append("partial_visual_coverage")
    return out


def opening_visual(series: FrameSeries | None, cuts: list[float], end: float) -> dict | None:
    if series is None:
        return None
    mask = frame_mask(series.times, 0.0, end)
    if not mask.any():
        return None
    opening_cuts = [c for c in cuts if c < end - 1e-6]
    is_cut = np.isin(np.round(series.times, 6), np.round(np.asarray(cuts, dtype=float), 6))
    return {
        "frames": int(mask.sum()),
        "brightness_mean": _mean(series.brightness[mask]),
        "contrast_std": _mean(series.contrast[mask]),
        "sharpness_lapvar": _mean(series.sharpness[mask]),
        "noise_sigma": _mean(series.noise[mask]),
        "motion_mean": _mean(series.motion[mask & ~is_cut]),
        "scene_cuts": len(opening_cuts),
        "scene_cut_times": opening_cuts,
        "first_scene_cut_s": opening_cuts[0] if opening_cuts else None,
    }
