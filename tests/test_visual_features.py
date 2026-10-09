from __future__ import annotations

import numpy as np

from hookline.config import PipelineConfig
from hookline.status import ABSENT, AVAILABLE, FAILED
from hookline.visual import FrameSeries, aggregate_visual, detect_cuts

CFG = PipelineConfig()
NAN = float("nan")


def series(n: int = 10, **overrides) -> FrameSeries:
    base = {
        "brightness": np.full(n, 120.0),
        "contrast": np.full(n, 50.0),
        "sharpness": np.full(n, 300.0),
        "noise": np.full(n, 0.5),
        "motion": np.r_[NAN, np.full(n - 1, 2.0)],
        "hist_change": np.r_[NAN, np.zeros(n - 1)],
        "struct_change": np.r_[NAN, np.zeros(n - 1)],
    }
    base.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return FrameSeries(times=np.arange(n) / 10, width=2, height=2, fps=10, **base)


def test_adjacent_cut_frames_collapse_to_one_transition() -> None:
    s = series(12, hist_change=[NAN, 0, 0, 0.5, 0.6, 0.9, 0.5, 0, 0, 0, 0.8, 0])
    assert detect_cuts(s, CFG) == [0.5, 1.0]


def test_structural_cut_needs_a_spike_not_sustained_change() -> None:
    sustained = series(12, struct_change=[NAN] + [0.3] * 11)
    assert detect_cuts(sustained, CFG) == []
    spike = series(12, struct_change=[NAN, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0])
    assert detect_cuts(spike, CFG) == [0.6]


def test_absent_or_failed_visual_keeps_values_missing() -> None:
    for status in (ABSENT, FAILED):
        v = aggregate_visual(None, 0.0, 0.5, status, None, CFG)
        assert v["visual_status"] == status
        assert v["brightness_mean"] is None and v["motion_mean"] is None and v["visual_coverage"] is None


def test_first_frame_only_window_marks_motion_not_applicable() -> None:
    v = aggregate_visual(series(), 0.0, 0.1, AVAILABLE, [0.0, 1.0], CFG)
    assert v["brightness_mean"] == 120.0
    assert v["motion_mean"] is None
    assert v["feature_status"]["motion_mean"].startswith("not_applicable")


def test_cut_frame_is_excluded_from_motion() -> None:
    s = series(motion=[NAN, 2, 2, 2, 2, 90, 2, 2, 2, 2])
    v = aggregate_visual(s, 0.5, 1.0, AVAILABLE, [0.0, 1.0], CFG, cut_times=[0.5])
    assert v["motion_max"] == 2.0
    assert v["scene_change_max"] == 0.0


def test_quality_warnings() -> None:
    dark_noisy = series(brightness=np.full(10, 20.0), noise=np.full(10, 5.0))
    v = aggregate_visual(dark_noisy, 0.0, 0.5, AVAILABLE, [0.0, 1.0], CFG)
    assert "dark" in v["visual_warnings"] and "possible_noise" in v["visual_warnings"]
    assert v["feature_status"]["sharpness_lapvar"].startswith("uncertain")
    flat = series(contrast=np.full(10, 3.0), sharpness=np.full(10, 1.0))
    w = aggregate_visual(flat, 0.0, 0.5, AVAILABLE, [0.0, 1.0], CFG)["visual_warnings"]
    assert "low_contrast" in w and "low_sharpness" not in w  # blur is not judged without contrast


def test_missing_frames_inside_stream_span_are_failures() -> None:
    v = aggregate_visual(series(10), 2.0, 2.5, AVAILABLE, [0.0, 3.0], CFG)
    assert v["visual_status"] == FAILED and v["brightness_mean"] is None
