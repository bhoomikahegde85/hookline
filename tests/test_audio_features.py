from __future__ import annotations

import numpy as np
import pytest

from hookline.audio import aggregate_audio, analyze_audio, opening_audio
from hookline.config import PipelineConfig
from hookline.status import ABSENT, AVAILABLE, FAILED, UNCERTAIN

SR = 16000
CFG = PipelineConfig()


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_digital_silence_is_measured_not_missing() -> None:
    s = analyze_audio(silence(2.0), SR, 0.0, CFG)
    a = aggregate_audio(s, 0.0, 0.5, AVAILABLE, [0.0, 2.0], CFG)
    assert a["audio_status"] == AVAILABLE
    assert a["rms"] == 0.0 and a["rms_dbfs"] == CFG.db_floor
    assert a["near_silence_ratio"] == 1.0
    assert a["onset_count"] == 0
    assert "digital_silence" in a["audio_warnings"]


def test_absent_audio_keeps_values_missing() -> None:
    a = aggregate_audio(None, 0.0, 0.5, ABSENT, None, CFG)
    assert a["audio_status"] == ABSENT
    for k in ("rms", "rms_dbfs", "near_silence_ratio", "onset_count", "clipping_ratio", "audio_coverage"):
        assert a[k] is None


def test_tone_level_matches_expected_dbfs() -> None:
    s = analyze_audio(tone(1.0, amp=0.5), SR, 0.0, CFG)
    a = aggregate_audio(s, 0.0, 0.5, AVAILABLE, [0.0, 1.0], CFG)
    assert a["rms_dbfs"] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.1)
    assert a["near_silence_ratio"] == 0.0
    assert a["clipping_ratio"] == 0.0


def test_clipping_is_flagged() -> None:
    x = np.clip(tone(1.0, amp=2.0), -1.0, 1.0)
    a = aggregate_audio(analyze_audio(x, SR, 0.0, CFG), 0.0, 0.5, AVAILABLE, [0.0, 1.0], CFG)
    assert a["clipping_ratio"] > 0.1
    assert "possible_clipping" in a["audio_warnings"]


def test_onsets_at_sound_starts_not_at_abrupt_stops_or_steady_tone() -> None:
    bursts = np.concatenate([silence(0.5), tone(0.3), silence(0.7), tone(0.3), silence(0.7)])
    s = analyze_audio(bursts, SR, 0.0, CFG)
    assert s.onset_times == pytest.approx([0.5, 1.5], abs=0.03)
    steady = analyze_audio(tone(2.0), SR, 0.0, CFG)
    assert len(steady.onset_times) == 0


def test_click_inside_a_steady_tone_is_an_onset() -> None:
    x = tone(2.0)
    x[SR : SR + 10] += 0.5
    assert analyze_audio(x, SR, 0.0, CFG).onset_times == pytest.approx([1.0], abs=0.03)


def test_offset_maps_samples_to_timeline_and_marks_uncovered_windows() -> None:
    # Audio stream starts 1.0 s into the timeline.
    s = analyze_audio(tone(1.0), SR, 1.0, CFG)
    before = aggregate_audio(s, 0.0, 0.5, AVAILABLE, [1.0, 2.0], CFG)
    assert before["audio_status"] == ABSENT and before["rms"] is None
    inside = aggregate_audio(s, 1.0, 1.5, AVAILABLE, [1.0, 2.0], CFG)
    assert inside["audio_status"] == AVAILABLE and inside["audio_coverage"] == 1.0
    straddle = aggregate_audio(s, 0.7, 1.1, AVAILABLE, [1.0, 2.0], CFG)
    assert straddle["audio_status"] == UNCERTAIN and straddle["audio_coverage"] == pytest.approx(0.25)
    assert "partial_audio_coverage" in straddle["audio_warnings"]


def test_missing_samples_inside_expected_span_are_failures() -> None:
    # Container says audio lasts 3 s, but only 1 s decoded (truncated file).
    s = analyze_audio(tone(1.0), SR, 0.0, CFG)
    a = aggregate_audio(s, 2.0, 2.5, AVAILABLE, [0.0, 3.0], CFG)
    assert a["audio_status"] == FAILED and a["rms"] is None


def test_declared_tail_longer_than_decoded_audio_is_not_a_failure() -> None:
    # Container declares 2.04 s of audio, 1.93 s decodes (AAC padding); last window 2.0-2.4.
    s = analyze_audio(tone(1.93), SR, 0.0, CFG)
    a = aggregate_audio(s, 2.0, 2.4, AVAILABLE, [0.0, 2.04], CFG)
    assert a["audio_status"] == ABSENT and a["rms"] is None


def test_opening_first_non_silent_time() -> None:
    x = np.concatenate([silence(1.2), tone(2.0)])
    o = opening_audio(analyze_audio(x, SR, 0.0, CFG), 5.0, CFG)
    assert o["first_non_silent_s"] == pytest.approx(1.2, abs=0.021)
    assert o["near_silence_ratio"] == pytest.approx(1.2 / 3.2, abs=0.02)
