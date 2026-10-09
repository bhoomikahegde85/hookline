"""Librosa audio features on hand-made WAVs (no FFmpeg, no network)."""

from __future__ import annotations

import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")

from hookline.config import LibrosaConfig  # noqa: E402
from hookline.context import MediaContext  # noqa: E402
from hookline.librosa_audio import (  # noqa: E402
    LIBROSA_COLUMNS,
    extract_librosa,
    librosa_key,
    librosa_window,
    low_energy_intervals,
)

SR = 16000
CFG = LibrosaConfig()


def write_wav(path: Path, x: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def segs(duration: float, step: float = 0.5, audio: str = "available") -> list[dict]:
    n = int(np.ceil(duration / step - 1e-9))
    return [{"segment_index": i, "start_s": round(i * step, 6), "end_s": round(min((i + 1) * step, duration), 6),
             "audio_status": audio} for i in range(n)]


def ctx(tmp_path: Path, x: np.ndarray | None, duration: float, offset: float = 0.0, stream: str = "available",
        **kw) -> MediaContext:
    path = write_wav(tmp_path / "a.wav", x) if x is not None else None
    base = dict(video_id="v", source_path="v.mp4", source_sha256="sha", prep_key="prep", duration=duration,
                audio_stream=stream, audio_path=str(path) if path else None, audio_offset=offset, segments=segs(duration))
    return MediaContext(**(base | kw))


def tone_then_silence(seconds_tone: float, seconds_silence: float) -> np.ndarray:
    t = np.arange(int(SR * seconds_tone)) / SR
    return np.concatenate([0.3 * np.sin(2 * np.pi * 1000 * t), np.zeros(int(SR * seconds_silence))])


def test_features_map_onto_segments_on_the_original_clock(tmp_path) -> None:
    art = extract_librosa(ctx(tmp_path, tone_then_silence(2.0, 2.0), 4.0), CFG)
    assert art["status"] == "available" and art["measured"]
    assert art["frames"]["count"] == 1 + 4 * SR // CFG.hop and art["frames"]["hop_s"] == pytest.approx(0.01)
    rows = {r["segment_index"]: r for r in art["segments"]}
    assert [r["librosa_frames"] for r in art["segments"]][:3] == [50, 50, 50]  # centred 10 ms frames per 0.5 s
    tone, silent = rows[1], rows[6]
    assert tone["librosa_centroid_hz"] == pytest.approx(1000, rel=0.05)
    assert tone["librosa_rolloff_hz"] > tone["librosa_centroid_hz"] and tone["librosa_zcr"] == pytest.approx(0.125, rel=0.1)
    assert tone["librosa_low_energy_ratio"] == 0.0 and tone["librosa_rms_db"] == pytest.approx(-13.5, abs=0.5)
    # Silence: measured as low energy, spectral shape left blank (never 0).
    assert silent["librosa_low_energy_ratio"] == 1.0 and silent["librosa_centroid_hz"] is None and silent["librosa_zcr"] is None
    assert any("spectral shape=not_applicable" in n for n in silent["notes"])
    (iv,) = art["low_energy_intervals"]
    assert iv["start_s"] == pytest.approx(2.0, abs=0.03) and iv["end_s"] == pytest.approx(4.0, abs=0.01)
    assert art["global"]["tempo_status"] == "not_applicable"  # shorter than 6 s


def test_audio_offset_shifts_frames_onto_the_video_timeline(tmp_path) -> None:
    # Audio starts 1 s into the video: the tone (first 1 s of the WAV) is at 1-2 s.
    c = ctx(tmp_path, tone_then_silence(1.0, 1.0), 3.0, offset=1.0)
    c.segments[0]["audio_status"] = c.segments[1]["audio_status"] = "absent"
    art = extract_librosa(c, CFG)
    rows = {r["segment_index"]: r for r in art["segments"]}
    assert rows[0]["librosa_frames"] == 0 and rows[2]["librosa_centroid_hz"] == pytest.approx(1000, rel=0.05)
    assert rows[5]["librosa_low_energy_ratio"] == 1.0
    # Centred 32 ms frames straddle a boundary by half a window: 2.00 and 2.01 s still hear the tone.
    assert rows[4]["librosa_low_energy_ratio"] == pytest.approx(0.96)
    assert art["frames"]["first_s"] == 1.0
    w, _ = librosa_window(art, c.segments[0])
    assert w["librosa_status"] == "absent" and all(w[k] is None for k in LIBROSA_COLUMNS[1:])


def test_digital_silence_and_very_short_clip(tmp_path) -> None:
    art = extract_librosa(ctx(tmp_path, np.zeros(SR), 1.0), CFG)
    assert art["status"] == "available" and art["global"]["low_energy_ratio"] == 1.0
    assert all(r["librosa_centroid_hz"] is None for r in art["segments"])
    assert art["global"]["centroid_hz_mean"] is None  # not 0

    short = tmp_path / "short"
    short.mkdir()
    art = extract_librosa(ctx(short, tone_then_silence(0.3, 0.0), 0.3), CFG)
    (row,) = art["segments"]
    assert art["frames"]["count"] == 1 + int(0.3 * SR) // CFG.hop and row["librosa_frames"] == art["frames"]["count"]
    assert row["librosa_centroid_hz"] == pytest.approx(1000, rel=0.1)


def test_missing_audio_failures_and_uncertain_streams(tmp_path) -> None:
    absent = extract_librosa(ctx(tmp_path, None, 2.0, stream="absent"), CFG)
    assert (absent["status"], absent["measured"], absent["reason"]) == ("absent", False, "no audio stream")
    unknown = extract_librosa(ctx(tmp_path, None, 2.0, stream="uncertain"), CFG)
    assert unknown["status"] == "uncertain"
    missing = extract_librosa(ctx(tmp_path, None, 2.0), CFG)
    assert missing["status"] == "failed" and "WAV missing" in missing["reason"]
    corrupt = tmp_path / "bad.wav"
    corrupt.write_bytes(b"RIFF\x00\x00garbage")
    broken = extract_librosa(ctx(tmp_path, None, 2.0, audio_path=str(corrupt)), CFG)
    assert broken["status"] == "failed" and "extraction failed" in broken["reason"]
    empty = extract_librosa(ctx(tmp_path, np.zeros(0), 2.0), CFG)
    assert empty["status"] == "failed" and "no samples" in empty["reason"]
    for art in (absent, missing, broken):
        w, _ = librosa_window(art, {"segment_index": 0, "start_s": 0.0, "end_s": 0.5, "audio_status": "available"})
        assert w["librosa_status"] == art["status"] and all(w[k] is None for k in LIBROSA_COLUMNS[1:])
    skipped, notes = librosa_window(None, {"segment_index": 0, "start_s": 0.0, "end_s": 0.5})
    assert skipped["librosa_status"] == "uncertain" and "stage skipped" in notes[0]


def test_failed_audio_window_keeps_values_blank(tmp_path) -> None:
    c = ctx(tmp_path, tone_then_silence(2.0, 0.0), 2.0)
    art = extract_librosa(c, CFG)
    w, _ = librosa_window(art, c.segments[1] | {"audio_status": "failed"})
    assert w["librosa_status"] == "failed" and w["librosa_rms_db"] is None
    w, _ = librosa_window(art, c.segments[1])
    assert w["librosa_status"] == "available" and w["librosa_rms_db"] is not None


def test_tempo_only_when_beats_land_on_onsets(tmp_path) -> None:
    rng = np.random.default_rng(0)

    def hits(times):
        y = 0.01 * rng.standard_normal(SR * 10)
        for b in times:
            i = int(b * SR)
            y[i:i + 400] += np.hanning(400)[: len(y[i:i + 400])] * 0.8 * rng.standard_normal(400)[: len(y[i:i + 400])]
        return y

    steady = tmp_path / "steady"
    steady.mkdir()
    art = extract_librosa(ctx(steady, hits(np.arange(0, 10, 0.5)), 10.0), CFG)
    assert art["global"]["tempo_status"] == "available" and art["global"]["tempo_bpm"] == pytest.approx(120, abs=3)
    random = tmp_path / "random"
    random.mkdir()
    art = extract_librosa(ctx(random, hits(np.sort(rng.uniform(0, 9.9, 20))), 10.0), CFG)
    assert art["global"]["tempo_status"] == "uncertain" and art["global"]["tempo_bpm"] is None


def test_low_energy_intervals_respect_minimum_length() -> None:
    times = np.arange(100) * 0.01
    low = np.zeros(100, bool)
    low[10:20] = True  # 0.1 s: too short
    low[50:90] = True  # 0.4 s
    out = low_energy_intervals(times, low, 0.01, 0.3, 1.0)
    assert out == [{"interval_id": 0, "start_s": 0.5, "end_s": 0.9}]


def test_cache_key_tracks_config_versions_and_segments(tmp_path, monkeypatch) -> None:
    c = ctx(tmp_path, tone_then_silence(1.0, 0.0), 1.0)
    key = librosa_key(c, CFG)
    assert librosa_key(c, CFG) == key
    assert librosa_key(c, replace(CFG, silence_dbfs=-60.0)) != key
    assert librosa_key(replace(c, prep_key="other"), CFG) != key
    assert librosa_key(replace(c, segments=segs(1.0, step=0.25)), CFG) != key
    import hookline.librosa_audio as la
    monkeypatch.setattr(la, "library_versions", lambda: {"librosa": "0.0-test", "numpy": "x"})
    assert librosa_key(c, CFG) != key
