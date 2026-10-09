"""Librosa audio features on the cached mono 16 kHz WAV, mapped onto the existing segments.

Complements (does not replace) the NumPy features in ``audio.py``. Frames use the same
STFT grid as Task 2 (n_fft 512, hop 160 = 10 ms) with centred frames: frame ``i`` is at
``audio offset + i * hop / sr`` on the original playback timeline, and belongs to the
segment containing that time.

Per frame: RMS (dBFS), spectral centroid / bandwidth / 85 % rolloff (Hz), zero-crossing
rate, onset strength (librosa mel spectral flux; relative units). Spectral-shape values
and zero-crossing rate are only taken from frames at or above ``silence_dbfs``: on
near-silent frames they describe noise or nothing, so they are left out, never zero.

Per video: low-energy intervals (runs of frames below ``silence_dbfs`` lasting at least
``low_energy_min_seconds``) and a global tempo, reported only when the audio is long
enough and the tracked beats are steady.

Descriptive only: a louder, brighter or busier window is not "more exciting", and a quiet
window is not "boring". None of this is an engagement or retention measure.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_wav, to_db
from .config import LIBROSA_FEATURE_VERSION, SCHEMA_VERSION, LibrosaConfig, stable_hash
from .context import MediaContext
from .status import ABSENT, AVAILABLE, FAILED, MEASURED_STATUSES, UNCERTAIN

LIBROSA_COLUMNS = [
    "librosa_status", "librosa_frames", "librosa_rms_db", "librosa_low_energy_ratio", "librosa_centroid_hz",
    "librosa_bandwidth_hz", "librosa_rolloff_hz", "librosa_zcr", "librosa_onset_strength",
]
TIMELINE_NOTE = "seconds from start of original playback; frame i at audio offset + i * hop / sr (centred frames)"
SHAPE_KEYS = ("centroid_hz", "bandwidth_hz", "rolloff_hz", "zcr")


def library_versions() -> dict[str, str | None]:
    try:
        import librosa
        version = librosa.__version__
    except Exception:  # librosa not installed: the stage reports failed
        version = None
    return {"librosa": version, "numpy": np.__version__}


def segments_signature(segments: list[dict]) -> list[list[float]]:
    return [[s["segment_index"], s["start_s"], s["end_s"]] for s in segments]


def librosa_key(ctx: MediaContext, cfg: LibrosaConfig) -> str:
    """Source identity (preprocessing key = source bytes + WAV settings) + config + library versions + segment grid."""
    return stable_hash("librosa", ctx.prep_key or ctx.source_sha256 or ctx.source_path, ctx.audio_offset,
                       cfg.cache_dict(), library_versions(), segments_signature(ctx.segments))


def _base(ctx: MediaContext, cfg: LibrosaConfig, key: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "librosa_feature_version": LIBROSA_FEATURE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": ctx.video_id,
        "source_sha256": ctx.source_sha256,
        "cache_key": key,
        "status": FAILED,
        "measured": False,
        "reason": None,
        "versions": library_versions(),
        "config": cfg.to_dict(),
        "timeline": {"convention": TIMELINE_NOTE, "offset_seconds": ctx.audio_offset, "duration_s": ctx.duration},
        "frames": None,
        "global": None,
        "low_energy_intervals": None,
        "segments": [],
        "timing": None,
    }


def frame_features(y: np.ndarray, sr: int, cfg: LibrosaConfig) -> dict[str, np.ndarray]:
    """Per-frame features on a centred frame grid of ``1 + len(y) // hop`` frames."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    kw = {"n_fft": cfg.n_fft, "hop_length": cfg.hop}
    rms = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop)[0]
    spec = np.abs(librosa.stft(y, **kw))
    feats = {
        "rms_db": to_db(rms, cfg.db_floor),
        "centroid_hz": librosa.feature.spectral_centroid(S=spec, sr=sr, **kw)[0],
        "bandwidth_hz": librosa.feature.spectral_bandwidth(S=spec, sr=sr, **kw)[0],
        "rolloff_hz": librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=cfg.rolloff_percent, **kw)[0],
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop)[0],
        "onset_strength": librosa.onset.onset_strength(y=y, sr=sr, n_mels=cfg.n_mels, **kw),
    }
    n = min(len(v) for v in feats.values())
    return {k: np.asarray(v[:n], dtype=np.float64) for k, v in feats.items()}


def low_energy_intervals(times: np.ndarray, low: np.ndarray, frame_s: float, min_s: float, end: float) -> list[dict]:
    out, start = [], None
    for t, is_low in zip(list(times) + [None], list(low) + [False]):
        if is_low and start is None:
            start = t
        elif not is_low and start is not None:
            stop = min(end, (t if t is not None else times[-1] + frame_s))
            if stop - start >= min_s - 1e-9:
                out.append({"interval_id": len(out), "start_s": round(float(start), 3), "end_s": round(float(stop), 3)})
            start = None
    return out


def global_tempo(onset_env: np.ndarray, sr: int, cfg: LibrosaConfig, seconds: float, voiced_ratio: float) -> dict:
    """Tempo only when reliable: long enough audio, not mostly silent, and the tracked beats land
    on detected onsets. (The beat tracker always returns evenly spaced beats, even for random
    onsets, so beat-interval regularity alone says nothing about reliability.)"""
    out: dict[str, Any] = {"tempo_status": None, "tempo_bpm": None, "beat_count": None, "beat_onset_agreement": None,
                           "reason": None}
    if seconds < cfg.tempo_min_seconds:
        return out | {"tempo_status": "not_applicable", "reason": f"audio shorter than {cfg.tempo_min_seconds:g} s"}
    if voiced_ratio < 0.2:
        return out | {"tempo_status": "not_applicable", "reason": "mostly low-energy audio"}
    import librosa

    bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=cfg.hop)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=cfg.hop)
    tol = max(1, int(round(0.05 * sr / cfg.hop)))
    agreement = float(np.mean([np.any(np.abs(onsets - b) <= tol) for b in beats])) if len(beats) and len(onsets) else 0.0
    out.update(beat_count=int(len(beats)), beat_onset_agreement=round(agreement, 3))
    if len(beats) < cfg.tempo_min_beats or agreement < cfg.tempo_min_beat_onset_agreement:
        return out | {"tempo_status": UNCERTAIN, "reason": "beats too few or not aligned with onsets; no reliable tempo"}
    return out | {"tempo_status": AVAILABLE, "tempo_bpm": round(float(np.atleast_1d(bpm)[0]), 1)}


def _mean(x: np.ndarray) -> float | None:
    return round(float(x.mean()), 4) if x.size else None


def segment_values(times: np.ndarray, feats: dict[str, np.ndarray], start: float, end: float, cfg: LibrosaConfig,
                   last: bool) -> tuple[dict, list[str]]:
    mask = (times >= start) & ((times < end) | (last & (times <= end + 1e-9)))
    n = int(mask.sum())
    vals: dict[str, Any] = {"librosa_frames": n}
    notes: list[str] = []
    if not n:
        notes.append("librosa=not_applicable: no frame centred in segment")
        return vals, notes
    rms_db = feats["rms_db"][mask]
    voiced = rms_db >= cfg.silence_dbfs
    vals.update(
        librosa_rms_db=_mean(rms_db),
        librosa_low_energy_ratio=round(float((~voiced).mean()), 4),
        librosa_onset_strength=_mean(feats["onset_strength"][mask]),
    )
    for k in SHAPE_KEYS:
        vals[f"librosa_{k}"] = _mean(feats[k][mask][voiced])
    if not voiced.any():
        notes.append("librosa spectral shape=not_applicable: every frame below silence threshold")
    return vals, notes


def extract_librosa(ctx: MediaContext, cfg: LibrosaConfig) -> dict:
    """Librosa features for one video's cached WAV. Never re-decodes the source."""
    art = _base(ctx, cfg, librosa_key(ctx, cfg))
    if ctx.audio_stream != AVAILABLE:
        art["status"] = ctx.audio_stream
        art["reason"] = {
            ABSENT: "no audio stream",
            UNCERTAIN: "audio presence unknown (inspection failed)",
            FAILED: "audio stream could not be decoded; librosa features not measured",
        }.get(ctx.audio_stream, f"audio stream status {ctx.audio_stream}")
        return art
    if art["versions"]["librosa"] is None:
        art["reason"] = "librosa is not installed (pip install -r requirements-models.txt)"
        return art
    if not ctx.audio_path or not Path(ctx.audio_path).is_file():
        art["reason"] = "cached WAV missing; re-run scripts/run_pipeline.py"
        return art
    t0 = time.perf_counter()
    try:
        y, sr = load_wav(ctx.audio_path)
        if y.size == 0:
            raise ValueError("WAV has no samples")
        feats = frame_features(y, sr, cfg)
    except Exception as exc:
        art["reason"] = f"librosa feature extraction failed: {type(exc).__name__}: {exc}"
        return art
    n = len(feats["rms_db"])
    times = ctx.audio_offset + np.arange(n) * cfg.hop / sr
    audio_end = ctx.audio_offset + y.size / sr
    end = min(audio_end, ctx.duration) if ctx.duration else audio_end
    low = feats["rms_db"] < cfg.silence_dbfs
    try:
        tempo = global_tempo(feats["onset_strength"], sr, cfg, y.size / sr, float((~low).mean()))
    except Exception as exc:
        tempo = {"tempo_status": FAILED, "tempo_bpm": None, "beat_count": None, "beat_onset_agreement": None,
                 "reason": f"beat tracking failed: {type(exc).__name__}: {exc}"}
    rows = []
    last = len(ctx.segments) - 1
    for i, seg in enumerate(ctx.segments):
        vals, notes = segment_values(times, feats, seg["start_s"], seg["end_s"], cfg, last=i == last)
        rows.append({"segment_index": seg["segment_index"], **vals, "notes": notes})
    art.update(
        status=AVAILABLE,
        measured=True,
        frames={"count": n, "hop_s": cfg.hop / sr, "frame_s": cfg.n_fft / sr, "sample_rate": sr,
                "first_s": round(float(times[0]), 4), "last_s": round(float(times[-1]), 4)},
        low_energy_intervals=low_energy_intervals(times, low, cfg.hop / sr, cfg.low_energy_min_seconds, end),
        segments=rows,
        timing={"processing_seconds": round(time.perf_counter() - t0, 3)},
    )
    art["global"] = {
        "audio_seconds": round(y.size / sr, 3),
        "low_energy_ratio": round(float(low.mean()), 4),
        "rms_db_mean": _mean(feats["rms_db"]),
        "centroid_hz_mean": _mean(feats["centroid_hz"][~low]),
        **tempo,
    }
    return art


def librosa_window(art: dict | None, seg: dict) -> tuple[dict, list[str]]:
    """Librosa columns for one merged-timeline row. The window status follows the segment's
    ``audio_status`` (same WAV and coverage rule as the Task 2 audio features)."""
    out: dict[str, Any] = {k: None for k in LIBROSA_COLUMNS}
    if art is None:
        out["librosa_status"] = UNCERTAIN
        return out, ["librosa=not extracted (stage skipped)"]
    if not art.get("measured"):
        out["librosa_status"] = art["status"]
        return out, []
    status = seg.get("audio_status") or UNCERTAIN
    out["librosa_status"] = status
    if status not in MEASURED_STATUSES:
        return out, []
    row = next((r for r in art["segments"] if r["segment_index"] == seg["segment_index"]), None)
    if row is None:
        out["librosa_status"] = FAILED
        return out, ["librosa=segment missing from librosa.json"]
    out.update({k: row.get(k) for k in LIBROSA_COLUMNS if k != "librosa_status"})
    return out, list(row.get("notes") or [])


def opening_librosa(art: dict | None, end: float) -> dict:
    if not art or not art.get("measured"):
        return {"status": None if art is None else art["status"], "low_energy_intervals": None, "tempo_bpm": None,
                "tempo_status": None}
    g = art["global"] or {}
    return {
        "status": art["status"],
        "low_energy_intervals": [iv for iv in art["low_energy_intervals"] if iv["start_s"] < end],
        "tempo_bpm": g.get("tempo_bpm"),
        "tempo_status": g.get("tempo_status"),
    }
