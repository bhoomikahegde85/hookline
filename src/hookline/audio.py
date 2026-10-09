"""Audio measurements on the mono 16 kHz WAV (NumPy only).

Descriptive only: louder or busier audio is not treated as better content.

- rms / rms_dbfs / peak_dbfs: level of the window (dBFS, 0 = full scale)
- near_silence_ratio: share of 20 ms frames below ``silence_dbfs``
- energy_variation_db: std of 20 ms frame levels (dB, floored at ``db_floor``)
- onset_strength_mean: mean spectral flux, i.e. the mean over frequency bins of
  positive dBFS increases per 10 ms hop (bins floored at ``spectrum_floor_db``);
  an activity proxy, not a beat tracker
- onset_count: local peaks of the spectral flux (syllable/note/beat-onset proxy)
- clipping_ratio: share of samples at or above ``clip_level``. Measured after
  down-mix and resampling, so it is approximate.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .status import AVAILABLE, UNCERTAIN, expects_data, window_status


@dataclass
class AudioSeries:
    samples: np.ndarray  # float32 in [-1, 1]
    sr: int
    offset: float  # timeline time of sample 0
    frame_starts: np.ndarray  # timeline times of 20 ms frames
    frame_db: np.ndarray
    flux_times: np.ndarray  # timeline times (frame centres) of spectral-flux values
    flux: np.ndarray  # NaN for the first STFT frame
    onset_times: np.ndarray

    @property
    def end(self) -> float:
        return self.offset + len(self.samples) / self.sr


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        sr, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return (data.astype(np.float32) / 32768.0), sr


def to_db(x: np.ndarray | float, floor: float) -> np.ndarray:
    return np.maximum(20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12)), floor)


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        return np.empty((0, size), dtype=x.dtype)
    n = 1 + (len(x) - size) // hop
    return np.lib.stride_tricks.as_strided(x, shape=(n, size), strides=(x.strides[0] * hop, x.strides[0]), writeable=False)


def spectral_flux(x: np.ndarray, n_fft: int, hop: int, floor_db: float) -> np.ndarray:
    frames = _frames(x, n_fft, hop)
    if len(frames) == 0:
        return np.empty(0)
    win = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * win, axis=1)) / (win.sum() / 2)  # full-scale sine peak ~ 0 dBFS
    db = np.maximum(20.0 * np.log10(np.maximum(mag, 1e-10)), floor_db)
    flux = np.full(len(frames), np.nan)
    flux[1:] = np.maximum(db[1:] - db[:-1], 0.0).mean(axis=1)
    return flux


def frame_energy_db(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    frames = _frames(x, n_fft, hop)
    return 10.0 * np.log10(np.maximum((frames.astype(np.float64) ** 2).mean(axis=1), 1e-12)) if len(frames) else np.empty(0)


def pick_onsets(
    times: np.ndarray, flux: np.ndarray, hop_seconds: float, cfg: PipelineConfig, energy_db: np.ndarray | None = None
) -> np.ndarray:
    """Local maxima that exceed the local mean by ``onset_delta_db``, at least ``onset_wait_seconds`` apart.

    With ``energy_db`` (aligned with ``flux``), peaks where the energy is falling are dropped afterwards:
    a sound stopping abruptly creates broadband spectral increase but is not an onset.
    """
    f = np.nan_to_num(flux, nan=0.0)
    m = max(1, int(round(cfg.onset_max_window / hop_seconds)))
    a = max(1, int(round(cfg.onset_avg_window / hop_seconds)))
    lag = cfg.onset_energy_lag_frames
    kept: list[int] = []
    for i in range(1, len(f) - 1):
        if f[i] <= 0 or f[i] < f[max(0, i - m) : i + m + 1].max():
            continue
        if f[i] < f[max(0, i - a) : i + a + 1].mean() + cfg.onset_delta_db:
            continue
        if kept and times[i] - times[kept[-1]] < cfg.onset_wait_seconds:
            continue
        kept.append(i)
    if energy_db is not None:
        kept = [i for i in kept if i < lag or energy_db[i] >= energy_db[i - lag] - cfg.onset_energy_drop_db]
    return times[kept] if kept else np.empty(0)


def analyze_audio(samples: np.ndarray, sr: int, offset: float, cfg: PipelineConfig) -> AudioSeries:
    size = max(1, int(round(cfg.audio_frame_seconds * sr)))
    frames = _frames(samples, size, size)
    frame_rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) if len(frames) else np.empty(0)
    frame_starts = offset + np.arange(len(frame_rms)) * size / sr
    flux = spectral_flux(samples, cfg.stft_n_fft, cfg.stft_hop, cfg.spectrum_floor_db)
    flux_times = offset + (np.arange(len(flux)) * cfg.stft_hop + cfg.stft_n_fft / 2) / sr
    return AudioSeries(
        samples=samples,
        sr=sr,
        offset=offset,
        frame_starts=frame_starts,
        frame_db=to_db(frame_rms, cfg.db_floor),
        flux_times=flux_times,
        flux=flux,
        onset_times=pick_onsets(flux_times, flux, cfg.stft_hop / sr, cfg, frame_energy_db(samples, cfg.stft_n_fft, cfg.stft_hop)),
    )


def _window_values(series: AudioSeries, start: float, end: float, cfg: PipelineConfig) -> dict | None:
    i0 = int(round((start - series.offset) * series.sr))
    i1 = int(round((end - series.offset) * series.sr))
    i0, i1 = max(0, i0), min(len(series.samples), i1)
    if i1 <= i0:
        return None
    x = series.samples[i0:i1].astype(np.float64)
    rms = float(np.sqrt((x**2).mean()))
    peak = float(np.abs(x).max())
    size = cfg.audio_frame_seconds
    fmask = (series.frame_starts + size / 2 >= start) & (series.frame_starts + size / 2 < end)
    fdb = series.frame_db[fmask]
    flux_mask = (series.flux_times >= start) & (series.flux_times < end)
    flux = series.flux[flux_mask]
    flux = flux[~np.isnan(flux)]
    return {
        "n": i1 - i0,
        "rms": rms,
        "rms_dbfs": float(to_db(rms, cfg.db_floor)),
        "peak_dbfs": float(to_db(peak, cfg.db_floor)),
        "near_silence_ratio": float((fdb < cfg.silence_dbfs).mean()) if fdb.size else None,
        "energy_variation_db": float(fdb.std()) if fdb.size >= 2 else None,
        "onset_strength_mean": float(flux.mean()) if flux.size else None,
        "onset_count": int(((series.onset_times >= start) & (series.onset_times < end)).sum()),
        "clipping_ratio": float((np.abs(x) >= cfg.clip_level).mean()),
        "digital_silence": peak == 0.0,
    }


def aggregate_audio(
    series: AudioSeries | None,
    start: float,
    end: float,
    stream: str,
    expected_span: list[float] | None,
    cfg: PipelineConfig,
) -> dict:
    """Audio values for one segment. Values stay None unless measured."""
    out: dict = {
        "audio_status": stream,
        "audio_coverage": None,
        "rms": None,
        "rms_dbfs": None,
        "peak_dbfs": None,
        "near_silence_ratio": None,
        "energy_variation_db": None,
        "onset_strength_mean": None,
        "onset_count": None,
        "clipping_ratio": None,
        "audio_warnings": [],
        "feature_status": {},
    }
    if series is None or stream != AVAILABLE:
        return out
    v = _window_values(series, start, end, cfg)
    n = v["n"] if v else 0
    coverage = min(1.0, n / ((end - start) * series.sr)) if end > start else 0.0
    status = window_status(stream, coverage, expects_data(start, end, expected_span, cfg.min_coverage), cfg.min_coverage)
    out.update(audio_status=status, audio_coverage=round(coverage, 4))
    if not v:
        return out
    for k in ("rms", "rms_dbfs", "peak_dbfs", "near_silence_ratio", "energy_variation_db", "onset_strength_mean", "onset_count", "clipping_ratio"):
        out[k] = v[k]
    if v["energy_variation_db"] is None:
        out["feature_status"]["energy_variation_db"] = "not_applicable: fewer than 2 audio frames"
    if v["onset_strength_mean"] is None:
        out["feature_status"]["onset_strength_mean"] = "not_applicable: no spectral frame centred in window"
    flags = out["audio_warnings"]
    if v["digital_silence"]:
        flags.append("digital_silence")
    if v["clipping_ratio"] > cfg.clipping_warn_ratio:
        flags.append("possible_clipping")
    if status == UNCERTAIN:
        flags.append("partial_audio_coverage")
    return out


def opening_audio(series: AudioSeries | None, end: float, cfg: PipelineConfig) -> dict | None:
    if series is None:
        return None
    v = _window_values(series, 0.0, end, cfg)
    if not v:
        return None
    loud = series.frame_starts[(series.frame_db >= cfg.silence_dbfs) & (series.frame_starts < end)]
    covered = min(end, series.end) - max(0.0, series.offset)
    return {
        "rms_dbfs": v["rms_dbfs"],
        "peak_dbfs": v["peak_dbfs"],
        "near_silence_ratio": v["near_silence_ratio"],
        "onset_count": v["onset_count"],
        "onset_rate_per_s": v["onset_count"] / covered if covered > 0 else None,
        "first_non_silent_s": round(float(loud[0]), 3) if loud.size else None,
        "clipping_ratio": v["clipping_ratio"],
        "digital_silence": v["digital_silence"],
    }
