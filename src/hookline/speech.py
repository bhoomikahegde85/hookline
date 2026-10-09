"""Timestamped speech transcription with Whisper (faster-whisper / CTranslate2).

Runs on the cached mono 16 kHz WAV from preprocessing (Whisper's native input).
Whisper times are relative to the WAV start and are shifted by the audio offset
from the timestamp map, so every time here is on the project timeline
(seconds from the start of the original's playback).

File-level status (``speech.json -> status``):
- ``absent``: the file has no audio stream, so there is nothing to hear.
- ``uncertain``: audio presence unknown (inspection failed), or every
  transcribed segment is flagged as low confidence.
- ``failed``: audio exists but could not be decoded, the model was unavailable,
  or transcription raised. ``speech_detected`` stays null: a failure is never
  evidence that the video has no speech.
- ``available``: transcription ran, or the audio is below ``silence_dbfs``
  throughout (model skipped). ``speech_detected`` is then true or false.

The language is detected per video unless configured; the task is always
"transcribe" (never translate). Whisper detects one language per file, so
code-switched speech may be transcribed in a single script.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .audio import load_wav, to_db
from .config import SCHEMA_VERSION, SPEECH_VERSION, SpeechConfig, stable_hash
from .context import BackendUnavailable, LazyBackend, MediaContext, clean, covered_length, peak_rss_mb, segment_locator
from .status import ABSENT, AVAILABLE, FAILED, UNCERTAIN

SPEECH_COLUMNS = [
    "speech_status", "speech_present", "speech_coverage", "speech_word_count", "speech_word_prob_mean", "speech_text",
]
WORD_COLUMNS = ["video_id", "word_index", "segment_index", "start_s", "end_s", "word", "probability", "asr_segment_id", "asr_flags"]
TIMELINE_NOTE = "seconds from start of original playback; Whisper time + audio offset_seconds"


class Transcriber(Protocol):
    def info(self) -> dict: ...

    def transcribe(self, samples: np.ndarray, sr: int) -> dict:
        """Return {language, language_probability, segments: [{start, end, text, avg_logprob,
        no_speech_prob, compression_ratio, words: [{start, end, word, probability}] | None}]}
        with times in seconds from the first sample."""
        ...


class FasterWhisperTranscriber:
    backend = "faster-whisper"

    def __init__(self, cfg: SpeechConfig, allow_download: bool = False):
        try:
            import faster_whisper
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise BackendUnavailable("faster-whisper is not installed (use .venv-models, see requirements-models.txt)") from exc
        self.cfg = cfg
        self.version = getattr(faster_whisper, "__version__", None)
        try:
            self.model = WhisperModel(
                cfg.model, device=cfg.device, compute_type=cfg.compute_type,
                revision=cfg.model_revision, local_files_only=not allow_download,
            )
        except Exception as exc:
            hint = "" if allow_download else "; it is not in the local cache, re-run with --download-models to fetch it"
            raise BackendUnavailable(f"could not load Whisper model {cfg.model!r}: {type(exc).__name__}: {exc}{hint}") from exc

    def info(self) -> dict:
        c = self.cfg
        return {
            "backend": self.backend, "backend_version": self.version, "model": c.model,
            "model_revision": c.model_revision, "compute_type": c.compute_type, "device": c.device,
        }

    def transcribe(self, samples: np.ndarray, sr: int) -> dict:
        if sr != 16000:
            raise ValueError(f"Whisper expects 16 kHz audio, got {sr} Hz")
        c = self.cfg
        segments, info = self.model.transcribe(
            samples, language=c.language, task="transcribe", beam_size=c.beam_size,
            word_timestamps=c.word_timestamps, vad_filter=c.vad_filter,
            condition_on_previous_text=c.condition_on_previous_text,
        )
        out = []
        for s in segments:  # generator: decoding happens here
            words = None
            if s.words is not None:
                words = [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability} for w in s.words]
            out.append({
                "start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio, "words": words,
            })
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_after_vad": getattr(info, "duration_after_vad", None),
            "segments": out,
        }


def speech_key(ctx: MediaContext, cfg: SpeechConfig) -> str:
    return stable_hash("speech", ctx.prep_key or ctx.source_sha256 or ctx.source_path, cfg.cache_dict())


def _silent_throughout(samples: np.ndarray, sr: int, silence_dbfs: float) -> bool:
    n = max(1, int(0.02 * sr))
    frames = samples[: len(samples) // n * n].reshape(-1, n) if len(samples) >= n else samples.reshape(1, -1)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return bool(frames.size) and float(to_db(rms.max(), -120.0)) < silence_dbfs


def _flags(seg: dict, cfg: SpeechConfig) -> list[str]:
    flags = []
    lp, nsp, cr = seg.get("avg_logprob"), seg.get("no_speech_prob"), seg.get("compression_ratio")
    if lp is not None and lp < cfg.low_logprob:
        flags.append("low_confidence")
        if nsp is not None and nsp > cfg.high_no_speech_prob:
            flags.append("possible_non_speech")
    if cr is not None and cr > cfg.high_compression_ratio:
        flags.append("repetitive")
    return flags


def _shift(t: float | None, offset: float, duration: float | None) -> float | None:
    if t is None:
        return None
    t = max(0.0, t + offset)
    return round(min(t, duration) if duration else t, 3)


def _normalise(result: dict, offset: float, duration: float | None, cfg: SpeechConfig) -> list[dict]:
    segs = []
    for i, s in enumerate(result.get("segments") or []):
        text = (s.get("text") or "").strip()
        words = None
        if s.get("words") is not None:
            words = [
                {
                    "start_s": _shift(w.get("start"), offset, duration),
                    "end_s": _shift(w.get("end"), offset, duration),
                    "word": (w.get("word") or "").strip(),
                    "probability": clean(w.get("probability")),
                }
                for w in s["words"]
                if (w.get("word") or "").strip()
            ]
        if not text and not words:
            continue
        segs.append({
            "id": i,
            "start_s": _shift(s.get("start"), offset, duration),
            "end_s": _shift(s.get("end"), offset, duration),
            "text": text,
            "avg_logprob": clean(s.get("avg_logprob")),
            "no_speech_prob": clean(s.get("no_speech_prob")),
            "compression_ratio": clean(s.get("compression_ratio")),
            "flags": _flags(s, cfg),
            "words": words,
        })
    return segs


def _base(ctx: MediaContext, cfg: SpeechConfig, key: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "speech_version": SPEECH_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": ctx.video_id,
        "source_path": ctx.source_path,
        "source_sha256": ctx.source_sha256,
        "cache_key": key,
        "status": FAILED,
        "measured": False,
        "speech_detected": None,
        "method": None,
        "reason": None,
        "language": None,
        "language_probability": None,
        "word_timestamps": None,
        "model": None,
        "config": cfg.to_dict(),
        "timeline": {"convention": TIMELINE_NOTE, "offset_seconds": ctx.audio_offset, "duration_s": ctx.duration},
        "timing": None,
        "stats": None,
        "warnings": [],
        "errors": [],
        "segments": [],
    }


def extract_speech(ctx: MediaContext, cfg: SpeechConfig, backend: LazyBackend | None) -> dict:
    """Transcribe one video's cached WAV. The model is only loaded if transcription is needed."""
    art = _base(ctx, cfg, speech_key(ctx, cfg))
    if ctx.audio_stream != AVAILABLE:
        art["status"] = ctx.audio_stream
        art["reason"] = {
            ABSENT: "no audio stream",
            UNCERTAIN: "audio presence unknown (inspection failed)",
            FAILED: "audio stream could not be decoded; speech not measured",
        }.get(ctx.audio_stream, f"audio stream status {ctx.audio_stream}")
        return art
    if not ctx.audio_path or not Path(ctx.audio_path).is_file():
        art["reason"] = "cached WAV missing; re-run scripts/run_pipeline.py"
        return art
    try:
        samples, sr = load_wav(ctx.audio_path)
    except Exception as exc:
        art["reason"] = f"could not read cached WAV: {type(exc).__name__}: {exc}"
        return art

    audio_seconds = len(samples) / sr if sr else 0.0
    if cfg.skip_silent_audio and _silent_throughout(samples, sr, cfg.silence_dbfs):
        art.update(status=AVAILABLE, measured=True, speech_detected=False, method="skipped_silent_audio",
                   reason=f"every 20 ms frame is below {cfg.silence_dbfs} dBFS; model not run")
        art["stats"] = {"segments": 0, "words": 0, "speech_seconds": 0.0, "audio_seconds": round(audio_seconds, 3)}
        return art
    if backend is None:
        art["reason"] = "no transcription backend configured"
        return art

    try:
        transcriber = backend.get()
    except BackendUnavailable as exc:
        art["reason"] = f"model unavailable: {exc}"
        return art
    art["model"] = transcriber.info()
    started = time.perf_counter()
    try:
        result = transcriber.transcribe(samples, sr)
    except Exception as exc:
        art["reason"] = f"transcription failed: {type(exc).__name__}: {exc}"
        art["errors"].append(art["reason"])
        return art
    elapsed = time.perf_counter() - started

    segs = _normalise(result, ctx.audio_offset, ctx.duration, cfg)
    has_words = any(s["words"] is not None for s in segs) if segs else None
    confident = [s for s in segs if not s["flags"]]
    art.update(
        status=AVAILABLE if confident or not segs else UNCERTAIN,
        measured=True,
        method="asr",
        language=result.get("language"),
        language_probability=clean(result.get("language_probability")),
        word_timestamps=has_words,
        segments=segs,
    )
    art["speech_detected"] = True if confident else (False if not segs else None)
    if segs and not confident:
        art["reason"] = "every transcribed segment is flagged low confidence (possible hallucination on music/noise)"
        art["warnings"].append(art["reason"])
    elif any(s["flags"] for s in segs):
        art["warnings"].append(f"{sum(1 for s in segs if s['flags'])} of {len(segs)} segments flagged low confidence")
    if has_words is False:
        art["warnings"].append("backend returned no word timestamps; segment times used for coverage")
    art["timing"] = {
        "audio_seconds": round(audio_seconds, 3),
        "processing_seconds": round(elapsed, 3),
        "realtime_factor": round(elapsed / audio_seconds, 4) if audio_seconds else None,
        "process_peak_rss_mb": peak_rss_mb(),
    }
    art["stats"] = {
        "segments": len(segs),
        "words": sum(len(s["words"] or []) for s in segs),
        "speech_seconds": round(covered_length([(s["start_s"], s["end_s"]) for s in segs], 0.0, float("inf")), 3),
        "audio_seconds": round(audio_seconds, 3),
    }
    return art


def _words(art: dict) -> list[tuple[dict, dict]]:
    return [(w, s) for s in art.get("segments") or [] for w in (s.get("words") or [])]


def word_rows(art: dict | None, segments: list[dict]) -> list[dict]:
    """One row per word with its exact times and the segment containing its midpoint."""
    if not art or not art.get("measured"):
        return []
    locate = segment_locator(segments)
    rows = []
    for w, s in _words(art):
        if w["start_s"] is None or w["end_s"] is None:
            continue
        rows.append({
            "video_id": art["video_id"], "word_index": len(rows), "segment_index": locate((w["start_s"] + w["end_s"]) / 2),
            "start_s": w["start_s"], "end_s": w["end_s"], "word": w["word"], "probability": w["probability"],
            "asr_segment_id": s["id"], "asr_flags": s["flags"],
        })
    return rows


def speech_window(art: dict | None, seg: dict, last: bool = False) -> tuple[dict, list[str]]:
    """Speech columns for one segment row of the features report. Words belong to the segment
    containing their midpoint (``last``: the final segment also takes midpoints at its end)."""
    out: dict[str, Any] = {k: None for k in SPEECH_COLUMNS}
    if art is None:
        out["speech_status"] = UNCERTAIN
        return out, ["speech=not extracted (stage skipped)"]
    if not art.get("measured"):
        out["speech_status"] = art["status"]
        return out, []
    audio_status = seg.get("audio_status")
    if audio_status in (ABSENT, FAILED):
        out["speech_status"] = audio_status
        return out, []

    start, end = seg["start_s"], seg["end_s"]
    duration = end - start
    def inside(mid: float) -> bool:
        return start <= mid < end or (last and mid >= end)

    words = [(w, s) for w, s in _words(art) if w["start_s"] is not None and w["end_s"] is not None
             and inside((w["start_s"] + w["end_s"]) / 2)]
    if art.get("word_timestamps") is False:
        spans = [(s["start_s"], s["end_s"]) for s in art["segments"]]
    else:
        spans = [(w["start_s"], max(w["end_s"], w["start_s"] + 0.02)) for w, _ in _words(art) if w["start_s"] is not None]
    coverage = covered_length(spans, start, end) / duration if duration > 0 else 0.0
    notes: list[str] = []
    status = AVAILABLE if audio_status == AVAILABLE else audio_status
    if art["status"] == UNCERTAIN or (words and all(s["flags"] for _, s in words)):
        status = UNCERTAIN
        notes.append("speech=low-confidence transcription")
    probs = [w["probability"] for w, _ in words if w["probability"] is not None]
    out.update(
        speech_status=status,
        speech_present=bool(words) or coverage > 0,
        speech_coverage=round(coverage, 4),
        speech_word_count=len(words) if art.get("word_timestamps") is not False else None,
        speech_word_prob_mean=round(float(np.mean(probs)), 4) if probs else None,
        speech_text=" ".join(w["word"] for w, _ in words) or None,
    )
    if art.get("word_timestamps") is False:
        notes.append("speech_word_count=not_applicable: no word timestamps")
    return out, notes


def opening_speech(art: dict | None, rows: list[dict], end: float) -> dict:
    """Speech in [0, end): first word, words, transcript. Absence is only claimed if fully measured."""
    summary: dict[str, Any] = {"status": UNCERTAIN if art is None else art["status"], "speech_detected": None,
                               "language": None, "first_word_s": None, "word_count": None, "words_per_s": None,
                               "speech_coverage": None, "transcript": None}
    if art is None or not art.get("measured") or end <= 0:
        return summary
    words = [w for w, _ in _words(art) if w["start_s"] is not None and w["start_s"] < end]
    segs = [s for s in art["segments"] if s["start_s"] is not None and s["start_s"] < end]
    spans = ([(w["start_s"], max(w["end_s"], w["start_s"] + 0.02)) for w in words] if art.get("word_timestamps") is not False
             else [(s["start_s"], s["end_s"]) for s in segs])
    fully_measured = all(r.get("speech_status") == AVAILABLE for r in rows) and bool(rows)
    found = bool(words or segs)
    summary.update(
        language=art.get("language"),
        speech_detected=True if found else (False if fully_measured else None),
        first_word_s=(min(w["start_s"] for w in words) if words else (min(s["start_s"] for s in segs) if segs else None)),
        word_count=len(words) if art.get("word_timestamps") is not False else None,
        speech_coverage=round(covered_length(spans, 0.0, end) / end, 4),
        transcript=(" ".join(w["word"] for w in words) if words else " ".join(s["text"] for s in segs)) or None,
    )
    if summary["word_count"] is not None:
        summary["words_per_s"] = round(summary["word_count"] / end, 3)
    return summary
