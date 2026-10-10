"""Speech stage with mocked transcribers (no model downloads, no FFmpeg needed)."""

from __future__ import annotations

import sys
import types
import wave
from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hookline.config import SpeechConfig
from hookline.context import BackendUnavailable, LazyBackend, MediaContext
from hookline.speech import (
    SPEECH_COLUMNS,
    WORD_COLUMNS,
    FasterWhisperTranscriber,
    extract_speech,
    opening_speech,
    speech_key,
    speech_window,
    word_rows,
)

CFG = SpeechConfig()


def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * 220 * t)


def ctx_for(tmp_path: Path, samples: np.ndarray | None = None, **kw) -> MediaContext:
    audio_path = str(write_wav(tmp_path / "a.wav", samples)) if samples is not None else None
    base = dict(video_id="v", source_path="/x/v.mp4", source_sha256="abc", prep_key="k1", duration=4.0,
                visual_stream="available", audio_stream="available", audio_path=audio_path,
                audio_span=[0.0, 4.0], video_span=[0.0, 4.0])
    return MediaContext(**(base | kw))


class FakeTranscriber:
    def __init__(self, result: dict | None = None, exc: Exception | None = None):
        self.result, self.exc, self.calls = result, exc, 0

    def info(self) -> dict:
        return {"backend": "fake", "model": "fake-1"}

    def transcribe(self, samples, sr) -> dict:
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result


def must_not_load() -> LazyBackend:
    def factory():
        raise AssertionError("model must not be loaded")
    return LazyBackend(factory)


def seg(start, end, words, text=None, lp=-0.2, nsp=0.05, cr=1.3):
    return {"start": start, "end": end, "text": text or " ".join(w[2] for w in words), "avg_logprob": lp,
            "no_speech_prob": nsp, "compression_ratio": cr,
            "words": [{"start": a, "end": b, "word": f" {w}", "probability": 0.9} for a, b, w in words]}


HINDI = {"language": "hi", "language_probability": 0.97, "segments": [
    seg(0.2, 1.4, [(0.2, 0.6, "नमस्ते"), (0.7, 1.4, "दोस्तों")]),
    seg(2.0, 2.8, [(2.0, 2.3, "आज"), (2.4, 2.8, "देखिए")]),
]}


@pytest.mark.parametrize("stream,reason", [
    ("absent", "no audio stream"),
    ("failed", "could not be decoded"),
    ("uncertain", "presence unknown"),
])
def test_non_available_audio_keeps_its_status_and_never_loads_model(tmp_path, stream, reason) -> None:
    backend = must_not_load()
    art = extract_speech(ctx_for(tmp_path, audio_stream=stream), CFG, backend)
    assert art["status"] == stream and reason in art["reason"]
    assert art["measured"] is False and art["speech_detected"] is None  # not evidence of no speech
    assert art["segments"] == [] and not backend.loaded and backend.error is None


def test_silent_audio_is_measured_no_speech_without_model(tmp_path) -> None:
    backend = must_not_load()
    art = extract_speech(ctx_for(tmp_path, np.zeros(16000 * 2)), CFG, backend)
    assert art["status"] == "available" and art["measured"] is True
    assert art["speech_detected"] is False and art["method"] == "skipped_silent_audio"
    assert backend.error is None


def test_model_unavailable_is_failure_not_absence(tmp_path) -> None:
    def factory():
        raise BackendUnavailable("faster-whisper is not installed")
    backend = LazyBackend(factory)
    ctx = ctx_for(tmp_path, tone(2))
    art = extract_speech(ctx, CFG, backend)
    assert art["status"] == "failed" and art["speech_detected"] is None and art["measured"] is False
    assert "model unavailable" in art["reason"] and "not installed" in art["reason"]
    again = extract_speech(ctx, CFG, backend)  # load attempted once, failure remembered
    assert again["status"] == "failed" and backend.error == "faster-whisper is not installed"


def test_transcription_error_is_failure(tmp_path) -> None:
    fake = FakeTranscriber(exc=RuntimeError("CUDA out of memory"))
    art = extract_speech(ctx_for(tmp_path, tone(2)), CFG, LazyBackend(lambda: fake))
    assert art["status"] == "failed" and art["speech_detected"] is None
    assert "transcription failed" in art["reason"] and art["errors"]
    assert art["model"] == {"backend": "fake", "model": "fake-1"}


def test_no_segments_is_measured_no_speech(tmp_path) -> None:
    fake = FakeTranscriber({"language": "en", "language_probability": 0.4, "segments": []})
    art = extract_speech(ctx_for(tmp_path, tone(2)), CFG, LazyBackend(lambda: fake))
    assert art["status"] == "available" and art["speech_detected"] is False and art["method"] == "asr"
    assert fake.calls == 1 and art["stats"]["words"] == 0


def test_timestamps_shift_by_audio_offset_and_language_is_detected(tmp_path) -> None:
    fake = FakeTranscriber(HINDI)
    ctx = ctx_for(tmp_path, tone(3), audio_offset=1.0, duration=3.5)
    art = extract_speech(ctx, CFG, LazyBackend(lambda: fake))
    assert CFG.language is None  # not forced to English
    assert art["status"] == "available" and art["speech_detected"] is True
    assert art["language"] == "hi" and art["language_probability"] == 0.97
    s0, s1 = art["segments"]
    assert (s0["start_s"], s0["end_s"]) == (1.2, 2.4)
    assert [(w["start_s"], w["end_s"], w["word"]) for w in s0["words"]] == [(1.2, 1.6, "नमस्ते"), (1.7, 2.4, "दोस्तों")]
    assert s1["end_s"] == 3.5  # clamped to the timeline duration (3.8 > 3.5)
    assert art["word_timestamps"] is True and art["timeline"]["offset_seconds"] == 1.0
    assert art["stats"]["words"] == 4 and art["timing"]["audio_seconds"] == 3.0


def test_all_low_confidence_segments_are_uncertain(tmp_path) -> None:
    result = {"language": "en", "language_probability": 0.3, "segments": [
        seg(0.0, 2.0, [(0.0, 2.0, "Thank you.")], lp=-1.4, nsp=0.8),
        seg(2.0, 3.0, [(2.0, 3.0, "you you you")], lp=-0.5, cr=3.1),
    ]}
    art = extract_speech(ctx_for(tmp_path, tone(3)), CFG, LazyBackend(lambda: FakeTranscriber(result)))
    assert art["status"] == "uncertain" and art["speech_detected"] is None
    assert art["segments"][0]["flags"] == ["low_confidence", "possible_non_speech"]
    assert art["segments"][1]["flags"] == ["repetitive"]


def test_empty_segments_are_dropped(tmp_path) -> None:
    result = {"language": "en", "segments": [{"start": 0, "end": 1, "text": "  ", "words": []}]}
    art = extract_speech(ctx_for(tmp_path, tone(2)), CFG, LazyBackend(lambda: FakeTranscriber(result)))
    assert art["segments"] == [] and art["speech_detected"] is False


def test_missing_wav_is_failed(tmp_path) -> None:
    ctx = ctx_for(tmp_path, audio_path=str(tmp_path / "gone.wav"))
    art = extract_speech(ctx, CFG, must_not_load())
    assert art["status"] == "failed" and "WAV missing" in art["reason"]


def window(start, end, audio="available"):
    return {"start_s": start, "end_s": end, "audio_status": audio}


def test_speech_window_assigns_words_by_midpoint(tmp_path) -> None:
    art = extract_speech(ctx_for(tmp_path, tone(3)), CFG, LazyBackend(lambda: FakeTranscriber(HINDI)))
    w, notes = speech_window(art, window(1.0, 1.5))
    assert w["speech_status"] == "available" and notes == []
    assert w["speech_word_count"] == 1 and w["speech_text"] == "दोस्तों"  # 0.7-1.4, midpoint 1.05
    assert w["speech_coverage"] == pytest.approx(0.4 / 0.5)
    assert w["speech_word_prob_mean"] == 0.9
    between, _ = speech_window(art, window(0.5, 1.0))  # words overlap but no midpoint falls here
    assert between["speech_word_count"] == 0 and between["speech_present"] is True
    assert between["speech_coverage"] == pytest.approx(0.4 / 0.5)  # 0.5-0.6 and 0.7-1.0
    gap, _ = speech_window(art, window(1.5, 2.0))
    assert gap["speech_present"] is False and gap["speech_word_count"] == 0 and gap["speech_text"] is None
    assert gap["speech_coverage"] == 0.0  # measured zero, not missing


def test_words_map_onto_scene_cut_segments_once_with_exact_times(tmp_path) -> None:
    art = extract_speech(ctx_for(tmp_path, tone(3)), CFG, LazyBackend(lambda: FakeTranscriber(HINDI)))
    bounds = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.2), (1.2, 1.7), (1.7, 2.2), (2.2, 2.5)]  # cut at 1.2 s; timeline ends early
    segments = [{"segment_index": i, "start_s": a, "end_s": b, "audio_status": "available"} for i, (a, b) in enumerate(bounds)]
    rows = word_rows(art | {"video_id": "v"}, segments)
    assert list(rows[0]) == WORD_COLUMNS
    assert [(r["start_s"], r["end_s"], r["word"], r["segment_index"]) for r in rows] == [
        (0.2, 0.6, "नमस्ते", 0), (0.7, 1.4, "दोस्तों", 2), (2.0, 2.3, "आज", 4), (2.4, 2.8, "देखिए", 5)]
    assert [r["asr_segment_id"] for r in rows] == [0, 0, 1, 1]

    last = len(segments) - 1
    counts = [speech_window(art, s, last=i == last)[0]["speech_word_count"] for i, s in enumerate(segments)]
    assert counts == [1, 0, 1, 0, 1, 1]  # every word counted exactly once, matching words.csv
    assert speech_window(art, segments[-1])[0]["speech_word_count"] == 0  # midpoint 2.6 is past a non-final end
    assert word_rows({"status": "failed", "measured": False}, segments) == [] and word_rows(None, segments) == []


def test_speech_window_statuses(tmp_path) -> None:
    art = extract_speech(ctx_for(tmp_path, tone(3)), CFG, LazyBackend(lambda: FakeTranscriber(HINDI)))
    absent, _ = speech_window(art, window(3.0, 3.5, audio="absent"))
    assert absent["speech_status"] == "absent" and all(absent[c] is None for c in SPEECH_COLUMNS[1:])
    partial, _ = speech_window(art, window(0.0, 0.5, audio="uncertain"))
    assert partial["speech_status"] == "uncertain" and partial["speech_word_count"] == 1
    skipped, notes = speech_window(None, window(0.0, 0.5))
    assert skipped["speech_status"] == "uncertain" and "not extracted" in notes[0]
    failed_art = extract_speech(ctx_for(tmp_path, audio_stream="failed"), CFG, None)
    failed, _ = speech_window(failed_art, window(0.0, 0.5))
    assert failed["speech_status"] == "failed" and failed["speech_present"] is None


def test_speech_window_without_word_timestamps(tmp_path) -> None:
    result = {"language": "en", "segments": [{**seg(0.0, 1.0, []), "text": "hello there", "words": None}]}
    art = extract_speech(ctx_for(tmp_path, tone(2)), CFG, LazyBackend(lambda: FakeTranscriber(result)))
    assert art["word_timestamps"] is False and any("no word timestamps" in w for w in art["warnings"])
    w, notes = speech_window(art, window(0.0, 0.5))
    assert w["speech_coverage"] == 1.0 and w["speech_word_count"] is None
    assert any("not_applicable" in n for n in notes)


def test_opening_only_claims_no_speech_when_fully_measured(tmp_path) -> None:
    empty = {"language": "en", "segments": []}
    art = extract_speech(ctx_for(tmp_path, tone(2)), CFG, LazyBackend(lambda: FakeTranscriber(empty)))
    rows = [{"speech_status": "available"}, {"speech_status": "available"}]
    assert opening_speech(art, rows, 1.0)["speech_detected"] is False
    rows[1]["speech_status"] = "uncertain"
    assert opening_speech(art, rows, 1.0)["speech_detected"] is None
    assert opening_speech(None, rows, 1.0)["speech_detected"] is None

    art = extract_speech(ctx_for(tmp_path, tone(3)), CFG, LazyBackend(lambda: FakeTranscriber(HINDI)))
    o = opening_speech(art, [{"speech_status": "available"}] * 6, 3.0)
    assert o["speech_detected"] is True and o["first_word_s"] == 0.2 and o["word_count"] == 4
    assert o["words_per_s"] == pytest.approx(4 / 3, abs=1e-3) and o["language"] == "hi"
    assert o["transcript"].startswith("नमस्ते")


def test_speech_cache_key_tracks_model_and_config_not_device() -> None:
    ctx = MediaContext(video_id="v", source_path="/x", source_sha256="s", prep_key="k1", duration=1.0)
    base = speech_key(ctx, CFG)
    assert speech_key(ctx, replace(CFG, device="cuda")) == base
    assert speech_key(ctx, replace(CFG, model="base")) != base
    assert speech_key(ctx, replace(CFG, model_revision="abc123")) != base
    assert speech_key(ctx, replace(CFG, vad_filter=False)) != base
    assert speech_key(replace(ctx, prep_key="k2"), CFG) != base


Word = namedtuple("Word", "start end word probability")
Segment = namedtuple("Segment", "start end text avg_logprob no_speech_prob compression_ratio words")
Info = namedtuple("Info", "language language_probability duration_after_vad")


def fake_faster_whisper(monkeypatch, fail_load: bool = False) -> dict:
    calls: dict = {}

    class WhisperModel:
        def __init__(self, model, **kw):
            calls["init"] = {"model": model, **kw}
            if fail_load:
                raise RuntimeError("model not found in local cache")

        def transcribe(self, audio, **kw):
            calls["transcribe"] = kw
            gen = (Segment(0.0, 1.0, " Bonjour", -0.2, 0.01, 1.1, [Word(0.0, 0.5, " Bonjour", 0.95)]) for _ in range(1))
            return gen, Info("fr", 0.9, 1.0)

    mod = types.ModuleType("faster_whisper")
    mod.__version__ = "9.9.9"
    mod.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return calls


def test_faster_whisper_adapter_offline_by_default(monkeypatch) -> None:
    calls = fake_faster_whisper(monkeypatch)
    t = FasterWhisperTranscriber(CFG)
    assert calls["init"]["local_files_only"] is True and calls["init"]["model"] == "small"
    out = t.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    kw = calls["transcribe"]
    assert kw["task"] == "transcribe" and kw["language"] is None and kw["word_timestamps"] is True
    assert out["language"] == "fr" and out["segments"][0]["words"][0]["word"] == " Bonjour"
    assert t.info()["backend_version"] == "9.9.9"
    with pytest.raises(ValueError):
        t.transcribe(np.zeros(100, dtype=np.float32), 44100)
    FasterWhisperTranscriber(CFG, allow_download=True)
    assert calls["init"]["local_files_only"] is False


def test_faster_whisper_adapter_load_failure_hints_download(monkeypatch) -> None:
    fake_faster_whisper(monkeypatch, fail_load=True)
    with pytest.raises(BackendUnavailable, match="--download-models"):
        FasterWhisperTranscriber(CFG)


def test_faster_whisper_missing_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(BackendUnavailable, match="not installed"):
        FasterWhisperTranscriber(CFG)


# ------------------------------------------------------------- confidence flags and speech_result

def flagged(lp=-0.87, nsp=0.9, probs=(0.31, 0.54, 0.56)):
    s = seg(6.5, 8.1, [(6.5, 6.7, "अपने"), (6.7, 7.4, "काई"), (7.4, 8.1, "गया")], lp=lp, nsp=nsp)
    for w, p in zip(s["words"], probs):
        w["probability"] = p
    return s


def test_music_hallucination_pattern_is_uncertain_but_kept(tmp_path) -> None:
    # The FL004 pattern: decoder confident enough to keep text, but Whisper's own no-speech
    # probability is high, the words are unsure and the language guess is weak.
    result = {"language": "hi", "language_probability": 0.23, "segments": [flagged()]}
    art = extract_speech(ctx_for(tmp_path, tone(8.5), duration=8.5), CFG, LazyBackend(lambda: FakeTranscriber(result)))
    (s,) = art["segments"]
    assert s["flags"] == ["high_no_speech_prob", "low_word_probability"]
    assert art["status"] == "uncertain" and art["measured"] and art["speech_detected"] is None
    assert art["speech_result"] == "uncertain_speech" and art["flags"] == ["low_language_confidence"]
    assert [w["word"] for w in s["words"]] == ["अपने", "काई", "गया"] and s["words"][0]["start_s"] == 6.5  # nothing deleted
    assert any("wrong script or hallucinated" in w for w in art["warnings"])


def test_each_flag_alone() -> None:
    from hookline.speech import _flags
    assert _flags(flagged(lp=-0.2, nsp=0.05, probs=(0.9, 0.9, 0.9)), CFG) == []
    assert _flags(flagged(lp=-0.2, nsp=0.9, probs=(0.9, 0.9, 0.9)), CFG) == ["high_no_speech_prob"]
    assert _flags(flagged(lp=-1.5, nsp=0.9, probs=(0.9, 0.9, 0.9)), CFG) == ["low_confidence", "possible_non_speech"]
    assert _flags(flagged(lp=-0.2, nsp=0.05, probs=(0.3, 0.4, 0.5)), CFG) == ["low_word_probability"]
    assert _flags(flagged(lp=-0.2, nsp=0.05, probs=(None, None, None)), CFG) == []  # no probabilities: no claim


def test_speech_result_states(tmp_path) -> None:
    def result_for(**kw):
        c = ctx_for(tmp_path, kw.pop("samples", tone(4.0)), **kw.pop("ctx", {}))
        return extract_speech(c, CFG, kw.pop("backend"))

    good = {"language": "hi", "language_probability": 0.97, "segments": [flagged(lp=-0.2, nsp=0.05, probs=(0.9, 0.9, 0.9))]}
    assert result_for(backend=LazyBackend(lambda: FakeTranscriber(good)))["speech_result"] == "speech"
    empty = {"language": "en", "language_probability": 0.4, "segments": []}
    none = result_for(backend=LazyBackend(lambda: FakeTranscriber(empty)))
    assert (none["speech_result"], none["speech_detected"], none["flags"]) == ("no_speech", False, [])  # no words: no language flag
    silent = result_for(samples=np.zeros(16000 * 4), backend=must_not_load())
    assert silent["speech_result"] == "no_speech" and silent["method"] == "skipped_silent_audio"
    crash = result_for(backend=LazyBackend(lambda: FakeTranscriber(exc=RuntimeError("boom"))))
    assert (crash["speech_result"], crash["speech_detected"], crash["status"]) == ("failed", None, "failed")
    missing_model = result_for(backend=LazyBackend(lambda: (_ for _ in ()).throw(BackendUnavailable("not cached"))))
    assert missing_model["speech_result"] == "failed"
    absent = result_for(samples=None, ctx={"audio_stream": "absent"}, backend=must_not_load())
    assert absent["speech_result"] == "absent"
    unknown = result_for(samples=None, ctx={"audio_stream": "uncertain"}, backend=must_not_load())
    assert unknown["speech_result"] == "unknown" and unknown["speech_detected"] is None


def test_language_flag_only_for_detected_language_and_does_not_hide_speech(tmp_path) -> None:
    confident_words = {"language": "hi", "language_probability": 0.3,
                       "segments": [flagged(lp=-0.2, nsp=0.05, probs=(0.95, 0.9, 0.92))]}
    art = extract_speech(ctx_for(tmp_path, tone(8.5), duration=8.5), CFG, LazyBackend(lambda: FakeTranscriber(confident_words)))
    assert art["speech_result"] == "speech" and art["flags"] == ["low_language_confidence"] and art["status"] == "available"
    forced = extract_speech(ctx_for(tmp_path, tone(8.5), duration=8.5), replace(CFG, language="hi"),
                            LazyBackend(lambda: FakeTranscriber(confident_words)))
    assert forced["flags"] == []


def test_flag_thresholds_are_part_of_the_cache_key(tmp_path) -> None:
    c = ctx_for(tmp_path, tone(1.0))
    assert speech_key(c, replace(CFG, low_word_probability=0.4)) != speech_key(c, CFG)
