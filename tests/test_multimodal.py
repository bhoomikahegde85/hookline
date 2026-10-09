"""Speech + on-screen-text stage end to end on FFmpeg clips, with mocked models."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hookline.config import GeminiConfig, OcrConfig, PipelineConfig, SiglipConfig, SpeechConfig
from hookline.context import BackendUnavailable, LazyBackend
from hookline.features import SEGMENT_COLUMNS
from hookline.multimodal import (
    MERGED_COLUMNS,
    OPENING_MM_COLUMNS,
    Backends,
    process_video_multimodal,
    run_multimodal,
)
from hookline.ocr import OBSERVATION_COLUMNS
from hookline.speech import WORD_COLUMNS
from hookline.pipeline import VideoInput

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg not installed")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG, SCFG, OCFG = PipelineConfig(), SpeechConfig(), OcrConfig()
BOX = "drawbox=x=20:y=40:w=100:h=30:color=white:t=fill"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("mm_clips")
    v = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    a = ["-c:a", "aac", "-b:a", "64k"]
    gray = "color=gray:s=180x320:r=30"
    c = {k: d / f"{k}.mp4" for k in ("talk", "silent_video", "audio_only", "silent_audio", "very_short")}
    ffmpeg("-f", "lavfi", "-i", f"{gray}:d=4", "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=4",
           "-vf", f"{BOX}:enable='between(t,1.0,2.4)'", *v, *a, "-shortest", str(c["talk"]))
    ffmpeg("-f", "lavfi", "-i", f"{gray}:d=3", "-vf", f"{BOX}:enable='gte(t,0.5)'", "-an", *v, str(c["silent_video"]))
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=330:duration=3", "-vn", *a, str(c["audio_only"]))
    ffmpeg("-f", "lavfi", "-i", f"{gray}:d=2", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "2", *v, *a,
           str(c["silent_audio"]))
    ffmpeg("-f", "lavfi", "-i", f"{gray}:d=0.3", "-f", "lavfi", "-i", "sine=duration=0.3", *v, *a, "-shortest",
           str(c["very_short"]))
    c["zero_bytes"] = d / "zero_bytes.mp4"
    c["zero_bytes"].write_bytes(b"")
    return c


WORDS = [(0.3, 0.7, "hey"), (0.8, 1.2, "look"), (2.0, 2.6, "this")]


class FakeTranscriber:
    def __init__(self, exc: Exception | None = None):
        self.calls, self.exc = 0, exc

    def info(self) -> dict:
        return {"backend": "fake-asr", "model": "fake"}

    def transcribe(self, samples, sr) -> dict:
        self.calls += 1
        if self.exc:
            raise self.exc
        end = len(samples) / sr
        words = [{"start": s, "end": e, "word": f" {w}", "probability": 0.9} for s, e, w in WORDS if e <= end]
        segs = [{"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(x["word"] for x in words),
                 "avg_logprob": -0.2, "no_speech_prob": 0.02, "compression_ratio": 1.2, "words": words}] if words else []
        return {"language": "en", "language_probability": 0.95, "segments": segs}


class BrightBoxReader:
    def __init__(self):
        self.calls = 0

    def info(self) -> dict:
        return {"backend": "mock-bright-box"}

    def read(self, img) -> list[dict]:
        self.calls += 1
        ys, xs = np.nonzero(img.mean(axis=2) > 200)
        if xs.size == 0:
            return []
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        return [{"text": "TEXT ON SCREEN", "score": 0.95, "box": box, "poly": None}]


def make_backends(asr=None, ocr=None):
    asr, ocr = asr or FakeTranscriber(), ocr or BrightBoxReader()
    return Backends(LazyBackend(lambda: asr), LazyBackend(lambda: ocr)), asr, ocr


def run_one(clip: Path, root: Path, backends: Backends, **kw) -> dict:
    return process_video_multimodal(VideoInput(clip.stem, clip), root, CFG, kw.pop("speech_cfg", SCFG),
                                    kw.pop("ocr_cfg", OCFG), backends, **kw)


def test_per_video_artifacts_and_opening_summary(clips, tmp_path) -> None:
    backends, asr, ocr = make_backends()
    res = run_one(clips["talk"], tmp_path, backends)
    assert res["status"] == "ok" and res["features_status"] == "ok"
    out = tmp_path / "features" / "talk"
    for name in ("speech.json", "ocr.json", "segments_merged.csv", "words.csv", "text_observations.csv",
                 "opening_multimodal.json", "report.json"):
        assert (out / name).is_file(), name

    rows = read_csv(out / "segments_merged.csv")
    assert list(rows[0].keys()) == MERGED_COLUMNS
    base = read_csv(out / "segments.csv")
    assert len(rows) == len(base)
    for merged, seg in zip(rows, base):  # the existing segments and their visual/audio values are untouched
        assert all(merged[c] == seg[c] for c in SEGMENT_COLUMNS if c != "notes")
    by_start = {float(r["start_s"]): r for r in rows}
    assert by_start[0.5]["speech_text"] == "hey" and by_start[0.5]["speech_word_count"] == "1"
    assert by_start[1.5]["speech_present"] == "false" and by_start[1.5]["speech_word_count"] == "0"
    assert by_start[1.0]["on_screen_text"] == "TEXT ON SCREEN" and by_start[1.0]["text_new_count"] == "1"
    assert by_start[1.5]["text_present"] == "true" and by_start[1.5]["text_new_count"] == "0"
    assert by_start[0.5]["text_present"] == "false" and by_start[0.5]["text_event_count"] == "0"

    o = json.loads((out / "opening_multimodal.json").read_text())
    assert o["stage_status"] == "ok" and "not an engagement" in o["note"]
    assert o["speech"]["speech_detected"] is True and o["speech"]["first_word_s"] == 0.3
    assert o["speech"]["transcript"] == "hey look this" and o["speech"]["language"] == "en"
    assert o["text"]["first_text_s"] == pytest.approx(1.0, abs=0.01) and o["text"]["texts"] == ["TEXT ON SCREEN"]
    assert o["visual"]["stream_status"] == "available" and o["audio"]["first_non_silent_s"] is not None
    assert o["speech"]["status_coverage"]["available"] > 0.9
    assert asr.calls == 1 and ocr.calls == 8

    words = read_csv(out / "words.csv")
    assert list(words[0].keys()) == WORD_COLUMNS
    assert [(float(w["start_s"]), float(w["end_s"]), w["word"]) for w in words] == WORDS  # exact times kept
    seg_of = {int(r["segment_index"]): (float(r["start_s"]), float(r["end_s"])) for r in rows}
    for w in words:
        a, b = seg_of[int(w["segment_index"])]
        assert a <= (float(w["start_s"]) + float(w["end_s"])) / 2 < b

    obs = read_csv(out / "text_observations.csv")
    assert list(obs[0].keys()) == OBSERVATION_COLUMNS
    assert sorted({float(r["t_s"]) for r in obs}) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], abs=0.01)
    hits = [r for r in obs if r["text"]]
    assert [float(r["t_s"]) for r in hits] == pytest.approx([1.0, 1.5, 2.0], abs=0.01)
    assert all(r["text"] == "TEXT ON SCREEN" and float(r["x0"]) == pytest.approx(20 / 180, abs=0.02) for r in hits)
    empty = [r for r in obs if not r["text"]]
    assert all(r["sample_status"] == "available" and r["event_id"] == "" for r in empty)  # observed, nothing there
    for r in obs:
        a, b = seg_of[int(r["segment_index"])]
        assert a <= float(r["t_s"]) < b


def test_one_fps_ocr_maps_onto_existing_segments(clips, tmp_path) -> None:
    backends, _, ocr = make_backends()
    res = run_one(clips["talk"], tmp_path, backends, ocr_cfg=replace(OCFG, sample_fps=1.0, max_seconds=10))
    times = [f["t_s"] for f in res["ocr"]["frames"]]
    assert times == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=0.01) and ocr.calls == 4
    for r in res["rows"]:
        if any(r["start_s"] <= t < r["end_s"] for t in times):
            assert r["text_status"] == "available" and r["text_samples"] == 1, r
        else:
            assert r["text_status"] == "uncertain" and r["text_samples"] == 0 and "inferred between samples" in r["notes"], r
    seg = next(r for r in res["rows"] if r["start_s"] <= 2.6 < r["end_s"])
    assert seg["text_present"] is True and seg["text_status"] == "uncertain"  # held from 2.0, only inferred
    o = res["opening"]["text"]
    assert o["samples"] == 4 and o["sample_interval_s"] == 1.0 and o["text_detected"] is True


def test_missing_modalities_stay_blank_and_explicit(clips, tmp_path) -> None:
    backends, asr, _ = make_backends()
    res = run_one(clips["silent_video"], tmp_path, backends)
    assert res["status"] == "ok" and res["speech"]["status"] == "absent" and asr.calls == 0
    out = tmp_path / "features" / "silent_video"
    rows = read_csv(out / "segments_merged.csv")
    for r in rows:
        assert r["speech_status"] == "absent" and r["audio_status"] == "absent" and r["visual_status"] == "available"
        assert all(r[c] == "" for c in ("speech_present", "speech_coverage", "speech_word_count", "speech_text"))
    assert read_csv(out / "words.csv") == []
    o = res["opening"]
    assert o["speech"]["status"] == "absent" and o["speech"]["speech_detected"] is None  # not "no speech"
    assert o["text"]["text_detected"] is True

    res = run_one(clips["audio_only"], tmp_path, backends)
    assert res["status"] == "ok" and res["ocr"]["status"] == "absent"
    rows = read_csv(tmp_path / "features" / "audio_only" / "segments_merged.csv")
    assert all(r["text_status"] == "absent" and r["text_present"] == "" and r["text_coverage"] == "" for r in rows)
    assert all(r["brightness_mean"] == "" and r["rms"] != "" for r in rows)
    assert res["opening"]["text"]["text_detected"] is None
    assert read_csv(tmp_path / "features" / "audio_only" / "text_observations.csv") == []


def test_silent_audio_track_is_measured_no_speech(clips, tmp_path) -> None:
    backends, asr, _ = make_backends()
    res = run_one(clips["silent_audio"], tmp_path, backends)
    assert res["speech"]["status"] == "available" and res["speech"]["speech_detected"] is False
    assert res["speech"]["method"] == "skipped_silent_audio" and asr.calls == 0
    assert res["opening"]["speech"]["speech_detected"] is False  # measured, unlike a missing stream
    rows = read_csv(tmp_path / "features" / "silent_audio" / "segments_merged.csv")
    assert all(r["speech_present"] == "false" and r["speech_coverage"] == "0.0" for r in rows)


def test_unreadable_file_is_uncertain_not_absent(clips, tmp_path) -> None:
    backends, asr, ocr = make_backends()
    res = run_one(clips["zero_bytes"], tmp_path, backends)
    assert res["status"] == "failed" and res["rows"] == []
    assert res["speech"]["status"] == "uncertain" and res["ocr"]["status"] == "uncertain"
    assert res["opening"]["speech"]["speech_detected"] is None and res["opening"]["text"]["text_detected"] is None
    assert asr.calls == 0 and ocr.calls == 0


def test_very_short_clip(clips, tmp_path) -> None:
    backends, _, _ = make_backends()
    res = run_one(clips["very_short"], tmp_path, backends)
    assert len(res["rows"]) == 1 and res["opening"]["shorter_than_opening_window"] is True
    assert res["ocr"]["sampling"]["frames"] == 1
    assert res["rows"][0]["text_status"] in ("available", "uncertain")


def test_model_failure_is_partial_not_absence(clips, tmp_path) -> None:
    def missing():
        raise BackendUnavailable("faster-whisper is not installed")
    backends = Backends(LazyBackend(missing), LazyBackend(BrightBoxReader))
    res = run_one(clips["talk"], tmp_path, backends)
    assert res["status"] == "partial"
    assert res["speech"]["status"] == "failed" and res["speech"]["speech_detected"] is None
    assert all(r["speech_status"] == "failed" and r["speech_present"] is None for r in res["rows"])
    assert res["opening"]["speech"]["speech_detected"] is None


def test_caching_config_changes_and_failure_retry(clips, tmp_path) -> None:
    clip = clips["talk"]
    before = sha(clip)
    backends, asr, ocr = make_backends()
    first = run_one(clip, tmp_path, backends)
    assert first["cache"] == {"features": False, "speech": False, "ocr": False, "librosa": False, "gemini": None}
    report_bytes = (tmp_path / "features" / "talk" / "report.json").read_bytes()

    second = run_one(clip, tmp_path, backends)
    assert second["cache"] == {"features": True, "speech": True, "ocr": True, "librosa": True, "gemini": None}
    assert (asr.calls, ocr.calls) == (1, 8)
    assert (tmp_path / "features" / "talk" / "report.json").read_bytes() == report_bytes  # features untouched

    third = run_one(clip, tmp_path, backends, speech_cfg=replace(SCFG, model="base"))
    assert third["cache"]["speech"] is False and third["cache"]["ocr"] is True and asr.calls == 2
    fourth = run_one(clip, tmp_path, backends, speech_cfg=replace(SCFG, model="base"), ocr_cfg=replace(OCFG, sample_fps=1.0))
    assert fourth["cache"]["speech"] is True and fourth["cache"]["ocr"] is False and ocr.calls == 12
    assert fourth["ocr"]["sampling"]["fps"] == 1.0
    fifth = run_one(clip, tmp_path, backends, speech_cfg=replace(SCFG, model="base", device="cuda"),
                    ocr_cfg=replace(OCFG, sample_fps=1.0), force=True)
    assert fifth["cache"]["speech"] is False and fifth["cache"]["features"] is True

    crashing = Backends(LazyBackend(lambda: FakeTranscriber(exc=RuntimeError("boom"))), backends.ocr)
    failed = run_one(clip, tmp_path, crashing, speech_cfg=replace(SCFG, beam_size=1))
    assert failed["speech"]["status"] == "failed"
    retried = run_one(clip, tmp_path, backends, speech_cfg=replace(SCFG, beam_size=1))
    assert retried["cache"]["speech"] is False and retried["speech"]["status"] == "available"
    assert sha(clip) == before


def test_skipped_stage_is_uncertain_not_missing(clips, tmp_path) -> None:
    backends, _, ocr = make_backends()
    res = run_one(clips["talk"], tmp_path, backends, run_ocr=False)
    assert res["ocr"] is None and ocr.calls == 0 and res["status"] == "ok"
    assert all(r["text_status"] == "uncertain" and "not extracted" in r["notes"] for r in res["rows"])
    assert res["opening"]["text"]["text_detected"] is None
    assert not (tmp_path / "features" / "talk" / "ocr.json").exists()


def test_batch_run_writes_run_outputs_and_never_touches_inputs(clips, tmp_path) -> None:
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in ("talk", "silent_video", "audio_only", "zero_bytes"):
        shutil.copy(clips[name], inputs / f"{name}.mp4")
    before = {p.name: sha(p) for p in inputs.iterdir()}
    backends, _, _ = make_backends()
    logs: list[str] = []
    summary = run_multimodal([inputs], tmp_path / "artifacts", CFG, SCFG, OCFG, backends, log=logs.append)
    assert summary["counts"] == {"videos": 4, "ok": 3, "partial": 0, "failed": 1}
    assert len(logs) == 4 and any("zero_bytes: failed" in line for line in logs)
    feats = tmp_path / "artifacts" / "features"
    all_rows = read_csv(feats / "segments_merged_all.csv")
    assert list(all_rows[0].keys()) == MERGED_COLUMNS
    assert {r["video_id"] for r in all_rows} == {"talk", "silent_video", "audio_only"}
    opening = read_csv(feats / "first5s_multimodal.csv")
    assert list(opening[0].keys()) == OPENING_MM_COLUMNS and len(opening) == 4
    silent = next(r for r in opening if r["video_id"] == "silent_video")
    assert silent["speech_status"] == "absent" and silent["speech_detected"] == "" and silent["word_count"] == ""
    talk = next(r for r in opening if r["video_id"] == "talk")
    assert talk["speech_detected"] == "true" and talk["texts"] == "TEXT ON SCREEN"
    run = json.loads((feats / "speech_ocr_run_summary.json").read_text())
    assert run["config"]["speech"]["model"] == "small" and len(run["videos"]) == 4
    assert {p.name: sha(p) for p in inputs.iterdir()} == before  # nothing written next to the originals
    assert not (tmp_path / "artifacts" / "cache" / "_ocr_tmp").exists() or not any(
        (tmp_path / "artifacts" / "cache" / "_ocr_tmp").iterdir())


def test_cli_without_models_installed_fails_gracefully(clips, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # simulate not installed, never download
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in ("talk", "silent_video"):
        shutil.copy(clips[name], inputs / f"{name}.mp4")
    spec = importlib.util.spec_from_file_location("run_speech_ocr", PROJECT_ROOT / "scripts" / "run_speech_ocr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = mod.main(["--input", str(inputs), "--artifacts", str(tmp_path / "artifacts"), "--skip-librosa"])
    out = capsys.readouterr().out
    assert code == 1
    assert "speech model unavailable: faster-whisper is not installed" in out
    assert "ocr model unavailable: paddleocr/paddlepaddle are not installed" in out
    run = json.loads((tmp_path / "artifacts" / "features" / "speech_ocr_run_summary.json").read_text())
    by_id = {v["video_id"]: v for v in run["videos"]}
    assert by_id["talk"]["status"] == "failed" and by_id["talk"]["speech_detected"] is None
    assert by_id["silent_video"]["status"] == "partial" and by_id["silent_video"]["speech_status"] == "absent"


GEMINI_RESPONSE = {
    "video_summary": "Grey frame with a white box and a steady tone.",
    "opening_hook": {"assessment": "Plain opening.", "evidence": [
        {"start_s": 0.0, "end_s": 1.0, "kind": "observation", "modality": "visual", "description": "grey frame",
         "confidence": 0.9}]},
    "narrative_progression": "None.", "pacing": "Static.",
    "moments": [{"start_s": 1.0, "end_s": 2.4, "kind": "observation", "modality": "visual", "category": "visual_change",
                 "description": "a white box appears", "confidence": 0.9}],
    "strengths": [], "weaknesses": [],
    "edit_suggestions": [{"start_s": 0.0, "end_s": 1.0, "action": "Show the box earlier.", "rationale": "Nothing changes.",
                          "expected_effect": "The change arrives sooner.", "confidence": 0.4, "evidence_moment_indices": [0]}],
    "uncertainties": [],
}


class FakeGemini:
    def __init__(self, responses=None):
        self.calls, self.responses = 0, list(responses or [])

    def analyse(self, video_path, prompt, schema, cfg):
        self.calls += 1
        r = self.responses.pop(0) if self.responses else GEMINI_RESPONSE
        if isinstance(r, Exception):
            raise r
        return json.dumps(r), {"total_token_count": 1000}


def test_librosa_gemini_and_recommendations_join_the_timeline(clips, tmp_path, monkeypatch) -> None:
    pytest.importorskip("librosa")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    clip = clips["talk"]
    before = sha(clip)
    backends, _, _ = make_backends()
    fake = FakeGemini()
    backends.gemini = lambda: fake
    gcfg = GeminiConfig(mode="remote", allow_remote=True)
    first = run_one(clip, tmp_path, backends, gemini_cfg=gcfg)
    out = tmp_path / "features" / "talk"
    assert fake.calls == 1 and first["cache"]["gemini"] is False and first["cache"]["librosa"] is False
    assert first["librosa"]["status"] == "available" and first["gemini"]["status"] == "available"

    rows = read_csv(out / "segments_merged.csv")
    assert list(rows[0]) == MERGED_COLUMNS
    assert {r["librosa_status"] for r in rows} == {"available"} and all(r["librosa_rms_db"] for r in rows)
    assert float(rows[0]["librosa_centroid_hz"]) == pytest.approx(300, rel=0.1)  # 300 Hz sine
    by_start = {float(r["start_s"]): r for r in rows}
    assert by_start[1.5]["gemini_moment_ids"] == "m0" and by_start[3.0]["gemini_moment_ids"] == ""
    recs = json.loads((out / "recommendations.json").read_text())
    assert recs["evidence_problems"] == [] and recs["inputs"]["gemini"] == "available"
    gem_recs = [r for r in recs["recommendations"] if r["source"] == "gemini"]
    assert len(gem_recs) == 1 and {"type": "gemini_moment", "id": "m0"} in gem_recs[0]["evidence"]
    assert gem_recs[0]["id"] in by_start[0.0]["recommendation_ids"].split("|")
    words = read_csv(out / "words.csv")
    assert [w["word"] for w in words] == ["hey", "look", "this"]  # Whisper words untouched by Gemini

    second = run_one(clip, tmp_path, backends, gemini_cfg=gcfg)
    assert fake.calls == 1 and second["cache"]["gemini"] is True and second["cache"]["librosa"] is True
    third = run_one(clip, tmp_path, backends, gemini_cfg=replace(gcfg, model="another-model"))
    assert fake.calls == 2 and third["cache"]["gemini"] is False
    good = (out / "gemini.json").read_bytes()

    class Err(Exception):
        code = 400

    backends.gemini = lambda: FakeGemini([Err("bad request")])
    failed = run_one(clip, tmp_path, backends, gemini_cfg=replace(gcfg, model="third-model"))
    assert failed["gemini"]["status"] == "failed" and failed["cache"]["gemini_written"] is False
    assert (out / "gemini.json").read_bytes() == good  # a failure never overwrites a usable result
    assert {r["gemini_status"] for r in read_csv(out / "segments_merged.csv")} == {"failed"}
    assert failed["status"] == "partial"
    assert sha(clip) == before


def test_gemini_off_by_default_and_never_called(clips, tmp_path) -> None:
    backends, _, _ = make_backends()
    backends.gemini = lambda: pytest.fail("Gemini transport created while disabled")
    res = run_one(clips["talk"], tmp_path, backends)
    assert res["gemini"] is None and not (tmp_path / "features" / "talk" / "gemini.json").exists()
    rows = read_csv(tmp_path / "features" / "talk" / "segments_merged.csv")
    assert {r["gemini_status"] for r in rows} == {"uncertain"} and "gemini=not run (disabled)" in rows[0]["notes"]


class FakeSiglip:
    logit_scale, logit_bias = 10.0, -5.0

    def __init__(self, fail: bool = False):
        self.fail, self.image_calls = fail, 0

    def info(self):
        return {"backend": "fake", "device": "cpu", "mps_fallback": None}

    def encode_images(self, images):
        self.image_calls += 1
        if self.fail:
            raise RuntimeError("inference exploded")
        v = np.array([[*(im.reshape(-1, 3).mean(0) / 255), 0.2] for im in images], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode_texts(self, texts):
        v = np.eye(4, dtype=np.float32)[np.arange(len(texts)) % 4] + 0.1
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_siglip_is_off_by_default(clips, tmp_path) -> None:
    backends, _, _ = make_backends()
    backends.siglip = LazyBackend(lambda: pytest.fail("SigLIP loaded while disabled"))
    res = run_one(clips["talk"], tmp_path, backends)
    out = tmp_path / "features" / "talk"
    assert res["siglip"] is None and "siglip" not in res["cache"]
    assert not (out / "siglip.json").exists() and not (out / "siglip_embeddings.npy").exists()
    rows = read_csv(out / "segments_merged.csv")
    assert {r["siglip_status"] for r in rows} == {"uncertain"} and all(r["siglip_score_text_screen"] == "" for r in rows)


def test_siglip_joins_timeline_caches_and_keeps_good_artifact(clips, tmp_path) -> None:
    clip = clips["talk"]
    before = sha(clip)
    backends, _, _ = make_backends()
    loads = []
    model = FakeSiglip()
    backends.siglip = LazyBackend(lambda: loads.append(1) or model)
    first = run_one(clip, tmp_path, backends, run_siglip=True)
    out = tmp_path / "features" / "talk"
    art = json.loads((out / "siglip.json").read_text())
    assert first["siglip"]["status"] == "available" and first["cache"]["siglip"] is False and loads == [1]
    frame_times = [float(r["time_s"]) for r in read_csv(out / "frames.csv")]
    assert all(frame_times[f["frame_index"]] == pytest.approx(f["t_s"]) for f in art["frames"])
    assert np.load(out / "siglip_embeddings.npy").shape == (len(art["frames"]), 4)
    rows = read_csv(out / "segments_merged.csv")
    assert list(rows[0]) == MERGED_COLUMNS
    for r in rows:  # every segment's frame ids exist and lie inside it
        ids = [int(i) for i in r["siglip_frame_ids"].split("|") if i]
        assert ids and all(float(r["start_s"]) <= frame_times[i] < float(r["end_s"]) + 1e-9 for i in ids)
        assert r["siglip_status"] == "available" and 0 <= float(r["siglip_score_text_screen"]) <= 1

    second = run_one(clip, tmp_path, backends, run_siglip=True)
    assert second["cache"]["siglip"] is True and loads == [1] and model.image_calls == 1  # no reload, no inference

    good = (out / "siglip.json").read_bytes(), (out / "siglip_embeddings.npy").read_bytes()
    failing = make_backends()[0]
    failing.siglip = LazyBackend(lambda: FakeSiglip(fail=True))
    failed = run_one(clip, tmp_path, failing, run_siglip=True, siglip_cfg=SiglipConfig(sample_fps=5.0))
    assert failed["siglip"]["status"] == "failed" and "inference exploded" in failed["siglip"]["reason"]
    assert failed["cache"]["siglip_written"] is False and failed["status"] == "partial"
    assert ((out / "siglip.json").read_bytes(), (out / "siglip_embeddings.npy").read_bytes()) == good
    assert {r["siglip_status"] for r in read_csv(out / "segments_merged.csv")} == {"failed"}
    assert sha(clip) == before
