"""SigLIP stage with a fake embedder and fake frame reader: no model download, no FFmpeg, no network."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hookline import siglip as sg
from hookline.config import SiglipConfig
from hookline.context import BackendUnavailable, LazyBackend, MediaContext

CFG = SiglipConfig()
D = 8


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class FakeEmbedder:
    """Image embedding from the frame's mean colour; fixed text embeddings."""

    logit_scale, logit_bias = 10.0, -5.0

    def __init__(self, image_fn=None, text_fn=None):
        self.image_calls = self.text_calls = 0
        self.image_fn, self.text_fn = image_fn, text_fn

    def info(self):
        return {"backend": "fake", "device": "cpu", "mps_fallback": None}

    def encode_images(self, images):
        self.image_calls += 1
        if self.image_fn:
            return self.image_fn(images)
        return unit([[*(img.reshape(-1, 3).mean(0) / 255), 0.1, 0, 0, 0, 0] for img in images])

    def encode_texts(self, texts):
        self.text_calls += 1
        if self.text_fn:
            return self.text_fn(texts)
        rng = np.random.default_rng(0)
        return unit(rng.normal(size=(len(texts), D)))


def colour_frames(colours):
    """Proxy frames (BGR) with the given RGB colour per frame."""
    return [np.full((32, 18, 3), c[::-1], np.uint8) for c in colours]


def reader_for(frames, errors=()):
    class Reader:
        def __init__(self, proxy, w, h):
            self.errors = list(errors)

        def __iter__(self):
            yield from frames
    return Reader


def segs(bounds, visual="available"):
    return [{"segment_index": i, "start_s": a, "end_s": b, "visual_status": visual} for i, (a, b) in enumerate(bounds)]


def ctx(tmp_path, n_frames=30, cuts=(), bounds=None, **kw) -> MediaContext:
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"x")
    dur = n_frames / 10
    bounds = bounds or [(round(i * 0.5, 6), round(min((i + 1) * 0.5, dur), 6)) for i in range(int(np.ceil(dur / 0.5)))]
    base = dict(video_id="v", source_path="v.mp4", source_sha256="sha", prep_key="prep", duration=dur,
                visual_stream="available", proxy_path=str(proxy), proxy_size=[18, 32], proxy_fps=10.0,
                segments=segs(bounds), scene_cut_times=list(cuts))
    return MediaContext(**(base | kw))


TIMES = [round(i / 10, 6) for i in range(30)]


def run(c, frames, embedder=None, cfg=CFG, times=TIMES):
    emb = embedder or FakeEmbedder()
    art, e = sg.extract_siglip(c, cfg, LazyBackend(lambda: emb), times, reader_factory=reader_for(frames))
    return art, e, emb


# ------------------------------------------------------------- frame selection and alignment

def test_frame_selection_grid_plus_segment_cover() -> None:
    s = segs([(0.0, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.5)])  # 0.5-0.7 is the grid's frame; 0.7-1.0 has none
    idx = sg.frame_selection([i / 10 for i in range(15)], s, 10.0, CFG)
    assert idx == [0, 5, 8, 10]  # 8 = frame nearest the 0.85 s midpoint of the uncovered segment
    assert sg.frame_selection([i / 10 for i in range(15)], s, 10.0, replace(CFG, cover_every_segment=False)) == [0, 5, 10]
    assert sg.frame_selection([], s, 10.0, CFG) == []
    assert sg.frame_selection([i / 10 for i in range(5)], s[:1], 10.0, replace(CFG, sample_fps=20.0)) == [0, 1, 2, 3, 4]


def test_frames_keep_original_timestamps_and_map_to_segments(tmp_path) -> None:
    colours = [(200, 0, 0)] * 15 + [(0, 0, 200)] * 15  # content changes at 1.5 s, a detected cut there
    art, emb, model = run(ctx(tmp_path, cuts=[1.5]), colour_frames(colours))
    assert art["status"] == "available" and art["measured"] and emb.shape == (6, D)
    assert [(f["frame_index"], f["t_s"]) for f in art["frames"]] == [(i, TIMES[i]) for i in (0, 5, 10, 15, 20, 25)]
    by_seg = {r["segment_index"]: r for r in art["segments"]}
    assert [by_seg[i]["siglip_frame_ids"] for i in range(6)] == [[0], [5], [10], [15], [20], [25]]
    cut_frame = next(f for f in art["frames"] if f["frame_index"] == 15)
    assert cut_frame["crosses_cut"] is True and cut_frame["change"] > 0.5
    # The change across the cut is reported separately from within-shot change.
    assert by_seg[3]["siglip_cut_change"] == pytest.approx(cut_frame["change"], abs=1e-4)
    assert by_seg[3]["siglip_within_shot_change"] is None
    assert by_seg[1]["siglip_within_shot_change"] == pytest.approx(0.0, abs=1e-4) and by_seg[1]["siglip_cut_change"] is None
    assert by_seg[0]["siglip_within_shot_change"] is None  # first frame has no predecessor
    assert model.text_calls == 1 and model.image_calls == 1  # one batch, text encoded once


def test_scores_are_sigmoid_of_scaled_cosine_and_labelled_as_scores(tmp_path) -> None:
    art, emb, model = run(ctx(tmp_path), colour_frames([(10, 200, 30)] * 30))
    text = model.encode_texts([""] * len(sg.SIGLIP_LABELS))
    want = 1 / (1 + np.exp(-(emb[0] @ text.T * 10.0 - 5.0)))
    got = [art["frames"][0]["scores"][k] for k, _ in sg.SIGLIP_LABELS]
    assert got == pytest.approx(want, abs=1e-5)
    assert all(0 <= v <= 1 for f in art["frames"] for v in f["scores"].values())
    assert "not calibrated probabilities" in art["score_note"]
    assert [lab["prompt"] for lab in art["labels"]][0] == "This is a photo of a person speaking directly to the camera."
    row = art["segments"][0]
    assert row["siglip_top_label"] == max((k for k, _ in sg.SIGLIP_LABELS), key=lambda k: row[f"siglip_score_{k}"])


def test_short_segment_and_last_frame_at_end(tmp_path) -> None:
    times = [0.0, 0.1, 0.2]
    c = ctx(tmp_path, n_frames=3, bounds=[(0.0, 0.2)])
    c.duration = 0.2
    art, _, _ = run(c, colour_frames([(1, 2, 3)] * 3), times=times)
    assert [f["frame_index"] for f in art["frames"]] == [0]
    assert art["segments"][0]["siglip_frames"] == 1


# ------------------------------------------------------------- validation and failures

@pytest.mark.parametrize("image_fn,text_fn,reason", [
    (lambda imgs: np.ones((len(imgs), D)), None, "not L2-normalised"),
    (lambda imgs: np.full((len(imgs), D), np.nan), None, "NaN"),
    (lambda imgs: unit(np.ones((len(imgs) + 1, D))), None, "expected"),
    (None, lambda t: unit(np.ones((len(t), D + 1))), "embedding sizes differ"),
    (lambda imgs: (_ for _ in ()).throw(RuntimeError("MPS kernel missing")), None, "inference failed: RuntimeError"),
])
def test_invalid_model_output_fails_without_values(tmp_path, image_fn, text_fn, reason) -> None:
    art, emb, _ = run(ctx(tmp_path), colour_frames([(5, 5, 5)] * 30), FakeEmbedder(image_fn, text_fn))
    assert art["status"] == "failed" and not art["measured"] and emb is None and reason in art["reason"]
    assert art["frames"] == [] and art["segments"] == []


def test_missing_video_proxy_frames_and_model(tmp_path) -> None:
    for stream in ("absent", "uncertain", "failed"):
        art, _, model = run(ctx(tmp_path, visual_stream=stream), colour_frames([(1, 1, 1)] * 30))
        assert art["status"] == stream and model.image_calls == 0
    art, _, _ = run(ctx(tmp_path, proxy_path=str(tmp_path / "missing.mp4")), [])
    assert art["status"] == "failed" and "proxy missing" in art["reason"]
    art, _, _ = run(ctx(tmp_path), [], times=[])
    assert art["status"] == "failed" and "no proxy frames" in art["reason"]
    art, _, model = run(ctx(tmp_path), [])  # proxy decodes to nothing (corrupt)
    assert art["status"] == "failed" and "no selected proxy frame could be decoded" in art["reason"] and model.image_calls == 0
    loads = []

    def broken():
        loads.append(1)
        raise OSError("weights not found")
    art, _ = sg.extract_siglip(ctx(tmp_path), CFG, LazyBackend(broken), TIMES, reader_factory=reader_for(colour_frames([(1, 1, 1)] * 30)))
    assert art["status"] == "failed" and "model unavailable" in art["reason"] and loads == [1]
    art, _ = sg.extract_siglip(ctx(tmp_path), CFG, None, TIMES, reader_factory=reader_for([]))
    assert art["status"] == "failed"


def test_partially_decoded_proxy_warns_and_uses_what_decoded(tmp_path) -> None:
    art, emb, _ = run(ctx(tmp_path), colour_frames([(9, 9, 9)] * 12))  # only frames 0-11 decode
    assert art["status"] == "available" and [f["frame_index"] for f in art["frames"]] == [0, 5, 10]
    assert any("could not be decoded" in w for w in art["warnings"])


# ------------------------------------------------------------- caching and writing

def test_cache_key_inputs(tmp_path, monkeypatch) -> None:
    c = ctx(tmp_path)
    times = [0.0, 0.5, 1.0]
    key = sg.siglip_key(c, CFG, times)
    assert sg.siglip_key(c, replace(CFG, device="cpu", batch_size=2), times) == key  # runtime-only settings
    for other in (replace(CFG, model_revision="abc"), replace(CFG, model_id="google/other"), replace(CFG, sample_fps=1.0),
                  replace(CFG, prompt_template="a photo of {}")):
        assert sg.siglip_key(c, other, times) != key
    assert sg.siglip_key(c, CFG, [0.0, 0.5, 1.1]) != key
    assert sg.siglip_key(replace(c, prep_key="other-proxy"), CFG, times) != key
    assert sg.siglip_key(replace(c, source_sha256="other"), CFG, times) != key
    monkeypatch.setattr(sg, "library_versions", lambda: {"torch": "0", "transformers": "0", "numpy": "0"})
    assert sg.siglip_key(c, CFG, times) != key


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_failure_never_replaces_a_good_artifact(tmp_path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    art, emb, _ = run(ctx(tmp_path), colour_frames([(3, 4, 5)] * 30))
    assert sg.write_siglip(out, art, emb, write_json) is True
    good_json, good_npy = (out / "siglip.json").read_bytes(), (out / sg.EMBEDDINGS_FILE).read_bytes()
    saved = np.load(out / sg.EMBEDDINGS_FILE)
    assert saved.dtype == np.float16 and saved.shape == (6, D)
    assert json.loads(good_json)["embedding"]["sha256"]
    bad, _, _ = run(ctx(tmp_path), colour_frames([(3, 4, 5)] * 30), FakeEmbedder(lambda i: np.ones((len(i), D))))
    assert sg.write_siglip(out, bad, None, write_json) is False
    assert (out / "siglip.json").read_bytes() == good_json and (out / sg.EMBEDDINGS_FILE).read_bytes() == good_npy
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert sg.write_siglip(fresh, bad, None, write_json) is True  # nothing to preserve: record the failure
    assert not (fresh / sg.EMBEDDINGS_FILE).exists()


# ------------------------------------------------------------- timeline columns and evidence ids

def test_window_statuses_and_evidence_ids_exist(tmp_path) -> None:
    c = ctx(tmp_path)
    art, _, _ = run(c, colour_frames([(3, 4, 5)] * 30))
    w, notes = sg.siglip_window(art, c.segments[2])
    assert w["siglip_status"] == "available" and w["siglip_frame_ids"] == [10] and notes == []
    assert set(w) == set(sg.SIGLIP_COLUMNS)
    ids = {i for r in art["segments"] for i in r["siglip_frame_ids"] or []}
    assert ids == {f["frame_index"] for f in art["frames"]} and all(0 <= i < len(TIMES) for i in ids)
    w, _ = sg.siglip_window(art, c.segments[2] | {"visual_status": "failed"})
    assert w["siglip_status"] == "failed" and w["siglip_score_text_screen"] is None
    w, notes = sg.siglip_window(None, c.segments[0])
    assert w["siglip_status"] == "uncertain" and notes == ["siglip=not run (disabled)"]
    failed = sg._base(c, CFG, "k") | {"status": "failed"}
    w, _ = sg.siglip_window(failed, c.segments[0])
    assert w["siglip_status"] == "failed" and all(w[k] is None for k in sg.SIGLIP_COLUMNS[1:])


# ------------------------------------------------------------- device selection and MPS fallback

def fake_torch(available: bool, built: bool = True):
    mps = types.SimpleNamespace(is_available=lambda: available, is_built=lambda: built)
    return types.SimpleNamespace(backends=types.SimpleNamespace(mps=mps))


def test_device_selection() -> None:
    assert sg.select_device("auto", fake_torch(True)) == "mps"
    assert sg.select_device("auto", fake_torch(False)) == "cpu"
    assert sg.select_device("auto", fake_torch(True, built=False)) == "cpu"
    assert sg.select_device("cpu", fake_torch(True)) == "cpu"
    assert sg.select_device("auto", types.SimpleNamespace()) == "cpu"
    with pytest.raises(BackendUnavailable):
        sg.select_device("mps", fake_torch(False))


def test_unsupported_mps_operation_falls_back_to_cpu_once() -> None:
    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    moved = []
    emb = sg.HfSiglipEmbedder.__new__(sg.HfSiglipEmbedder)
    emb._torch = types.SimpleNamespace(inference_mode=Ctx)
    emb._model = types.SimpleNamespace(to=moved.append)
    emb.device, emb.mps_fallback = "mps", None
    calls = []

    def fn():
        calls.append(emb.device)
        if emb.device == "mps":
            raise NotImplementedError("aten::op not implemented for MPS")
        return np.zeros((1, D))
    assert emb._run(fn).shape == (1, D)
    assert calls == ["mps", "cpu"] and moved == ["cpu"] and "not implemented" in emb.mps_fallback
    emb.device = "cpu"
    with pytest.raises(RuntimeError):
        emb._run(lambda: (_ for _ in ()).throw(RuntimeError("cpu failure")))  # no fallback from CPU


def test_missing_torch_is_a_clear_backend_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(BackendUnavailable, match="torch/transformers are not installed"):
        sg.HfSiglipEmbedder(CFG)


def test_config_validation() -> None:
    for bad in (dict(sample_fps=0), dict(batch_size=0), dict(device="cuda"), dict(prompt_template="no placeholder")):
        with pytest.raises(ValueError):
            SiglipConfig(**bad)
    assert CFG.model_id == "google/siglip2-base-patch16-224" and len(CFG.model_revision) == 40
