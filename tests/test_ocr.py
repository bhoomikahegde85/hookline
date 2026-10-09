"""On-screen-text stage with mocked OCR readers (no model downloads)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hookline.config import OcrConfig
from hookline.context import BackendUnavailable, LazyBackend, MediaContext
from hookline.media import probe_video
from hookline.ocr import (
    TEXT_COLUMNS,
    PaddleOcrReader,
    build_events,
    extract_ocr,
    observation_rows,
    ocr_key,
    opening_text,
    parse_paddle_result,
    text_window,
)

CFG = OcrConfig()
has_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg not installed")
BOX_A = [0.1, 0.1, 0.6, 0.2]
BOX_B = [0.1, 0.8, 0.6, 0.9]


def frame(t, texts=(), status="available"):
    dets = [{"text": text, "score": 0.9, "box_norm": list(box)} for text, box in texts]
    return {"t_s": t, "status": status, "detections": dets}


# ---------------------------------------------------------------- parsing

def test_parse_paddleocr_3x_predict_result() -> None:
    res = [{
        "rec_texts": ["SALE 50%", "शुभ दिन"],
        "rec_scores": np.array([0.98, 0.87]),
        "rec_polys": [np.array([[10, 20], [110, 20], [110, 50], [10, 50]]), np.array([[5, 60], [90, 60], [90, 80], [5, 80]])],
        "rec_boxes": np.array([[10, 20, 110, 50], [5, 60, 90, 80]]),
    }]
    dets = parse_paddle_result(res)
    assert [d["text"] for d in dets] == ["SALE 50%", "शुभ दिन"]
    assert dets[0]["box"] == [10.0, 20.0, 110.0, 50.0] and dets[1]["score"] == pytest.approx(0.87)
    assert dets[0]["poly"][2] == [110.0, 50.0]


def test_parse_paddleocr_2x_result_and_empty() -> None:
    res = [[[[[10, 20], [110, 22], [108, 50], [9, 48]], ("Hello", 0.91)]]]
    dets = parse_paddle_result(res)
    assert dets == [{"text": "Hello", "score": 0.91, "box": [9.0, 20.0, 110.0, 50.0],
                     "poly": [[10.0, 20.0], [110.0, 22.0], [108.0, 50.0], [9.0, 48.0]]}]
    assert parse_paddle_result([None]) == [] and parse_paddle_result(None) == []
    assert parse_paddle_result([{"rec_texts": [], "rec_scores": []}]) == []


# ---------------------------------------------------------------- adapter

def fake_paddleocr(monkeypatch) -> dict:
    calls: dict = {}

    class PaddleOCR:
        def __init__(self, **kw):
            calls["init"] = kw

        def _get_ocr_model_names(self, lang, ver):
            return ("PP-OCRv5_server_det", "devanagari_PP-OCRv5_mobile_rec") if lang == "hi" else (None, None)

        def predict(self, img):
            calls["predict_shape"] = img.shape
            return [{"rec_texts": ["Hi"], "rec_scores": [0.9], "rec_polys": [[[0, 0], [4, 0], [4, 2], [0, 2]]],
                     "rec_boxes": [[0, 0, 4, 2]]}]

    mod = types.ModuleType("paddleocr")
    mod.__version__ = "3.7.0"
    mod.PaddleOCR = PaddleOCR
    paddle = types.ModuleType("paddle")
    paddle.__version__ = "3.0.0"
    monkeypatch.setitem(sys.modules, "paddleocr", mod)
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    return calls


def test_paddle_adapter_refuses_to_download_by_default(monkeypatch, tmp_path) -> None:
    calls = fake_paddleocr(monkeypatch)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    with pytest.raises(BackendUnavailable, match="--download-models"):
        PaddleOcrReader(CFG)
    assert "init" not in calls  # never constructed, so nothing was fetched
    PaddleOcrReader(CFG, allow_download=True)
    assert calls["init"]["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert calls["init"]["use_doc_unwarping"] is False and calls["init"]["use_textline_orientation"] is False


def test_paddle_adapter_uses_local_models_and_lang_mapping(monkeypatch, tmp_path) -> None:
    calls = fake_paddleocr(monkeypatch)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(tmp_path))
    for name in ("PP-OCRv5_server_det", "devanagari_PP-OCRv5_mobile_rec"):
        (tmp_path / "official_models" / name).mkdir(parents=True)
    reader = PaddleOcrReader(replace(CFG, det_model=None, rec_model=None, lang="hi"))
    assert calls["init"]["lang"] == "hi" and "text_detection_model_name" not in calls["init"]
    assert reader.info()["rec_model"] == "devanagari_PP-OCRv5_mobile_rec"
    assert reader.read(np.zeros((8, 8, 3), np.uint8)) == [
        {"text": "Hi", "score": 0.9, "box": [0.0, 0.0, 4.0, 2.0], "poly": [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]}]


def test_paddle_missing_package(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(BackendUnavailable, match="not installed"):
        PaddleOcrReader(CFG)


def test_ocr_config_validation_and_cache_key() -> None:
    with pytest.raises(ValueError):
        OcrConfig(det_model="x", rec_model=None)
    with pytest.raises(ValueError):
        OcrConfig(sample_fps=0)
    ctx = MediaContext(video_id="v", source_path="/x", source_sha256="s", prep_key="k", duration=1.0)
    base = ocr_key(ctx, CFG)
    assert ocr_key(ctx, replace(CFG, device="gpu:0")) == base
    assert ocr_key(ctx, replace(CFG, sample_fps=1.0)) != base
    assert ocr_key(ctx, replace(CFG, rec_model="latin_PP-OCRv5_mobile_rec")) != base
    assert ocr_key(ctx, replace(CFG, min_score=0.7)) != base
    assert ocr_key(replace(ctx, source_sha256="t"), CFG) != base


# ---------------------------------------------------------------- events

def test_repeated_text_is_one_event_and_reappearance_links_back() -> None:
    frames = [
        frame(0.0, [("SALE 50%", BOX_A)]),
        frame(0.5, [("SALE 5O%", BOX_A)]),   # OCR jitter: same event
        frame(1.0),                          # one missed sample is bridged
        frame(1.5, [("SALE 50%", BOX_A)]),
        frame(2.0), frame(2.5),
        frame(3.0, [("sale  50%", BOX_A)]),  # gone for 2 samples: new event, marked as a repeat
    ]
    events = build_events(frames, CFG, 0.5, 3.5)
    assert len(events) == 2
    first, again = events
    assert first["text"] == "SALE 50%" and first["frames"] == 3
    assert (first["first_seen_s"], first["last_seen_s"], first["visible_until_s"]) == (0.0, 1.5, 2.0)
    assert again["repeat_of"] == 0 and again["first_seen_s"] == 3.0 and first["repeat_of"] is None
    assert [d["event_id"] for f in frames for d in f["detections"]] == [0, 0, 0, 1]
    assert "text" not in frames[0]["detections"][0]  # stored once on the event
    assert frames[1]["detections"][0]["text"] == "SALE 5O%"  # variant kept per frame


def test_different_text_or_place_are_separate_events() -> None:
    frames = [
        frame(0.0, [("Hello", BOX_A), ("Hello", BOX_B)]),  # same caption twice on screen
        frame(0.5, [("World", BOX_A), ("Hello", BOX_B)]),  # caption changes in place
    ]
    events = build_events(frames, CFG, 0.5, 1.0)
    assert [(e["text"], e["first_seen_s"], e["repeat_of"]) for e in events] == [
        ("Hello", 0.0, None), ("Hello", 0.0, None), ("World", 0.5, None)]
    assert events[1]["last_seen_s"] == 0.5 and events[0]["last_seen_s"] == 0.0


def test_moving_text_stays_one_event_but_gaps_need_overlap() -> None:
    shifted = [0.4, 0.1, 0.9, 0.2]  # IoU with BOX_A is 0.25 (camera pan)
    far = [0.1, 0.7, 0.6, 0.8]
    frames = [frame(0.0, [("MILANO", BOX_A)]), frame(0.5, [("MILANO", shifted)]), frame(1.0), frame(1.5, [("MILANO", far)])]
    events = build_events(frames, CFG, 0.5, 2.0)
    assert [(e["first_seen_s"], e["last_seen_s"], e["repeat_of"]) for e in events] == [(0.0, 0.5, None), (1.5, 1.5, 0)]


def test_failed_frames_are_skipped_not_empty() -> None:
    frames = [frame(0.0, [("A", BOX_A)]), frame(0.5, status="failed"), frame(1.0, [("A", BOX_A)])]
    events = build_events(frames, CFG, 0.5, 1.5)
    assert len(events) == 1 and events[0]["frames"] == 2


# ---------------------------------------------------------------- windows

def art_from(frames, range_end, interval=0.5) -> dict:
    events = build_events(frames, CFG, interval, range_end)
    ok = any(f["status"] == "available" for f in frames)
    return {"status": "available" if ok else "failed", "measured": ok, "events": events, "frames": frames,
            "sampling": {"interval_s": interval, "fps": 1 / interval, "range_s": [0.0, range_end]}}


def seg(start, end, visual="available"):
    return {"start_s": start, "end_s": end, "visual_status": visual}


def test_text_window_values_and_empty_vs_failed() -> None:
    frames = [frame(0.0), frame(0.5, [("Hook!", BOX_A)]), frame(1.0, [("Hook!", BOX_A)]),
              frame(1.5, status="failed"), frame(2.0)]
    art = art_from(frames, 2.5)
    empty, notes = text_window(art, seg(0.0, 0.5), [0.0, 2.5], 0.5)
    assert empty["text_status"] == "available" and empty["text_present"] is False
    assert empty["text_event_count"] == 0 and empty["on_screen_text"] is None and notes == []
    hit, _ = text_window(art, seg(0.5, 1.0), [0.0, 2.5], 0.5)
    assert hit["text_present"] is True and hit["text_new_count"] == 1 and hit["on_screen_text"] == "Hook!"
    assert hit["text_area_ratio"] == pytest.approx(0.05)
    failed, _ = text_window(art, seg(1.5, 2.0), [0.0, 2.5], 0.5)
    assert failed["text_status"] == "failed" and failed["text_present"] is None and failed["text_coverage"] == 0.0
    straddle, _ = text_window(art, seg(1.25, 1.75), [0.0, 2.5], 0.6)  # its only sample (1.5) failed
    assert straddle["text_status"] == "failed"
    between, notes = text_window(art, seg(1.1, 1.4), [0.0, 2.5], 0.5)  # scene-cut segment with no sample inside
    assert between["text_status"] == "uncertain" and between["text_samples"] == 0
    assert between["text_present"] is True and between["on_screen_text"] == "Hook!"  # held from the 1.0 sample
    assert "inferred between samples" in notes[0]


def test_text_window_observed_values_come_from_samples_inside() -> None:
    frames = [frame(0.0, [("Title", BOX_A)]), frame(1.0, [("Title", BOX_A), ("Sub", BOX_B)]), frame(2.0)]
    art = art_from(frames, 3.0, interval=1.0)
    rows = {(a, b): text_window(art, seg(a, b), [0.0, 3.0], 0.5) for a, b in
            [(0.0, 0.5), (0.5, 1.0), (1.0, 1.3), (1.3, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]}
    status = {k: v[0]["text_status"] for k, v in rows.items()}
    assert status == {(0.0, 0.5): "available", (0.5, 1.0): "uncertain", (1.0, 1.3): "available",
                      (1.3, 1.5): "uncertain", (1.5, 2.0): "uncertain", (2.0, 2.5): "available", (2.5, 3.0): "uncertain"}
    obs = rows[(1.0, 1.3)][0]
    assert obs["text_samples"] == 1 and obs["on_screen_text"] == "Title | Sub" and obs["text_new_count"] == 1
    assert rows[(2.0, 2.5)][0]["text_present"] is False  # observed: nothing on screen at 2.0
    assert rows[(1.3, 1.5)][0]["text_event_count"] == 2 and rows[(1.3, 1.5)][0]["text_new_count"] == 0


def test_observation_rows_keep_sample_times_boxes_and_failures() -> None:
    frames = [frame(0.0), frame(1.0, [("Title", BOX_A)]), frame(2.0, status="failed")]
    frames[2]["error"] = "RuntimeError: boom"
    art = art_from(frames, 3.0, interval=1.0) | {"video_id": "v"}
    segments = [{"segment_index": i, "start_s": a, "end_s": b} for i, (a, b) in enumerate([(0.0, 0.5), (0.5, 1.2), (1.2, 3.0)])]
    rows = observation_rows(art, segments)
    assert [(r["t_s"], r["segment_index"], r["sample_status"], r.get("text")) for r in rows] == [
        (0.0, 0, "available", None), (1.0, 1, "available", "Title"), (2.0, 2, "failed", None)]
    assert (rows[1]["x0"], rows[1]["y0"], rows[1]["x1"], rows[1]["y1"]) == tuple(BOX_A) and rows[1]["event_id"] == 0
    assert rows[2]["error"] == "RuntimeError: boom"
    assert observation_rows(None, segments) == [] and observation_rows({"status": "absent", "frames": []}, segments) == []


def test_text_window_statuses() -> None:
    art = art_from([frame(0.0), frame(0.5)], 1.0)
    assert text_window(art, seg(0.0, 0.5, visual="absent"), [0.0, 1.0], 0.5)[0]["text_status"] == "absent"
    beyond, notes = text_window(art, seg(1.0, 1.5), [0.0, 2.0], 0.5)
    assert beyond["text_status"] == "uncertain" and "beyond OCR range" in notes[0]
    skipped, notes = text_window(None, seg(0.0, 0.5), [0.0, 1.0], 0.5)
    assert skipped["text_status"] == "uncertain" and "not extracted" in notes[0]
    not_measured = {"status": "failed", "measured": False}
    w, _ = text_window(not_measured, seg(0.0, 0.5), [0.0, 1.0], 0.5)
    assert w["text_status"] == "failed" and all(w[c] is None for c in TEXT_COLUMNS[1:])


def test_opening_text_only_claims_absence_when_fully_measured() -> None:
    art = art_from([frame(0.0), frame(0.5)], 1.0)
    o = opening_text(art, 1.0)
    assert o["text_detected"] is False and o["samples"] == 2 and o["sample_interval_s"] == 0.5  # none observed at 2 fps
    assert opening_text(art_from([frame(0.0), frame(0.5, status="failed")], 1.0), 1.0)["text_detected"] is None
    assert opening_text(art, 1.5)["text_detected"] is None  # OCR range ended before the opening did
    assert opening_text({"status": "failed", "measured": False}, 1.0)["text_detected"] is None
    art = art_from([frame(0.0), frame(0.5, [("Wait for it", BOX_A)]), frame(1.0, [("Wait for it", BOX_A)])], 1.5)
    o = opening_text(art, 1.5)
    assert o["text_detected"] is True and o["first_text_s"] == 0.5 and o["texts"] == ["Wait for it"]
    assert o["screen_coverage"] == pytest.approx(1.0 / 1.5, abs=1e-3)


def test_opening_texts_rank_persistent_text_first() -> None:
    art = art_from([frame(0.0, [("LOVALE", BOX_B)]), frame(0.5, [("Hook line", BOX_A)]), frame(1.0, [("Hook line", BOX_A)])], 1.5)
    o = opening_text(art, 1.5)
    assert o["texts"] == ["Hook line", "LOVALE"] and o["first_text_s"] == 0.0
    assert o["events"] == 2 and o["single_sample_events"] == 1


# ---------------------------------------------------------------- extraction on real clips

class BrightBoxReader:
    """Content-based mock: reports the bright rectangle as text, so timing comes from the video."""

    def __init__(self, fail_at: set[int] = frozenset(), text: str = "HELLO"):
        self.calls, self.fail_at, self.text = 0, set(fail_at), text

    def info(self) -> dict:
        return {"backend": "mock-bright-box"}

    def read(self, img: np.ndarray) -> list[dict]:
        self.calls += 1
        if self.calls - 1 in self.fail_at:
            raise RuntimeError("simulated OCR crash")
        ys, xs = np.nonzero(img.mean(axis=2) > 200)
        if xs.size == 0:
            return []
        x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        return [{"text": self.text, "score": 0.95, "box": [x0, y0, x1, y1], "poly": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}]


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def ocr_clips(tmp_path_factory) -> dict[str, Path]:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("FFmpeg not installed")
    d = tmp_path_factory.mktemp("ocr_clips")
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    box = "drawbox=x=20:y=40:w=100:h=30:color=white:t=fill"
    c = {name: d / f"{name}.mp4" for name in ("caption", "vfr", "plain", "brief", "repeat")}
    grey = ["-f", "lavfi", "-i", "color=gray:s=180x320:r=30"]
    # Caption visible from 1.0 s to 2.4 s on a grey 4 s clip.
    ffmpeg(grey[0], grey[1], grey[2], f"{grey[3]}:d=4", "-vf", f"{box}:enable='between(t,1.0,2.4)'", "-an", *enc, str(c["caption"]))
    # Caption flashed from 1.3 s to 1.9 s: between two 1 fps samples, inside two 4 fps samples.
    ffmpeg(grey[0], grey[1], grey[2], f"{grey[3]}:d=4", "-vf", f"{box}:enable='between(t,1.3,1.9)'", "-an", *enc, str(c["brief"]))
    # The same caption shown twice: 0 to 1.9 s and 4.0 to 5.9 s of a 7 s clip.
    ffmpeg(grey[0], grey[1], grey[2], f"{grey[3]}:d=7", "-vf", f"{box}:enable='between(t,0,1.9)+between(t,4.0,5.9)'", "-an",
           *enc, str(c["repeat"]))
    # Variable frame rate (30 fps for 1.5 s, then 10 fps); caption from 2.0 s.
    ffmpeg("-f", "lavfi", "-i", "color=gray:s=180x320:r=30:d=1.5", "-f", "lavfi", "-i", "color=gray:s=180x320:r=10:d=2.5",
           "-filter_complex", f"[0:v]settb=1/90000[a];[1:v]settb=1/90000[b];[a][b]concat=n=2:v=1:a=0,{box}:enable='gte(t,2.0)'[v]",
           "-map", "[v]", *enc, "-fps_mode", "passthrough", str(c["vfr"]))
    ffmpeg("-f", "lavfi", "-i", "color=gray:s=180x320:r=30:d=2", "-an", *enc, str(c["plain"]))
    return c


def clip_ctx(path: Path, **kw) -> MediaContext:
    p = probe_video(path)
    base = dict(video_id=path.stem, source_path=str(path), source_sha256=path.stem, prep_key=None,
                duration=p.duration_seconds, visual_stream="available", audio_stream="absent",
                video_span=[0.0, p.duration_seconds],
                width=p.width, height=p.height, source_fps=p.fps, source_vfr=p.variable_frame_rate)
    return MediaContext(**(base | kw))


def segs(duration, step=0.5):
    n = int(np.ceil(duration / step - 1e-9))
    return [seg(round(i * step, 6), round(min((i + 1) * step, duration), 6)) for i in range(n)]


def test_caption_timing_box_and_cleanup(ocr_clips, tmp_path) -> None:
    reader = BrightBoxReader()
    ctx = clip_ctx(ocr_clips["caption"])
    art = extract_ocr(ctx, CFG, LazyBackend(lambda: reader), tmp_path / "work")
    assert art["status"] == "available" and art["measured"] and art["text_detected"] is True
    assert art["sampling"]["frames"] == 8 and art["sampling"]["max_deviation_from_grid_s"] < 0.01
    assert [f["t_s"] for f in art["frames"]] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], abs=0.01)
    (ev,) = art["events"]
    assert ev["text"] == "HELLO" and ev["frames"] == 3
    assert ev["first_seen_s"] == pytest.approx(1.0, abs=0.01) and ev["visible_until_s"] == pytest.approx(2.5, abs=0.01)
    assert ev["box_norm"] == pytest.approx([20 / 180, 40 / 320, 120 / 180, 70 / 320], abs=0.02)
    assert art["sampling"]["frame_size"] == [180, 320]  # never upscaled
    assert list((tmp_path / "work").iterdir()) == []  # temporary frames removed
    rows = [text_window(art, s, ctx.video_span, 0.5)[0] for s in segs(ctx.duration)]
    assert [r["text_present"] for r in rows] == [False, False, True, True, True, False, False, False]


def test_samples_show_the_frame_on_screen_at_their_timestamp(ocr_clips, tmp_path) -> None:
    art = extract_ocr(clip_ctx(ocr_clips["caption"]), replace(CFG, sample_fps=1.0), LazyBackend(BrightBoxReader), tmp_path)
    assert [f["t_s"] for f in art["frames"]] == pytest.approx([0.0, 1.0, 2.0, 3.0], abs=0.01)
    assert [bool(f["detections"]) for f in art["frames"]] == [False, True, True, False]  # caption is on 1.0-2.4 s


def load_compare_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_ocr_runs.py"
    spec = importlib.util.spec_from_file_location("compare_ocr_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_brief_text_between_samples_is_uncertain_and_counted_as_a_miss(ocr_clips, tmp_path) -> None:
    ctx = clip_ctx(ocr_clips["brief"])
    one = extract_ocr(ctx, replace(CFG, sample_fps=1.0), LazyBackend(BrightBoxReader), tmp_path)
    assert one["events"] == [] and one["sampling"]["frames"] == 4  # 1 fps never sees the 0.6 s flash
    rows = {s["start_s"]: text_window(one, s, ctx.video_span, 0.5) for s in segs(ctx.duration)}
    flash = rows[1.5]  # the flash is on screen for this whole segment
    assert flash[0]["text_status"] == "uncertain" and flash[0]["text_samples"] == 0 and "inferred" in flash[1][0]
    assert rows[1.0][0]["text_status"] == "available"  # observed at 1.0 s, before the flash
    o = opening_text(one, 4.0)
    assert o["text_detected"] is False and o["sample_interval_s"] == 1.0  # a claim about the 1 fps samples only

    dense = extract_ocr(ctx, replace(CFG, sample_fps=4.0), LazyBackend(BrightBoxReader), tmp_path)
    (ev,) = dense["events"]
    assert ev["first_seen_s"] == pytest.approx(1.5, abs=0.01) and ev["frames"] == 2
    cmp = load_compare_script().compare(dense, one)
    assert cmp["ref_persistent"] == 1 and cmp["missed_persistent"] == 1
    assert cmp["missed"][0]["first_seen_s"] == pytest.approx(1.5, abs=0.01)
    assert load_compare_script().compare(dense, dense)["missed_persistent"] == 0


def test_compare_counts_sampling_misses_not_line_splits_or_noise() -> None:
    headline = ("MUMBAI LOCAL NIGHTMARE", BOX_A)
    split = [("MUMBAI LOCAL", BOX_A), ("NIGHTMARE", [0.6, 0.1, 0.9, 0.2])]
    extra = {1.25: [("Flash", BOX_B)], 1.5: [("Flash", BOX_B)], 1.75: [("Flash", BOX_B)], 2.5: [("x7", BOX_B)]}
    ref = art_from([frame(t, (split if t in (0.5, 1.0) else [headline]) + extra.get(t, [])) for t in (i * 0.25 for i in range(14))],
                   3.5, interval=0.25)
    test = art_from([frame(0.0, [headline]), frame(1.0, [headline]), frame(2.0, [headline]), frame(3.0, [headline])], 4.0, interval=1.0)
    r = load_compare_script().compare(ref, test)
    assert r["range_s"] == [0.0, 3.5] and r["shared_samples"] == 4 and r["identical_reads"] == 3  # 1.0 s was split in ref
    assert [m["text"] for m in r["missed"]] == ["Flash"]  # 1.25-2.0 s, between 1 fps samples
    assert r["missed_single_sample"] == 1  # "x7" at 2.5 s


def test_compare_line_breaks_word_boundaries_and_half_open_windows() -> None:
    compare = load_compare_script().compare
    quarter = [i * 0.25 for i in range(8)]
    # Ref splits the caption over two boxes; test reads it as one box with a line break.
    ref = art_from([frame(t, [("BIG SALE", BOX_A), ("TODAY!", BOX_B)]) for t in quarter], 2.0, interval=0.25)
    test = art_from([frame(0.0, [("Big sale\ntoday", BOX_A)]), frame(1.0, [("Big sale\ntoday", BOX_A)])], 2.0, interval=1.0)
    r = compare(ref, test)
    assert (r["ref_persistent"], r["missed_persistent"], r["merged_persistent"]) == (2, 0, 0)
    # ...and the other way round: one ref box, the test run reads two lines.
    assert compare(test, ref)["missed_persistent"] == 0
    # Containment needs whole words: "SALE" is not found inside "WHOLESALE".
    ref = art_from([frame(t, [("SALE", BOX_A)]) for t in quarter], 2.0, interval=0.25)
    test = art_from([frame(0.0, [("WHOLESALE", BOX_A)]), frame(1.0, [("WHOLESALE", BOX_A)])], 2.0, interval=1.0)
    assert compare(ref, test)["missed_persistent"] == 1

    # Ref caption on [1.0, 2.0). A test sample at 0.0 holds [0, 1) and one at 2.0 holds [2, 3):
    # both only touch the event, so neither matches (half-open windows).
    ref = art_from([frame(t, [("Follow us", BOX_A)] if 1.0 <= t < 2.0 else []) for t in (i * 0.25 for i in range(12))],
                   3.0, interval=0.25)
    touching = art_from([frame(0.0, [("Follow us", BOX_A)]), frame(1.0), frame(2.0, [("Follow us", BOX_A)])], 3.0, interval=1.0)
    assert compare(ref, touching)["missed_persistent"] == 1
    inside = art_from([frame(0.0), frame(1.0, [("Follow us", BOX_A)]), frame(2.0)], 3.0, interval=1.0)
    assert compare(ref, inside)["missed_persistent"] == 0
    # Float noise: 0.2 + 0.1 > 0.3 in binary floating point, but [0.2, 0.3) still only touches [0.3, ...).
    ref = art_from([frame(0.0), frame(0.15), frame(0.3, [("Hook", BOX_A)]), frame(0.45, [("Hook", BOX_A)])], 0.6, interval=0.15)
    test = art_from([frame(0.0), frame(0.1), frame(0.2, [("Hook", BOX_A)]), frame(0.3), frame(0.4), frame(0.5)], 0.6, interval=0.1)
    assert 0.2 + 0.1 > 0.3 and compare(ref, test)["missed_persistent"] == 1


def test_compare_one_sparse_event_cannot_satisfy_two_appearances() -> None:
    compare = load_compare_script().compare
    # Ref sees "Follow" on 0-0.5 s and again on 1.0-1.5 s (two events, the second a repeat).
    shown = {0.0, 0.25, 1.0, 1.25}
    ref = art_from([frame(t, [("Follow", BOX_A)] if t in shown else []) for t in (i * 0.25 for i in range(8))], 2.0, interval=0.25)
    assert [e["repeat_of"] for e in ref["events"]] == [None, 0]
    # The 1 fps run reads it at 0 s and 1 s, which links into a single event.
    test = art_from([frame(0.0, [("Follow", BOX_A)]), frame(1.0, [("Follow", BOX_A)])], 2.0, interval=1.0)
    assert len(test["events"]) == 1
    r = compare(ref, test)
    assert (r["missed_persistent"], r["merged_persistent"]) == (0, 1)
    assert r["merged"][0]["first_seen_s"] == pytest.approx(1.0)
    assert compare(ref, ref)["merged_persistent"] == 0  # each dense event is its own counterpart


def test_compare_groups_line_splits_and_misreads_into_one_episode() -> None:
    compare = load_compare_script().compare
    head, left, right, sign = BOX_A, [0.1, 0.1, 0.35, 0.2], [0.35, 0.1, 0.6, 0.2], BOX_B

    def reads(t):
        caption = [("MUMBAI LOCAL", left), ("NIGHTMARE", right)] if t in (0.5, 0.75, 2.5, 2.75) else [("MUMBAI LOCAL NIGHTMARE", head)]
        return caption + [("DWLZ" if t in (1.0, 1.25) else "DEALZ", sign)]  # a sign misread for two samples

    # Both texts stay on screen for the whole 4 s; at 4 fps the reader keeps splitting/misreading them.
    ref = art_from([frame(t, reads(t)) for t in (i * 0.25 for i in range(16))], 4.0, interval=0.25)
    assert len(ref["events"]) >= 6  # fragmented events...
    test = art_from([frame(t, [("MUMBAI LOCAL NIGHTMARE", head), ("DEALZ", sign)]) for t in (0.0, 1.0, 2.0, 3.0)], 4.0, interval=1.0)
    r = compare(ref, test)
    assert r["ref_episodes"] == 2 and r["ref_persistent"] == 2  # ...but two visibility episodes
    assert r["ref_fragment_events"] == len(ref["events"]) - 2
    assert (r["missed_persistent"], r["merged_persistent"]) == (0, 0)  # not reappearances, not misses


def test_compare_episode_linking_needs_same_place_and_alike_text() -> None:
    cmp = load_compare_script()
    # Thresholds measured on the pilot reels: misreads of one text 0.60-0.80, different texts <= 0.57.
    assert cmp.misread("dealz", "dwlz", BOX_A, BOX_A) and cmp.misread("where y", "where you going", BOX_A, BOX_A)
    assert not cmp.misread("dealz", "dwlz", BOX_A, BOX_B)  # same reading elsewhere on screen is not linked
    assert not cmp.misread("kind of", "artist", BOX_A, BOX_A)
    # 0.75 / 0.73 misreads are not the same caption on their own...
    assert not cmp.similar("ital", "iyal") and not cmp.similar("dlalza", "dealz")
    # ...but the episode they belong to also holds exact reads, which the sparse run can match.
    ref = art_from([frame(t, [("DEALZ" if t < 0.5 else "DLALZA", BOX_A)]) for t in (0.0, 0.25, 0.5, 0.75)], 1.0, interval=0.25)
    test = art_from([frame(0.0, [("DEALZ", BOX_A)])], 1.0, interval=1.0)
    r = cmp.compare(ref, test)
    assert (r["ref_events"], r["ref_episodes"], r["missed_persistent"]) == (2, 1, 0)

    # Different captions in the same box, back to back, stay separate: the second is missed.
    ref = art_from([frame(t, [("KIND OF" if t < 1.0 else "ARTIST", BOX_A)]) for t in (i * 0.25 for i in range(8))], 2.0, interval=0.25)
    test = art_from([frame(0.0, [("KIND OF", BOX_A)]), frame(1.0)], 2.0, interval=1.0)
    r = cmp.compare(ref, test)
    assert r["ref_episodes"] == 2 and [m["text"] for m in r["missed"]] == ["ARTIST"]


def test_repeated_caption_at_one_fps_links_back(ocr_clips, tmp_path) -> None:
    ctx = clip_ctx(ocr_clips["repeat"])
    art = extract_ocr(ctx, replace(CFG, sample_fps=1.0), LazyBackend(BrightBoxReader), tmp_path)
    first, again = art["events"]
    assert (first["first_seen_s"], first["last_seen_s"]) == pytest.approx((0.0, 1.0), abs=0.01)
    assert (again["first_seen_s"], again["last_seen_s"]) == pytest.approx((4.0, 5.0), abs=0.01)
    assert again["repeat_of"] == first["event_id"]
    rows = {s["start_s"]: text_window(art, s, ctx.video_span, 0.5)[0] for s in segs(ctx.duration)}
    assert rows[0.0]["text_new_count"] == 1 and rows[1.0]["text_new_count"] == 0
    assert rows[4.0]["text_present"] is True and rows[4.0]["text_new_count"] == 0  # a repeat, not new text
    assert rows[3.0]["text_present"] is False and rows[3.0]["text_status"] == "available"


def test_variable_frame_rate_source_keeps_timeline(ocr_clips, tmp_path) -> None:
    ctx = clip_ctx(ocr_clips["vfr"])
    assert ctx.source_vfr is True
    art = extract_ocr(ctx, CFG, LazyBackend(BrightBoxReader), tmp_path)
    assert art["sampling"]["source_variable_frame_rate"] is True
    assert art["sampling"]["max_deviation_from_grid_s"] < 0.01
    (ev,) = art["events"]
    assert ev["first_seen_s"] == pytest.approx(2.0, abs=0.01)


def test_empty_detection_vs_ocr_failure(ocr_clips, tmp_path) -> None:
    ctx = clip_ctx(ocr_clips["plain"])
    empty = extract_ocr(ctx, CFG, LazyBackend(BrightBoxReader), tmp_path)
    assert empty["status"] == "available" and empty["text_detected"] is False and empty["events"] == []
    assert all(f["status"] == "available" and f["detections"] == [] for f in empty["frames"])

    crash = BrightBoxReader(fail_at=set(range(100)))
    failed = extract_ocr(ctx, CFG, LazyBackend(lambda: crash), tmp_path)
    assert failed["status"] == "failed" and failed["measured"] is False and failed["text_detected"] is None
    assert "every sampled frame" in failed["reason"]
    assert all(f["status"] == "failed" and "simulated" in f["error"] for f in failed["frames"])

    partly = extract_ocr(ctx, CFG, LazyBackend(lambda: BrightBoxReader(fail_at={1})), tmp_path)
    assert partly["status"] == "available" and partly["sampling"]["failed_frames"] == 1
    assert any("OCR failed on 1 of 4" in w for w in partly["warnings"])
    rows = [text_window(partly, s, ctx.video_span, 0.5)[0]["text_status"] for s in segs(ctx.duration)]
    assert rows == ["available", "failed", "available", "available"]


class SymbolReader(BrightBoxReader):
    def read(self, img: np.ndarray) -> list[dict]:
        return [{"text": t, "score": 0.9, "box": [10, 10, 40, 20], "poly": None} for t in ("-", "*******", "=**")]


def test_symbol_only_detections_are_not_text(ocr_clips, tmp_path) -> None:
    art = extract_ocr(clip_ctx(ocr_clips["plain"]), CFG, LazyBackend(SymbolReader), tmp_path)
    assert art["status"] == "available" and art["text_detected"] is False and art["events"] == []


def test_model_unavailable_and_absent_video(ocr_clips, tmp_path) -> None:
    def factory():
        raise BackendUnavailable("paddleocr/paddlepaddle are not installed")
    art = extract_ocr(clip_ctx(ocr_clips["plain"]), CFG, LazyBackend(factory), tmp_path)
    assert art["status"] == "failed" and art["text_detected"] is None and "model unavailable" in art["reason"]

    def must_not_load():
        raise AssertionError("model must not be loaded")
    backend = LazyBackend(must_not_load)
    absent = extract_ocr(clip_ctx(ocr_clips["plain"], visual_stream="absent"), CFG, backend, tmp_path)
    assert absent["status"] == "absent" and absent["text_detected"] is None and backend.error is None
    unknown = extract_ocr(clip_ctx(ocr_clips["plain"], visual_stream="uncertain"), CFG, backend, tmp_path)
    assert unknown["status"] == "uncertain" and "unknown" in unknown["reason"]


def test_max_seconds_limits_range(ocr_clips, tmp_path) -> None:
    ctx = clip_ctx(ocr_clips["caption"])
    art = extract_ocr(ctx, replace(CFG, max_seconds=1.5), LazyBackend(BrightBoxReader), tmp_path)
    assert art["sampling"]["range_s"] == [0.0, 1.5] and art["sampling"]["frames"] == 3
    rows = [text_window(art, s, ctx.video_span, 0.5) for s in segs(ctx.duration)]
    assert [r[0]["text_status"] for r in rows[:3]] == ["available"] * 3
    assert all(r[0]["text_status"] == "uncertain" and r[0]["text_present"] is None for r in rows[3:])
    assert "beyond OCR range" in rows[3][1][0]
