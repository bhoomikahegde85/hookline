"""Merged evidence timeline (segments_merged.csv) built from hand-made artifacts: no FFmpeg, no models."""

from __future__ import annotations

from hookline.config import OcrConfig
from hookline.context import MediaContext
from hookline.features import SEGMENT_COLUMNS, _csv_value
from hookline.multimodal import MERGED_COLUMNS, merged_rows, stage_status
from hookline.ocr import build_events, observation_rows
from hookline.speech import word_rows

BOX = [0.1, 0.1, 0.6, 0.2]


def segment(i, start, end, scene, cut=False, audio="available"):
    row = {c: None for c in SEGMENT_COLUMNS}
    return row | {"video_id": "v", "segment_index": i, "start_s": start, "end_s": end, "duration_s": round(end - start, 6),
                  "is_partial": end - start < 0.5, "scene_index": scene, "scene_cut_at_start": cut,
                  "visual_status": "available", "brightness_mean": 100.0 + i, "motion_mean": 0.5 * i,
                  "audio_status": audio, "rms_dbfs": None if audio != "available" else -20.0 - i, "notes": ""}


def context(segments) -> MediaContext:
    return MediaContext(video_id="v", source_path="v.mp4", source_sha256="x", prep_key="k", duration=2.0,
                        visual_stream="available", audio_stream="available", video_span=[0.0, 2.0], segments=segments)


# 0.5 s grid with a scene cut at 1.3 s; audio decoding failed in the 1.3-1.8 s segment.
SEGMENTS = [segment(0, 0.0, 0.5, 0), segment(1, 0.5, 1.0, 0), segment(2, 1.0, 1.3, 0),
            segment(3, 1.3, 1.8, 1, cut=True, audio="failed"), segment(4, 1.8, 2.0, 1)]


def speech_art():
    words = [(0.1, 0.4, "hello"), (0.45, 0.7, "there"), (0.8, 0.95, "friend"), (1.4, 1.6, "again")]
    return {"video_id": "v", "status": "available", "measured": True, "word_timestamps": True,
            "segments": [{"id": 0, "start_s": 0.1, "end_s": 1.6, "flags": [],
                          "words": [{"start_s": a, "end_s": b, "word": w, "probability": 0.9} for a, b, w in words]}]}


def ocr_art():
    frames = [{"t_s": 0.0, "status": "available", "detections": [{"text": "HOOK", "score": 0.9, "box_norm": BOX}]},
              {"t_s": 1.0, "status": "available", "detections": []}]
    return {"video_id": "v", "status": "available", "measured": True, "frames": frames,
            "events": build_events(frames, OcrConfig(), 1.0, 2.0),
            "sampling": {"interval_s": 1.0, "fps": 1.0, "range_s": [0.0, 2.0]}}


def test_merged_rows_keep_features_timestamps_and_link_detail_tables() -> None:
    speech, ocr = speech_art(), ocr_art()
    words = word_rows(speech, SEGMENTS)
    rows = merged_rows(context(SEGMENTS), speech, ocr, 0.5, words)
    assert [list(r) for r in rows] == [MERGED_COLUMNS] * 5
    for r, s in zip(rows, SEGMENTS):  # timeline, scene boundaries and visual/audio values untouched
        assert {c: r[c] for c in SEGMENT_COLUMNS if c != "notes"} == {c: s[c] for c in SEGMENT_COLUMNS if c != "notes"}
    assert rows[3]["scene_cut_at_start"] is True and rows[3]["start_s"] == 1.3

    # Speech: each word in the segment holding its midpoint; first/last link into words.csv.
    assert [(r["speech_word_index_first"], r["speech_word_index_last"], r["speech_text"]) for r in rows[:3]] == [
        (0, 0, "hello"), (1, 2, "there friend"), (None, None, None)]
    by_index = {w["word_index"]: w for w in words}
    assert (by_index[1]["start_s"], by_index[1]["end_s"], by_index[1]["segment_index"]) == (0.45, 0.7, 1)  # exact times kept
    assert rows[2]["speech_status"] == "available" and rows[2]["speech_present"] is False  # measured: no words here

    # Text: observed where a sample lies in the segment, inferred between samples; event ids link to the observations.
    assert [(r["text_status"], r["text_evidence"], r["text_present"], r["text_event_ids"]) for r in rows] == [
        ("available", "observed", True, [0]),
        ("uncertain", "inferred", True, [0]),     # held from the 0.0 s sample
        ("available", "observed", False, None),   # the 1.0 s sample read nothing
        ("uncertain", "inferred", False, None),
        ("uncertain", "inferred", False, None),
    ]
    obs = observation_rows(ocr, SEGMENTS)
    assert [(o["t_s"], o["segment_index"], o.get("event_id"), o.get("x0")) for o in obs] == [(0.0, 0, 0, 0.1), (1.0, 2, None, None)]
    assert "text=inferred between samples" in rows[1]["notes"]


def test_failed_audio_window_keeps_word_detail_but_no_segment_values() -> None:
    speech = speech_art()
    words = word_rows(speech, SEGMENTS)
    row = merged_rows(context(SEGMENTS), speech, ocr_art(), 0.5, words)[3]
    assert row["speech_status"] == "failed" and row["audio_status"] == "failed"
    assert all(row[c] is None for c in ("speech_present", "speech_coverage", "speech_word_count", "speech_text",
                                        "speech_word_index_first", "speech_word_index_last"))
    assert [w["word"] for w in words if w["segment_index"] == 3] == ["again"]  # still in words.csv with exact times


def test_absent_and_failed_modalities_stay_blank_not_zero() -> None:
    speech = {"video_id": "v", "status": "absent", "measured": False, "reason": "no audio stream"}
    ocr = {"video_id": "v", "status": "failed", "measured": False, "reason": "OCR model unavailable", "frames": [], "events": []}
    segs = [segment(i, s["start_s"], s["end_s"], s["scene_index"], audio="absent") for i, s in enumerate(SEGMENTS)]
    rows = merged_rows(context(segs), speech, ocr, 0.5)
    for r in rows:
        assert r["speech_status"] == "absent" and r["text_status"] == "failed"
        assert all(r[c] is None for c in MERGED_COLUMNS if c.startswith(("speech_", "text_", "on_screen")) and not c.endswith("_status"))
        assert r["brightness_mean"] is not None  # other modalities are still reported
    csv_row = {k: _csv_value(rows[0][k]) for k in MERGED_COLUMNS}
    assert csv_row["speech_word_count"] == "" and csv_row["text_present"] == "" and csv_row["text_event_ids"] == ""
    assert word_rows(speech, segs) == [] and observation_rows(ocr, segs) == []
    assert stage_status([speech, ocr], rows) == "partial"  # failed OCR is never reported as "no text"


def test_skipped_stages_are_uncertain() -> None:
    rows = merged_rows(context(SEGMENTS), None, None, 0.5)
    assert {(r["speech_status"], r["text_status"]) for r in rows} == {("uncertain", "uncertain")}
    assert all(r["text_evidence"] is None and r["speech_word_index_first"] is None for r in rows)
