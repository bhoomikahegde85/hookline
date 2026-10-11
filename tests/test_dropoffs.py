"""Drop-off detection on a predicted retention curve. Curves here are synthetic test fixtures, not results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hookline.dropoffs import DROPOFFS_FILE, THRESHOLDS, analyze, classify_seconds, merge_runs, write_dropoffs
from hookline.video_report import write_video_report

# 0 steep (1.0 -> 0.5), 1 steep, 2-5 steady decline, 6-9 flat, rest gradual
CURVE = [1.0, 0.5, 0.3, 0.285, 0.27, 0.256, 0.243, 0.243, 0.2425, 0.2422, 0.2420]


def write(d: Path, name: str, data) -> None:
    (d / name).write_text(json.dumps(data))


def video_dir(tmp_path: Path, curve=CURVE, **extra) -> Path:
    d = tmp_path / "features" / "v"
    d.mkdir(parents=True)
    art = {"status": "success", "video_id": "v", "cache_key": "k", "horizon_s": len(curve) - 1, "partial_coverage": False,
           "model": {"checkpoint_repo": "m"}, "timestamps_s": list(range(len(curve))), "retention": curve} | extra
    write(d, "ttcc_retention.json", art)
    return d


def test_classification_uses_share_of_remaining_viewers() -> None:
    kinds = [s["kind"] for s in classify_seconds(CURVE)]
    assert kinds[:2] == ["steep", "steep"]
    assert kinds[2:5] == ["decline", "decline", "decline"]  # 0.285/0.3 = 5 % lost, 0.27, 0.256 ...
    assert kinds[6] == "hold"
    tiny = classify_seconds([1.0, 0.02, 0.0005])  # 97 % of the few remaining viewers, but under the 0.02 absolute minimum
    assert tiny[1]["relative_loss"] > 0.9 and tiny[1]["kind"] == "decline"


def test_runs_merge_into_intervals_with_shares_of_total_loss() -> None:
    secs = classify_seconds(CURVE)
    total = CURVE[0] - CURVE[-1]
    ivs = merge_runs(secs, total)
    steep = ivs[0]
    assert (steep["kind"], steep["start_s"], steep["end_s"]) == ("steep_drop", 0, 2)
    assert steep["share_of_total_predicted_loss"] == pytest.approx((1.0 - 0.3) / total)
    assert any(i["kind"] == "sustained_decline" and i["end_s"] - i["start_s"] >= THRESHOLDS["decline_min_seconds"] for i in ivs)
    short = merge_runs(classify_seconds([1.0, 0.5, 0.3, 0.285, 0.27, 0.27, 0.27]), 0.73)  # only two decline seconds: not sustained
    assert [i["kind"] for i in short] == ["steep_drop"]


def test_no_prediction_means_no_analysis(tmp_path) -> None:
    d = tmp_path / "v"
    d.mkdir()
    assert analyze(d) is None and write_dropoffs(d) is None
    write(d, "ttcc_retention.json", {"status": "failed"})
    assert analyze(d) is None
    write(d, "ttcc_retention.json", {"status": "success", "video_id": "v", "retention": [1.0, 0.9], "timestamps_s": [0, 1], "horizon_s": 1})
    assert analyze(d) is None  # too short to say anything


def test_missing_evidence_is_not_run_never_nothing(tmp_path) -> None:
    d = video_dir(tmp_path)
    out = analyze(d)
    assert set(out["evidence_availability"].values()) == {"not run"}
    first = out["intervals"][0]["observed"]
    assert first["words"] == "not run" and first["on_screen_text"] == "not run" and first["scene_cuts"] == "not run"
    assert out["intervals"][0]["candidate_causes"] == [] and out["confidence"] is None and "no calibrated" in out["confidence_note"]


def test_observed_evidence_and_candidates_come_from_overlapping_artifacts(tmp_path) -> None:
    d = video_dir(tmp_path)
    write(d, "speech.json", {"measured": True, "status": "available", "language": "hi"})
    (d / "words.csv").write_text("video_id,word_index,segment_index,start_s,end_s,word,probability,asr_segment_id,asr_flags\n"
                                 "v,0,0,0.2,0.5,hello,0.9,0,\nv,1,1,5.0,5.4,later,0.9,0,\n")
    write(d, "ocr.json", {"measured": True, "events": [{"event_id": 0, "text": "SALE", "first_seen_s": 1.0, "visible_until_s": 2.0}]})
    write(d, "report.json", {"scene_cut_times": [1.2, 7.5], "segments": [{"start_s": 0.0, "end_s": 1.0, "visual_status": "available", "motion_mean": 8.0}]})
    write(d, "recommendations.json", {"recommendations": [
        {"rule": "no_early_visual_change", "start_s": 0.0, "end_s": 1.5, "suggestion": "add a cut", "evidence": [{"type": "segment", "id": 0}]},
        {"rule": "static_stretch", "start_s": 8.0, "end_s": 10.0, "suggestion": "vary it", "evidence": []}]})
    out = analyze(d)
    steep = out["intervals"][0]
    assert steep["observed"]["words"]["text"] == "hello" and steep["observed"]["on_screen_text"][0]["text"] == "SALE"
    assert steep["observed"]["scene_cuts"]["times_s"] == [1.2] and steep["observed"]["motion_mean"] == 8.0
    assert [c["rule"] for c in steep["candidate_causes"]] == ["no_early_visual_change"]  # the later rule does not overlap
    assert steep["candidate_causes"][0]["status"] == "inferred, unvalidated"
    assert out["evidence_availability"]["speech"] == "available" and out["speech_language"] == "hi"
    second1 = out["seconds"][0]
    assert second1["t"] == 1 and second1["observed"]["words"]["count"] == 1  # the hello word falls in (0, 1]
    assert out["seconds"][5]["observed"]["words"]["text"] == "later"  # t=6 covers (5, 6]


def test_write_report_section_and_partial_flag(tmp_path) -> None:
    d = video_dir(tmp_path, partial_coverage=True, duration_s=70.0, inference={}, summary={}, note="n", convention={"t0": "x"},
                  generated_at="now", model={"checkpoint_repo": "m", "checkpoint_revision": "r" * 12, "head_type": "hazard"})
    path = write_dropoffs(d)
    assert path.name == DROPOFFS_FILE and json.loads(path.read_text())["source"]["partial_coverage"] is True
    write(d, "ttcc_status.json", {"status": "success", "prediction_is_current": True})
    text = write_video_report(d, "v").read_text()
    assert "### Predicted drop-offs" in text and "steep drop" in text and "unvalidated" in text
    assert "not run" in text  # evidence sources that were not run are named, not hidden


def test_segments_cover_the_whole_span_and_are_scored_without_merging_the_two_kinds_of_number(tmp_path) -> None:
    d = video_dir(tmp_path)
    write(d, "scores.json", {"segments": [
        {"start_s": 0.0, "end_s": 0.5, "attention_risk": 40.0, "moment_quality": 5.0},
        {"start_s": 0.5, "end_s": 1.0, "attention_risk": 20.0, "moment_quality": 0.0},
        {"start_s": 3.0, "end_s": 3.5, "attention_risk": 0.0, "moment_quality": 10.0}]})
    out = analyze(d)
    segs = out["segments"]
    assert segs[0]["start_s"] == 0 and segs[-1]["end_s"] == len(CURVE) - 1
    assert all(a["end_s"] == b["start_s"] for a, b in zip(segs, segs[1:]))  # contiguous, no gaps or overlaps
    assert sum(s["share_of_total_predicted_loss"] for s in segs) == pytest.approx(1.0)
    first = segs[0]
    assert first["kind"] == "steep_drop" and first["rank_by_predicted_loss"] == 1 and first["interval_id"] == "d0"
    assert first["heuristic"]["attention_risk"] == {"mean": 20.0, "max": 40.0} and first["heuristic"]["available"] is True
    assert "combined" not in json.dumps(segs) and "score" not in first  # no single merged score is invented
    assert sorted(s["rank_by_predicted_loss"] for s in segs) == list(range(1, len(segs) + 1))
    assert any(s["kind"] == "unflagged" for s in segs) or all(s["interval_id"] for s in segs)


def test_segments_without_heuristic_scores_say_so(tmp_path) -> None:
    out = analyze(video_dir(tmp_path))
    assert out["evidence_availability"]["heuristic_scores"] == "not run"
    for s in out["segments"]:
        assert s["heuristic"] == {"available": False, "attention_risk": {"mean": None, "max": None}, "moment_quality": {"mean": None, "max": None}}


def test_even_segments_are_chosen_from_the_duration_alone() -> None:
    from hookline.dropoffs import even_segments

    def curve(n):
        return [0.93 ** t for t in range(n + 1)]
    for n, count in ((12, 3), (30, 8), (60, 12), (120, 12), (4, 1)):
        segs = even_segments(curve(n), 1.0)
        assert len(segs) == count and segs[0]["start_s"] == 0 and segs[-1]["end_s"] == n
        assert all(a["end_s"] == b["start_s"] for a, b in zip(segs, segs[1:]))
        lengths = [s["end_s"] - s["start_s"] for s in segs]
        assert max(lengths) - min(lengths) <= 1                                   # evenly divided: whole seconds, lengths differ by at most one
    assert [(s["start_s"], s["end_s"]) for s in even_segments(curve(12), 1.0)] == [(0, 4), (4, 8), (8, 12)]
    assert even_segments([1.0], 0.0) == []


def test_embedded_page_data_survives_html_comment_and_script_markers() -> None:
    from hookline.review_page import _embed
    data = {"t": "x <!-- y </script> z", "n": 1}
    assert json.loads(_embed(data)) == data and "<" not in _embed(data)


def test_a_gemini_analysis_of_an_older_segmentation_is_stale(tmp_path) -> None:
    from hookline.gemini_edit import segments_key
    from hookline.segment_recommendations import build_segment_recommendations
    d = video_dir(tmp_path)
    assert write_dropoffs(d)
    dr = json.loads((d / DROPOFFS_FILE).read_text())
    gem = {"measured": True, "ttcc_cache_key": "k", "segments_key": segments_key([{"id": "s0", "start_s": 0, "end_s": 99}]),
           "analysis": {"segments": [{"segment_id": "s0", "summary": "old", "observations": [], "hypothesis": "", "hypothesis_confidence": None, "suggestions": []}], "overall": None}}
    write(d, "gemini_edit.json", gem)
    assert build_segment_recommendations(d)["gemini"]["stale"] is True
    write(d, "gemini_edit.json", gem | {"segments_key": segments_key(dr["segments"])})
    assert build_segment_recommendations(d)["gemini"]["stale"] is False
