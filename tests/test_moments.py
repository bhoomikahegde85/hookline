"""Strongest / weakest moments on a predicted retention curve. Curves and evidence here are synthetic test fixtures, not results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hookline.moments import MOMENTS_FILE, find_moments, write_moments

# hazard per second t = 1..24 (the share of the remaining viewers lost in that second)
HAZARDS = [0.5] * 3 + [0.20] * 3 + [0.05] * 3 + [0.01] * 3 + [0.08] * 3 + [0.02] * 3 + [0.03] * 3 + [0.15] * 3


def curve(hazards: list[float]) -> list[float]:
    r = [1.0]
    for h in hazards:
        r.append(r[-1] * math.exp(-h))
    return r


def write(d: Path, name: str, data) -> None:
    (d / name).write_text(json.dumps(data))


def video_dir(tmp_path: Path, retention=None, **extra) -> Path:
    retention = curve(HAZARDS) if retention is None else retention
    d = tmp_path / "features" / "v"
    d.mkdir(parents=True)
    art = {"status": "success", "video_id": "v", "horizon_s": len(retention) - 1, "timestamps_s": list(range(len(retention))), "retention": retention} | extra
    write(d, "ttcc_retention.json", art)
    return d


def add_evidence(d: Path) -> None:
    """Speech at 9.5-11.5 s and 21-23 s only; one cut at 10.5 s; quiet audio 12-15 s; text at 9-10 s."""
    write(d, "speech.json", {"measured": True, "status": "available", "language": "en"})
    (d / "words.csv").write_text("video_id,word_index,segment_index,start_s,end_s,word,probability,asr_segment_id,asr_flags\n"
                                 "v,0,0,9.5,9.9,keep,0.9,0,\nv,1,0,10.0,10.4,going,0.9,0,\nv,2,0,21.0,21.4,later,0.9,0,\n")
    write(d, "ocr.json", {"measured": True, "events": [{"event_id": 0, "text": "BIG NEWS", "first_seen_s": 9.0, "visible_until_s": 10.0}]})
    write(d, "librosa.json", {"measured": True, "low_energy_intervals": [{"interval_id": 0, "start_s": 12.0, "end_s": 15.0}]})
    write(d, "report.json", {"scene_cut_times": [10.5], "segments": [{"start_s": 0.0, "end_s": 24.0, "visual_status": "available", "motion_mean": 4.0}]})


def windows(moments: list[dict]) -> list[tuple[int, int]]:
    return [(m["start_s"], m["end_s"]) for m in moments]


def test_exact_windows_ordering_and_hazards(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path))
    assert out["schema"] == "hookline.moments" and out["video_id"] == "v" and out["window_s"] == 3 and out["skip_opening_s"] == 3
    assert windows(out["weakest"]) == [(3, 6), (21, 24), (12, 15)]
    assert windows(out["strongest"]) == [(9, 12), (15, 18), (18, 21)]
    weak, strong = out["weakest"], out["strongest"]
    assert [m["rank"] for m in weak] == [1, 2, 3] and [m["rank"] for m in strong] == [1, 2, 3]
    assert [m["mean_hazard"] for m in weak] == pytest.approx([0.20, 0.15, 0.08], abs=1e-6)
    assert [m["mean_hazard"] for m in strong] == pytest.approx([0.01, 0.02, 0.03], abs=1e-6)
    assert weak[0]["mean_hazard"] >= weak[1]["mean_hazard"] >= weak[2]["mean_hazard"]
    assert strong[0]["mean_hazard"] <= strong[1]["mean_hazard"] <= strong[2]["mean_hazard"]
    r = curve(HAZARDS)
    assert weak[0]["retention_start"] == pytest.approx(r[3]) and weak[0]["retention_end"] == pytest.approx(r[6])
    assert weak[0]["predicted_fall"] == pytest.approx(r[3] - r[6])
    # median hazard over seconds 4..24 is 0.05
    assert weak[0]["relative_to_video_median_hazard"] == pytest.approx(0.20 / 0.05, abs=1e-3)
    assert strong[0]["relative_to_video_median_hazard"] == pytest.approx(0.01 / 0.05, abs=1e-3)
    assert all(m["suggested_action"] is None for m in strong) and all(m["suggested_action"] is not None for m in weak)
    assert "model prediction, not audience data" in out["note"]


def test_windows_do_not_overlap_and_no_window_in_both_lists(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path))
    spans = sorted(windows(out["weakest"]) + windows(out["strongest"]))
    assert len(spans) == 6
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, f"{(a0, a1)} overlaps {(b0, b1)}"
    assert not set(windows(out["weakest"])) & set(windows(out["strongest"]))


def test_opening_is_excluded(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path))  # the first 3 s have by far the highest hazard (0.5)
    assert all(m["start_s"] >= 3 for m in out["weakest"] + out["strongest"])
    assert (0, 3) not in windows(out["weakest"])
    later = find_moments(video_dir(tmp_path / "b"), skip_opening_s=0)
    assert windows(later["weakest"])[0] == (0, 3)
    assert later["skip_opening_s"] == 0


def test_k_and_window_size(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path), k=1, window_s=2)
    assert windows(out["weakest"]) == [(3, 5)]  # seconds 4 and 5 have hazard 0.20
    assert windows(out["strongest"]) == [(9, 11)]  # (9, 11) and (10, 12) tie at 0.01: the earlier wins
    assert out["window_s"] == 2
    assert len(find_moments(video_dir(tmp_path / "b"), k=2)["weakest"]) == 2


def test_too_few_windows_gives_fewer_and_says_so(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path, curve([0.5] * 3 + [0.2] * 3 + [0.01] * 3)))  # room for only two windows after the opening
    assert windows(out["weakest"]) == [(3, 6)] and windows(out["strongest"]) == [(6, 9)]  # alternating picks: each list gets one
    assert "fewer" in out["note"] and "model prediction, not audience data" in out["note"]


def test_short_video_returns_empty_with_reason(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path, curve([0.5, 0.3, 0.2, 0.1, 0.1])))  # T = 5 < 3 + 3
    assert out["strongest"] == [] and out["weakest"] == [] and "at least 6 s" in out["reason"]
    assert out["schema"] == "hookline.moments" and out["curve_is_stitched"] is False


def test_flat_curve_has_no_standout_moments(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path, [1.0] * 20))
    assert out["strongest"] == [] and out["weakest"] == [] and "flat" in out["reason"]
    steady = find_moments(video_dir(tmp_path / "b", curve([0.5] * 3 + [0.1] * 17)))  # constant hazard after the opening: all windows tie
    assert steady["strongest"] == [] and steady["weakest"] == []


def test_ties_break_to_the_earlier_window(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path, curve([0.5] * 3 + [0.1] * 6 + [0.3] * 3 + [0.1] * 6)), k=2)
    assert windows(out["weakest"]) == [(9, 12), (6, 9)]  # (6, 9) ties with (3, 6) and (12, 15) at 0.1, but (3, 6) went to strongest
    assert windows(out["strongest"]) == [(3, 6), (12, 15)]  # the earliest of the tied windows first


def test_stitched_flag_is_carried(tmp_path) -> None:
    assert find_moments(video_dir(tmp_path, stitched=True))["curve_is_stitched"] is True
    assert find_moments(video_dir(tmp_path / "b"))["curve_is_stitched"] is False


def test_zero_and_rising_retention_do_not_break_hazard(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path, [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0]), k=1)
    assert windows(out["weakest"]) == [(3, 6)]                          # the fall to zero is the worst; windows that start at zero retention are never ranked
    assert all(m["retention_start"] >= 0.02 for m in out["weakest"] + out["strongest"])
    assert all(math.isfinite(m["mean_hazard"]) for m in out["weakest"] + out["strongest"])
    tiny = find_moments(video_dir(tmp_path / "z", [1.0, 0.5, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), k=1)
    assert tiny["strongest"] == [] and tiny["weakest"] == [] and "below 2%" in tiny["reason"]   # nothing left to rank: nothing is claimed
    rising = find_moments(video_dir(tmp_path / "b", [1.0, 0.9, 0.8, 0.7, 0.75, 0.8, 0.7, 0.6, 0.5]), k=1)  # a rise counts as no loss, not negative hazard
    assert windows(rising["weakest"]) == [(5, 8)] and rising["weakest"][0]["mean_hazard"] == pytest.approx(math.log(0.8 / 0.7) / 3 + math.log(0.7 / 0.6) / 3 + math.log(0.6 / 0.5) / 3)


def test_heuristic_scores_are_overlap_weighted_and_optional(tmp_path) -> None:
    d = video_dir(tmp_path)
    assert find_moments(d)["weakest"][0]["heuristic"] is None
    seg = lambda a, b, risk, q: {"start_s": a, "end_s": b, "heuristic": {"available": True, "attention_risk": {"mean": risk, "max": 99.0}, "moment_quality": {"mean": q, "max": 99.0}}}  # noqa: E731
    write(d, "ttcc_dropoffs.json", {"segments": [seg(0, 9, 80.0, 20.0), seg(9, 13, 40.0, 60.0), seg(13, 24, 70.0, 30.0),
                                                 {"start_s": 0, "end_s": 24, "heuristic": {"available": False}}]})
    out = find_moments(d)
    by_start = {m["start_s"]: m for m in out["weakest"] + out["strongest"]}
    assert by_start[3]["heuristic"] == {"attention_risk": 80.0, "moment_quality": 20.0}
    assert by_start[12]["heuristic"] == {"attention_risk": 60.0, "moment_quality": 40.0}  # (1 s x 40 + 2 s x 70) / 3 and (1 x 60 + 2 x 30) / 3
    assert set(by_start[9]["heuristic"]) == {"attention_risk", "moment_quality"}


def test_missing_or_malformed_files_return_none(tmp_path) -> None:
    d = tmp_path / "v"
    d.mkdir()
    assert find_moments(d) is None
    write(d, "ttcc_retention.json", {"status": "failed"})
    assert find_moments(d) is None
    (d / "ttcc_retention.json").write_text("{not json")
    assert find_moments(d) is None
    write(d, "ttcc_retention.json", {"status": "success", "video_id": "v", "retention": "oops"})
    assert find_moments(d) is None
    write(d, "ttcc_retention.json", {"status": "success", "video_id": "v", "retention": [1.0, None, 0.5]})
    assert find_moments(d) is None
    write(d, "ttcc_retention.json", {"status": "success", "video_id": "v", "retention": [1.0, 0.9, 0.8], "timestamps_s": [0, 2, 4]})
    assert find_moments(d) is None


def test_malformed_evidence_does_not_raise(tmp_path) -> None:
    d = video_dir(tmp_path)
    (d / "speech.json").write_text("{broken")
    (d / "ttcc_dropoffs.json").write_text("[1, 2")
    out = find_moments(d)
    assert windows(out["weakest"])[0] == (3, 6) and out["weakest"][0]["heuristic"] is None


def test_without_evidence_observed_is_not_run_and_claims_nothing(tmp_path) -> None:
    out = find_moments(video_dir(tmp_path))
    first = out["weakest"][0]
    assert first["observed"]["words"] == "not run" and first["observed"]["scene_cuts"] == "not run"
    joined = " ".join(first["why"] + out["strongest"][0]["why"])
    assert "no speech" not in joined and "no scene cut" not in joined and "dead air" not in joined
    assert "no measured feature" in first["why"][0]
    assert first["suggested_action"]["action"] == "review"
    assert first["heuristic"] is None


def test_why_strings_are_backed_by_observed_data(tmp_path) -> None:
    d = video_dir(tmp_path)
    add_evidence(d)
    out = find_moments(d)
    weak = {m["start_s"]: m for m in out["weakest"]}
    strong = {m["start_s"]: m for m in out["strongest"]}

    # no speech and no cut at 3-6 s
    w = weak[3]
    assert w["observed"]["words"]["count"] == 0 and w["observed"]["scene_cuts"]["count"] == 0
    assert any("no speech and no scene cut for 3 s" in line and "dead air" in line for line in w["why"])
    assert w["suggested_action"]["action"] == "add" and (w["suggested_action"]["start_s"], w["suggested_action"]["end_s"]) == (3, 6)

    # speech (one word) but no cut at 21-24 s: must not claim silence
    w = weak[21]
    assert w["observed"]["words"]["count"] == 1 and w["observed"]["scene_cuts"]["count"] == 0
    assert any("no scene cut" in line for line in w["why"]) and not any("no speech" in line for line in w["why"])

    # quiet audio at 12-15 s: trim proposal, quiet line
    w = weak[12]
    assert w["observed"]["quiet_audio"] and any("quiet audio" in line for line in w["why"])
    assert w["suggested_action"]["action"] == "trim" and "note" in w["suggested_action"]

    # speech plus a cut at 9-12 s
    s = strong[9]
    assert s["observed"]["words"]["count"] == 2 and s["observed"]["scene_cuts"]["count"] == 1
    assert s["why"][0].startswith("speech (2 words) plus 1 scene cut") and any("BIG NEWS" in line for line in s["why"])
    assert s["suggested_action"] is None

    # no speech, no cut, no text at 15-18 s: only the generic "no measured feature" line, never a speech claim
    s = strong[15]
    assert s["observed"]["words"]["count"] == 0 and s["observed"]["scene_cuts"]["count"] == 0
    assert len(s["why"]) == 1 and "no measured feature" in s["why"][0] and "speech" not in s["why"][0].replace("no measured", "")

    # global rule: every claim maps back to the observed dict
    for m in out["weakest"] + out["strongest"]:
        text = " ".join(m["why"])
        obs = m["observed"]
        if "no speech" in text:
            assert obs["words"]["count"] == 0
        if text.startswith("speech") or " speech (" in text:
            assert obs["words"]["count"] > 0
        if "no scene cut" in text:
            assert obs["scene_cuts"]["count"] == 0
        if "quiet audio" in text:
            assert obs["quiet_audio"]
        if "on-screen text" in text and "only on-screen text" not in text:
            assert obs["on_screen_text"]


def test_write_moments_is_atomic_and_roundtrips(tmp_path) -> None:
    d = video_dir(tmp_path)
    out = find_moments(d)
    (d / MOMENTS_FILE).write_text("old")
    path = write_moments(d, out)
    assert path == d / MOMENTS_FILE and not (d / "moments.json.tmp").exists()
    assert json.loads(path.read_text()) == json.loads(json.dumps(out))
    write_moments(d, {"schema": "hookline.moments", "weakest": [], "strongest": []})  # a rewrite replaces the file whole
    assert json.loads(path.read_text())["weakest"] == [] and not (d / "moments.json.tmp").exists()
