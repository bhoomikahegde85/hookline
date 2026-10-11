"""Hook report: percentiles against the reference reels, hand-written direction rules, verdict sources and graceful degradation.
Synthetic reference set with known values, so every expected percentile is computed by hand."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hookline import hook_report as hr
from hookline.views_eval import BANDS

N = 40  # regular reels: reference reel i has first_word_s = 0.1 + 0.1 i, ..., an evenly spread ladder per feature
ACCOUNT_WORDS = ("band", "handle", "views", "follower")


def write_opening(d: Path, first_word=1.0, first_cut=1.0, cuts_per_s=0.4, motion=10.0, brightness=120.0, rms=-30.0, wps=2.5, first_text=1.0, speech=True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "opening_multimodal.json").write_text(json.dumps({
        "analysed_seconds": 5.0, "duration_s": 30.0,
        "visual": {"stream_status": "available", "scene_cuts": cuts_per_s * 5, "first_scene_cut_s": first_cut, "motion_mean": motion, "brightness_mean": brightness},
        "audio": {"stream_status": "available", "rms_dbfs": rms},
        "speech": {"status": "available", "speech_detected": speech, "first_word_s": first_word if speech else None, "words_per_s": wps},
        "text": {"status": "available", "first_text_s": first_text}}))


def write_ttcc(d: Path, level: float) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "ttcc_retention.json").write_text(json.dumps({"status": "success", "retention": [1.0] + [level * 0.97 ** t for t in range(1, 31)], "horizon_s": 30,
                                                       "inference": {"window_end_s": 30.0}, "policy": "hook"}))


def write_siglip(d: Path, person: float) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "siglip.json").write_text(json.dumps({"status": "available", "frames": [{"t_s": 0.5, "scores": {"person_closeup": person, "talking_to_camera": 0.0}}]}))


@pytest.fixture()
def world(tmp_path):
    data, feats, ttcc = tmp_path / "data", tmp_path / "features", tmp_path / "ttcc"
    meta: dict[str, list] = {"face": [], "faceless": [], "outliers": []}
    for i in range(N):
        vid = f"R{i:02d}"
        meta["face" if i % 2 else "faceless"].append([vid, f"h{i}", BANDS[i % 4], 1000 + i])
        write_opening(feats / vid, first_word=0.1 + 0.1 * i, first_cut=0.2 + 0.1 * i, cuts_per_s=0.01 * (i + 1), motion=1 + 0.5 * i, brightness=60 + 3 * i, rms=-40 + 0.5 * i,
                      wps=1 + 0.1 * i, first_text=0.3 + 0.1 * i)
        write_ttcc(ttcc / vid, 0.5 + 0.01 * i)
        write_siglip(feats / vid, 0.02 * i)
    for j in range(4):
        meta["outliers"].append([f"O{j}", f"o{j}", BANDS[j], 10**9])
        write_opening(feats / f"O{j}", first_word=0.05)                       # would shift the percentiles if outliers were in the reference
        write_ttcc(ttcc / f"O{j}", 0.99)
    for g, rows in meta.items():
        (data / g).mkdir(parents=True)
        with (data / g / "metadata.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "instagram_handle", "follower_band", "views"])
            w.writerows(rows)
    return data, feats, ttcc, tmp_path


def new_video(tmp_path: Path, name="NEW", **kw) -> Path:
    d = tmp_path / "newfeat" / name
    write_opening(d, **kw)
    return d


def factor(rep: dict, key: str) -> dict:
    return next(f for f in rep["factors"] if f["key"] == key)


def test_percentiles_are_among_the_reference_reels_and_outliers_are_ignored(world) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, first_word=0.25)
    write_ttcc(d, 0.7)
    rep = hr.build_hook_report(d, data, feats, ttcc)
    assert rep["status"] == "ok" and rep["schema"] == "hookline.hook_report" and rep["video_id"] == "NEW"
    assert rep["reference"]["n_reels"] == N                                    # 40 regular reels, not the 4 outliers (0.05 s first words)
    f = factor(rep, "first_word_s")
    assert f["value"] == 0.25 and f["percentile_vs_reference"] == pytest.approx(100 * 2 / N)  # reels at 0.1 s and 0.2 s are lower
    assert factor(rep, "ttcc_R3")["percentile_vs_reference"] is not None and rep["reference"]["n_with_ttcc"] == N


def test_the_video_itself_is_excluded_from_the_reference(world) -> None:
    data, feats, ttcc, tmp = world
    rep = hr.build_hook_report(feats / "R10", data, feats, ttcc)               # R10 is in the reference set: first_word_s = 1.1
    assert rep["reference"]["n_reels"] == N - 1
    assert factor(rep, "first_word_s")["percentile_vs_reference"] == pytest.approx(100 * 10 / (N - 1), abs=0.06)  # not (10 + 0.5) / 40


def test_direction_rules(world) -> None:
    data, feats, ttcc, tmp = world
    fast = hr.build_hook_report(new_video(tmp, "FAST", first_word=0.2), data, feats, ttcc)
    late = hr.build_hook_report(new_video(tmp, "LATE", first_word=4.8), data, feats, ttcc)
    assert factor(fast, "first_word_s")["direction"] == "helps" and "first_word_s" in fast["strengths"]
    assert factor(late, "first_word_s")["direction"] == "hurts" and "first_word_s" in late["weaknesses"]
    mid, odd = (hr.build_hook_report(new_video(tmp, n, brightness=b), data, feats, ttcc) for n, b in (("MID", 118.5), ("DARK", 10.0)))
    assert factor(mid, "brightness_mean_0_5")["direction"] == "helps" and factor(odd, "brightness_mean_0_5")["direction"] == "hurts"   # mid_best: the median is the target
    assert "median" in factor(odd, "brightness_mean_0_5")["reading"]
    assert factor(hr.build_hook_report(new_video(tmp, "LOUD", rms=-5.0), data, feats, ttcc), "rms_dbfs_0_5")["direction"] == "helps"          # higher_better
    for f in fast["factors"]:
        assert f["why_it_matters"].startswith("Rule of thumb, not validated") and f["source"]


def test_events_that_never_happen_are_not_held_against_the_video(world) -> None:
    data, feats, ttcc, tmp = world
    d = tmp / "newfeat" / "STILL"
    write_opening(d, speech=False, cuts_per_s=0.0, first_text=None)
    o = json.loads((d / "opening_multimodal.json").read_text())
    o["visual"]["first_scene_cut_s"] = None
    o["text"]["first_text_s"] = None
    (d / "opening_multimodal.json").write_text(json.dumps(o))
    rep = hr.build_hook_report(d, data, feats, ttcc)
    for key in ("first_word_s", "first_cut_s", "first_text_s"):
        f = factor(rep, key)
        assert f["value"] == 5.0 and f["direction"] == "neutral" and "no " in f["reading"]
    assert "single continuous shot" in factor(rep, "first_cut_s")["reading"]
    late_cut = hr.build_hook_report(new_video(tmp, "SLOWCUT", first_cut=4.9, cuts_per_s=0.2), data, feats, ttcc)
    assert factor(late_cut, "first_cut_s")["direction"] == "hurts"


def test_ttcc_factors_exist_only_with_a_ttcc_prediction(world) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, "T")
    without = hr.build_hook_report(d, data, feats, ttcc)
    assert not any(f["key"].startswith("ttcc_") for f in without["factors"])
    write_ttcc(d, 0.95)                                                        # above every reference reel (0.5-0.89)
    with_ttcc = hr.build_hook_report(d, data, feats, ttcc)
    assert {"ttcc_R3", "ttcc_R5", "ttcc_R10"} <= {f["key"] for f in with_ttcc["factors"]}
    r3 = factor(with_ttcc, "ttcc_R3")
    assert r3["value"] == pytest.approx(0.95 * 0.97 ** 3, abs=1e-3) and r3["direction"] == "helps" and r3["percentile_vs_reference"] == 100.0
    assert "model prediction" in r3["source"]
    stitched = json.loads((d / "ttcc_retention.json").read_text()) | {"stitched": True}
    (d / "ttcc_retention.json").write_text(json.dumps(stitched))                # a stitched curve is not a hook prediction
    assert not any(f["key"].startswith("ttcc_") for f in hr.build_hook_report(d, data, feats, ttcc)["factors"])


def test_person_factor_needs_siglip_output(world) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, "P")
    assert "person_max_0_3s" not in {f["key"] for f in hr.build_hook_report(d, data, feats, ttcc)["factors"]}
    write_siglip(d, 0.95)
    f = factor(hr.build_hook_report(d, data, feats, ttcc), "person_max_0_3s")
    assert f["value"] == 0.95 and f["direction"] == "helps"


def test_strengths_and_weaknesses_are_ranked_and_capped(world) -> None:
    data, feats, ttcc, tmp = world
    weak = hr.build_hook_report(new_video(tmp, "BAD", first_word=4.9, first_cut=4.8, cuts_per_s=0.01, motion=0.2, brightness=5.0, rms=-60.0, wps=0.1, first_text=4.9), data, feats, ttcc)
    assert len(weak["weaknesses"]) == 3 and not weak["strengths"] and all(factor(weak, k)["direction"] == "hurts" for k in weak["weaknesses"])
    p = {k: factor(weak, k)["percentile_vs_reference"] for k in weak["weaknesses"]}
    fav = {k: (100 - v if hr.HOOK_RULES[k]["rule"] == "lower_better" else v if hr.HOOK_RULES[k]["rule"] == "higher_better" else 100 - 2 * abs(v - 50)) for k, v in p.items()}
    assert [fav[k] for k in weak["weaknesses"]] == sorted(fav.values())       # worst first
    strong = hr.build_hook_report(new_video(tmp, "GOOD", first_word=0.2, first_cut=0.3, cuts_per_s=0.2, motion=19.0, brightness=118.5, rms=-21.0, first_text=0.4), data, feats, ttcc)
    assert len(strong["strengths"]) == 3 and not strong["weaknesses"]          # more than three factors help: only the best three are listed
    assert sum(f["direction"] == "helps" for f in strong["factors"]) > 3 and factor(strong, "first_word_s")["direction"] == "helps"


def test_verdict_from_the_ttcc_hook_score(world) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, "V")
    for score, label in ((80, "strong"), (67, "strong"), (66, "average"), (34, "average"), (33, "weak"), (5, "weak")):
        (d / "hook_scores.json").write_text(json.dumps({"hook_score": {"score": score}, "breakout_score": {"score": 99}}))
        v = hr.build_hook_report(d, data, feats, ttcc)["verdict"]
        assert v["label"] == label and v["percentile"] == score and "model prediction" in v["basis"] and "TTCC" in v["basis"]


def test_verdict_from_measured_features_when_there_is_no_ttcc_score(world) -> None:
    data, feats, ttcc, tmp = world
    good = hr.build_hook_report(new_video(tmp, "G", first_word=0.2, first_cut=0.3, cuts_per_s=0.2, motion=19.0, brightness=118.5, first_text=0.4), data, feats, ttcc)
    bad = hr.build_hook_report(new_video(tmp, "B", first_word=4.9, first_cut=4.8, cuts_per_s=0.01, motion=0.2, brightness=5.0, rms=-60.0), data, feats, ttcc)
    assert (good["verdict"]["label"], bad["verdict"]["label"]) == ("strong", "weak")
    assert good["verdict"]["percentile"] is None and "from measured opening features only, no TTCC" in good["verdict"]["basis"]
    (new_video(tmp, "G") / "hook_scores.json").write_text(json.dumps({"hook_score": {"score": None, "reason": "no TTCC"}}))   # a null score falls back to the features
    assert hr.build_hook_report(new_video(tmp, "G", first_word=0.2, first_cut=0.3, cuts_per_s=0.2, motion=19.0, brightness=118.5, first_text=0.4), data, feats, ttcc)["verdict"]["label"] == "strong"


def test_verdict_is_not_rated_with_fewer_than_four_comparable_factors(world) -> None:
    data, feats, ttcc, tmp = world
    d = tmp / "newfeat" / "THIN"
    d.mkdir(parents=True)
    (d / "opening_multimodal.json").write_text(json.dumps({"analysed_seconds": 5.0, "visual": {"stream_status": "available", "scene_cuts": 2, "first_scene_cut_s": 1.0, "motion_mean": 9.0}}))   # no brightness, audio, speech or text
    rep = hr.build_hook_report(d, data, feats, ttcc)
    unknown = {f["key"] for f in rep["factors"] if f["direction"] == "unknown"}
    assert {"rms_dbfs_0_5", "first_word_s", "words_per_s_0_5", "first_text_s"} <= unknown
    assert factor(rep, "rms_dbfs_0_5")["value"] is None and factor(rep, "rms_dbfs_0_5")["percentile_vs_reference"] is None
    assert rep["status"] == "ok" and rep["verdict"]["label"] == "not rated" and rep["verdict"]["percentile"] is None
    json.dumps(rep, allow_nan=False)                                           # no NaN leaks into the file


def test_missing_inputs_never_raise(world, tmp_path) -> None:
    data, feats, ttcc, tmp = world
    empty = tmp / "empty"
    empty.mkdir()
    rep = hr.build_hook_report(empty, data, feats, ttcc)
    assert rep["status"] == "unavailable" and rep["reason"] and rep["verdict"]["label"] == "not rated" and rep["factors"] == [] and rep["strengths"] == []
    bad = tmp / "corrupt"
    bad.mkdir()
    (bad / "opening_multimodal.json").write_text("{not json")
    assert hr.build_hook_report(bad, data, feats, ttcc)["status"] == "unavailable"
    d = new_video(tmp, "OFFLINE", first_word=0.2)
    nodata = hr.build_hook_report(d, tmp / "nowhere", feats, None)             # reference set missing: the values stay, the comparison cannot be made
    assert nodata["status"] == "ok" and nodata["reference"]["n_reels"] == 0
    f = factor(nodata, "first_word_s")
    assert f["value"] == 0.2 and f["percentile_vs_reference"] is None and f["direction"] == "unknown" and nodata["verdict"]["label"] == "not rated"
    (feats / "R05" / "siglip.json").write_text("{broken")                      # one corrupt artifact in the reference set drops only that reel
    (feats / "R06" / "opening_multimodal.json").write_text("[]")
    rep = hr.build_hook_report(new_video(tmp, "AFTERCORRUPT", first_word=0.2), data, feats, ttcc)
    assert rep["status"] == "ok" and N - 2 <= rep["reference"]["n_reels"] <= N and factor(rep, "first_word_s")["percentile_vs_reference"] is not None


def test_gemini_text_only_when_the_analysis_was_measured(world) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, "GEM")
    assert hr.build_hook_report(d, data, feats, ttcc)["gemini_hook_assessment"] is None
    art = {"status": "available", "measured": True, "analysis": {"overall": {"hook_assessment": "  Opens on a face with a question.  "}}}
    (d / "gemini_edit.json").write_text(json.dumps(art))
    assert hr.build_hook_report(d, data, feats, ttcc)["gemini_hook_assessment"] == "Opens on a face with a question."
    for broken in (art | {"measured": False}, art | {"status": "failed"}, art | {"analysis": {"overall": {"hook_assessment": "   "}}}):
        (d / "gemini_edit.json").write_text(json.dumps(broken))
        assert hr.build_hook_report(d, data, feats, ttcc)["gemini_hook_assessment"] is None


def test_no_account_data_and_the_note_says_what_this_is(world) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, "ACC", first_word=0.2)
    (d / "hook_scores.json").write_text(json.dumps({"hook_score": {"score": 70}, "breakout_score": {"score": 90}}))
    (d / "gemini_edit.json").write_text(json.dumps({"status": "available", "measured": True, "analysis": {"overall": {"hook_assessment": "Quick start."}}}))
    rep = hr.build_hook_report(d, data, feats, ttcc)
    text = json.dumps(rep).lower()
    assert not [w for w in ACCOUNT_WORDS if w in text]
    assert "reference reels" in rep["note"] and "not validated against audience data" in rep["note"] and "not a viral verdict" in rep["note"]
    assert set(rep) == {"schema", "video_id", "status", "reason", "verdict", "factors", "strengths", "weaknesses", "gemini_hook_assessment", "reference", "note"}
    assert set(rep["factors"][0]) == {"key", "label", "value", "unit", "percentile_vs_reference", "reading", "direction", "why_it_matters", "source"}


def test_write_is_atomic_and_round_trips(world, monkeypatch) -> None:
    data, feats, ttcc, tmp = world
    d = new_video(tmp, "W")
    path = d / "hook_report.json"
    path.write_text('{"old": true}')
    rep = hr.build_hook_report(d, data, feats, ttcc)
    seen = {}
    real = Path.replace

    def spy(self, target):
        seen["tmp_has_new"] = json.loads(self.read_text())["video_id"] == "W"
        seen["final_still_old"] = json.loads(Path(target).read_text()) == {"old": True}      # the target is untouched until the swap
        return real(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    out = hr.write_hook_report(d, rep)
    assert out == path and seen == {"tmp_has_new": True, "final_still_old": True}
    assert json.loads(path.read_text()) == json.loads(json.dumps(rep)) and not list(d.glob("*.tmp"))
