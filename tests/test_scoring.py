"""Heuristic segment scores over measured evidence: contributions, caps, missing modalities, rankings."""

from __future__ import annotations

import pytest

from hookline.scoring import QUALITY_WEIGHTS, RISK_WEIGHTS, score_video


def rows(over=None, n=8):
    out = []
    for i in range(n):
        r = {"segment_index": i, "start_s": i * 0.5, "end_s": (i + 1) * 0.5, "visual_status": "available", "motion_mean": 4.0,
             "speech_status": "available", "speech_present": False, "speech_word_count": 0, "text_status": "available",
             "text_present": False, "librosa_low_energy_ratio": 0.0, "scene_cut_at_start": False, "siglip_within_shot_change": 0.02}
        r.update((over or {}).get(i, {}))
        out.append(r)
    return out


def rule(id_, s, e, text="Trim the silent lead-in."):
    return {"id": id_, "source": "rule", "start_s": s, "end_s": e, "suggestion": text}


def by_index(scores):
    return {s["segment_index"]: s for s in scores["segments"]}


def test_no_evidence_means_zero_not_bad() -> None:
    scores = score_video("v", rows(), [])
    assert scores["basis"] == "measured_only" and scores["scoring_version"] == "2"
    assert all(s["attention_risk"] == 0 and s["moment_quality"] == 0 for s in scores["segments"])
    assert scores["highest_attention_risk"] == [] and scores["highest_moment_quality"] == []
    assert "gemini" not in str(scores).lower() and "model_confidence" not in str(scores)


def test_missing_modalities_never_add_risk() -> None:
    missing = rows({i: {"speech_status": "failed", "text_status": "failed", "visual_status": "failed", "motion_mean": None,
                          "librosa_low_energy_ratio": None} for i in range(8)})
    scores = score_video("v", missing, [])
    assert all(s["attention_risk"] == 0 and s["moment_quality"] == 0 for s in scores["segments"])
    assert "not retention" in scores["note"]


def test_measured_contributions_and_rule_recommendations() -> None:
    rs = rows({0: {"motion_mean": 0.3, "librosa_low_energy_ratio": 0.9},
                 1: {"speech_word_count": 3, "speech_present": True, "text_present": True, "scene_cut_at_start": True}})
    sc = by_index(score_video("v", rs, [rule("r0", 0.0, 0.5)]))
    assert sc[0]["attention_risk"] == (RISK_WEIGHTS["rule_recommendation"] + RISK_WEIGHTS["measured_static"]
                                       + RISK_WEIGHTS["measured_quiet_no_speech"])
    assert [c["source"] for c in sc[0]["risk_contributions"]] == ["rule", "measured", "measured"]
    assert sc[1]["moment_quality"] == (QUALITY_WEIGHTS["measured_speech"] + QUALITY_WEIGHTS["measured_text"]
                                       + QUALITY_WEIGHTS["measured_visual_change"])
    assert sc[2]["attention_risk"] == 0 and sc[2]["moment_quality"] == 0


def test_recommendations_align_to_overlapping_segments_only_and_cap_at_100() -> None:
    recs = [rule(f"r{i}", 1.0, 2.0) for i in range(8)]  # eight rules overlap segments 2 and 3 (end-exclusive)
    sc = by_index(score_video("v", rows(), recs))
    assert sc[2]["attention_risk"] == sc[3]["attention_risk"] == 100.0  # 8 x 15 capped
    assert sc[1]["attention_risk"] == 0 and sc[4]["attention_risk"] == 0


def test_rankings_group_adjacent_segments_into_spans() -> None:
    rs = rows({6: {"speech_word_count": 2, "speech_present": True}, 7: {"speech_word_count": 2, "speech_present": True}})
    s = score_video("v", rs, [rule("r0", 3.0, 4.0, "Tighten this stretch.")], top_n=2)
    assert s["highest_attention_risk"] == [{"start_s": 3.0, "end_s": 4.0, "attention_risk": RISK_WEIGHTS["rule_recommendation"],
                                            "segment_indices": [6, 7], "reason": "Tighten this stretch."}]
    assert s["highest_moment_quality"][0]["segment_indices"] == [6, 7]
    assert s["highest_moment_quality"][0]["moment_quality"] == pytest.approx(QUALITY_WEIGHTS["measured_speech"])
    assert s["weights"]["cap"] == 100
