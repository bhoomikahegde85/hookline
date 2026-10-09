"""Evidence-linked recommendations from hand-made evidence (no FFmpeg, no models, no network)."""

from __future__ import annotations

import copy

from hookline.recommendations import build_recommendations, evidence_index, invalid_evidence, recommendation_ids


def rows(over=None, n=12, step=0.5):
    out = []
    for i in range(n):
        r = {"segment_index": i, "start_s": i * step, "end_s": (i + 1) * step, "scene_index": 0 if i < 8 else 1,
             "visual_status": "available", "audio_status": "available", "speech_status": "available", "motion_mean": 5.0}
        r.update((over or {}).get(i, {}))
        out.append(r)
    return out


def ocr(samples):
    frames = [{"t_s": t, "status": st, "detections": [{"event_id": e} for e in ids]} for t, st, ids in samples]
    events = [{"event_id": e} for e in sorted({e for _, _, ids in samples for e in ids})]
    return {"measured": True, "status": "available", "frames": frames, "events": events, "sampling": {"interval_s": 1.0}}


WORDS = [{"word_index": 0, "start_s": 3.2, "end_s": 3.4, "segment_index": 6},
         {"word_index": 1, "start_s": 3.5, "end_s": 3.8, "segment_index": 7}]
EMPTY_OCR = ocr([(t, "available", []) for t in range(6)])


def rules(recs):
    return [r["rule"] for r in recs]


def test_late_first_word_and_opening_without_words() -> None:
    recs = build_recommendations(rows(), WORDS, EMPTY_OCR, None, None, 6.0)
    assert rules(recs) == ["late_first_word", "opening_without_words"]
    late = recs[0]
    assert {"type": "word", "id": 0} in late["evidence"] and late["source"] == "rule" and late["basis"] == "measured"
    assert late["end_s"] == 3.2 and "not validated" in late["caveat"]
    early = [{"word_index": 0, "start_s": 2.5, "end_s": 2.8, "segment_index": 5}]  # a word inside the 3 s opening
    assert rules(build_recommendations(rows(), early, EMPTY_OCR, None, None, 6.0)) == ["late_first_word"]
    assert invalid_evidence(recs, evidence_index(rows(), WORDS, EMPTY_OCR, None, None)) == []


def test_rules_never_fire_on_missing_failed_or_uncertain_evidence() -> None:
    failed_speech = rows({1: {"speech_status": "failed"}})
    assert build_recommendations(failed_speech, WORDS, EMPTY_OCR, None, None, 6.0) == []
    uncertain_speech = rows({0: {"speech_status": "uncertain"}})
    assert build_recommendations(uncertain_speech, WORDS, EMPTY_OCR, None, None, 6.0) == []
    failed_ocr = ocr([(0, "failed", []), (1, "available", []), (2, "available", [])])
    assert rules(build_recommendations(rows(), [], failed_ocr, None, None, 6.0)) == []
    assert rules(build_recommendations(rows(), [], None, None, None, 6.0)) == []  # OCR not run: no text claim
    with_text = ocr([(0, "available", [3]), (1, "available", [3]), (2, "available", [])])
    recs = build_recommendations(rows(), WORDS, with_text, None, None, 6.0)
    assert rules(recs) == ["late_first_word"] and {"type": "text_event", "id": 3} in recs[0]["evidence"]
    assert "text-led" in recs[0]["why"]


def test_quiet_stretch_needs_long_measured_silence_without_speech() -> None:
    lib = {"measured": True, "low_energy_intervals": [
        {"interval_id": 0, "start_s": 0.5, "end_s": 2.5},   # long, no words: flagged
        {"interval_id": 1, "start_s": 3.0, "end_s": 3.5},   # too short
        {"interval_id": 2, "start_s": 4.0, "end_s": 6.0}]}  # the quiet ending: skipped
    words = [{"word_index": 0, "start_s": 0.1, "end_s": 0.3, "segment_index": 0}]
    recs = build_recommendations(rows(), words, None, lib, None, 6.0)
    assert rules(recs) == ["quiet_stretch"] and recs[0]["evidence"][0] == {"type": "librosa_interval", "id": 0}
    talking = words + [{"word_index": 1, "start_s": 1.0, "end_s": 1.4, "segment_index": 2}]
    assert build_recommendations(rows(), talking, None, lib, None, 6.0) == []
    audio_failed = rows({2: {"audio_status": "failed"}})
    assert build_recommendations(audio_failed, words, None, lib, None, 6.0) == []
    assert build_recommendations(rows(), words, None, {"measured": False, "status": "failed"}, None, 6.0) == []


def test_static_stretch_respects_scenes_and_missing_visuals() -> None:
    still = {i: {"motion_mean": 0.2} for i in range(12)}
    recs = build_recommendations(rows(still), [{"word_index": 0, "start_s": 0.1, "end_s": 0.2, "segment_index": 0}],
                                 None, None, None, 6.0)
    assert rules(recs) == ["static_stretch"]  # scene 0 (0-4 s) only; scene 1 is 2 s
    assert (recs[0]["start_s"], recs[0]["end_s"]) == (0.0, 4.0)
    gap = copy.deepcopy(still)
    gap[3] = {"motion_mean": None, "visual_status": "failed"}
    assert build_recommendations(rows(gap), [], None, None, None, 6.0) == []


def test_gemini_suggestions_pass_through_as_interpretation() -> None:
    gem = {"measured": True, "status": "available", "analysis": {
        "moments": [{"id": "m0", "start_s": 1.0, "end_s": 2.0}],
        "edit_suggestions": [{"id": "e0", "start_s": 1.0, "end_s": 2.0, "action": "Tighten the reveal.",
                              "rationale": "Long pause.", "expected_effect": "Gets to the point sooner.",
                              "confidence": 0.5, "evidence_moment_ids": ["m0"]}]}}
    rs = rows()
    recs = build_recommendations(rs, [], None, None, gem, 6.0)
    (g,) = recs
    assert g["source"] == "gemini" and g["basis"] == "model_interpretation" and g["model_confidence"] == 0.5
    assert {"type": "gemini_moment", "id": "m0"} in g["evidence"] and {"type": "segment", "id": 2} in g["evidence"]
    assert invalid_evidence(recs, evidence_index(rs, [], None, None, gem)) == []
    assert recommendation_ids(recs, rs[2]) == ["r0"] and recommendation_ids(recs, rs[0]) is None


def test_invalid_evidence_is_detected() -> None:
    rs = rows()
    recs = [{"id": "r0", "start_s": 0.0, "end_s": 1.0, "evidence": [{"type": "word", "id": 7}, {"type": "segment", "id": 99}]},
            {"id": "r1", "start_s": 2.0, "end_s": 2.0, "evidence": []},
            {"id": "r2", "start_s": 0.0, "end_s": 1.0, "evidence": [{"type": "retention", "id": 1}]}]
    problems = invalid_evidence(recs, evidence_index(rs, WORDS, None, None, None))
    assert problems == ["r0: word 7 does not exist", "r0: segment 99 does not exist", "r1: no evidence",
                        "r1: empty interval", "r2: unknown evidence type 'retention'"]
    assert build_recommendations([], [], None, None, None, None) == []
