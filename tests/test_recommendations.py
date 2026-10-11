"""Evidence-linked recommendations from hand-made evidence (no FFmpeg, no models, no network)."""

from __future__ import annotations

import copy

from hookline.recommendations import build_recommendations, evidence_index, invalid_evidence


def rows(over=None, n=12, step=0.5):
    out = []
    for i in range(n):
        r = {"segment_index": i, "start_s": i * step, "end_s": (i + 1) * step, "scene_index": 0 if i < 8 else 1,
             "visual_status": "available", "audio_status": "available", "speech_status": "available", "motion_mean": 5.0,
             "scene_cut_at_start": i == 3}  # a cut at 1.5 s: the early-visual-change rule stays quiet unless a test removes it
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
    recs = build_recommendations(rows(), WORDS, EMPTY_OCR, None, 6.0)
    assert rules(recs) == ["late_first_word", "opening_without_words"]
    late = recs[0]
    assert {"type": "word", "id": 0} in late["evidence"] and late["source"] == "rule" and late["basis"] == "measured"
    assert late["end_s"] == 3.2 and "not validated" in late["caveat"]
    early = [{"word_index": 0, "start_s": 2.5, "end_s": 2.8, "segment_index": 5}]  # a word inside the 3 s opening
    assert rules(build_recommendations(rows(), early, EMPTY_OCR, None, 6.0)) == ["late_first_word"]
    assert invalid_evidence(recs, evidence_index(rows(), WORDS, EMPTY_OCR, None)) == []


def test_rules_never_fire_on_missing_failed_or_uncertain_evidence() -> None:
    failed_speech = rows({1: {"speech_status": "failed"}})
    assert build_recommendations(failed_speech, WORDS, EMPTY_OCR, None, 6.0) == []
    uncertain_speech = rows({0: {"speech_status": "uncertain"}})
    assert build_recommendations(uncertain_speech, WORDS, EMPTY_OCR, None, 6.0) == []
    failed_ocr = ocr([(0, "failed", []), (1, "available", []), (2, "available", [])])
    assert rules(build_recommendations(rows(), [], failed_ocr, None, 6.0)) == []
    assert rules(build_recommendations(rows(), [], None, None, 6.0)) == []  # OCR not run: no text claim
    with_text = ocr([(0, "available", [3]), (1, "available", [3]), (2, "available", [])])
    recs = build_recommendations(rows(), WORDS, with_text, None, 6.0)
    assert rules(recs) == ["late_first_word"] and {"type": "text_event", "id": 3} in recs[0]["evidence"]
    assert "text-led" in recs[0]["why"]


def test_quiet_stretch_needs_long_measured_silence_without_speech() -> None:
    lib = {"measured": True, "low_energy_intervals": [
        {"interval_id": 0, "start_s": 0.5, "end_s": 2.5},   # long, no words: flagged
        {"interval_id": 1, "start_s": 3.0, "end_s": 3.5},   # too short
        {"interval_id": 2, "start_s": 4.0, "end_s": 6.0}]}  # the quiet ending: skipped
    words = [{"word_index": 0, "start_s": 0.1, "end_s": 0.3, "segment_index": 0}]
    recs = build_recommendations(rows(), words, None, lib, 6.0)
    assert rules(recs) == ["quiet_stretch"] and recs[0]["evidence"][0] == {"type": "librosa_interval", "id": 0}
    talking = words + [{"word_index": 1, "start_s": 1.0, "end_s": 1.4, "segment_index": 2}]
    assert build_recommendations(rows(), talking, None, lib, 6.0) == []
    audio_failed = rows({2: {"audio_status": "failed"}})
    assert build_recommendations(audio_failed, words, None, lib, 6.0) == []
    assert build_recommendations(rows(), words, None, {"measured": False, "status": "failed"}, 6.0) == []


def test_static_stretch_respects_scenes_and_missing_visuals() -> None:
    still = {i: {"motion_mean": 0.2} for i in range(12)}
    recs = build_recommendations(rows(still), [{"word_index": 0, "start_s": 0.1, "end_s": 0.2, "segment_index": 0}],
                                 None, None, 6.0)
    assert rules(recs) == ["static_stretch"]  # scene 0 (0-4 s) only; scene 1 is 2 s
    assert (recs[0]["start_s"], recs[0]["end_s"]) == (0.0, 4.0)
    gap = copy.deepcopy(still)
    gap[3] = {"motion_mean": None, "visual_status": "failed"}
    assert build_recommendations(rows(gap), [], None, None, 6.0) == []


def test_no_early_visual_change_needs_measured_vision_and_no_cut_or_content_change() -> None:
    long_rows = lambda over=None: rows({3: {"scene_cut_at_start": False}} | (over or {}), n=14)  # noqa: E731  7 s clip, no cut
    recs = build_recommendations(long_rows(), [], None, None, 7.0)
    assert "no_early_visual_change" in rules(recs)
    early = next(r for r in recs if r["rule"] == "no_early_visual_change")
    assert (early["start_s"], early["end_s"]) == (0.0, 3.0) and "SigLIP was not run" in early["why"]
    assert {"type": "segment", "id": 0} in early["evidence"] and early["source"] == "rule" and early["basis"] == "measured"
    cut = rows(n=14)  # default fixture: a cut at 1.5 s
    assert "no_early_visual_change" not in rules(build_recommendations(cut, [], None, None, 7.0))
    changed = long_rows({2: {"siglip_within_shot_change": 0.4}})
    assert "no_early_visual_change" not in rules(build_recommendations(changed, [], None, None, 7.0))
    sig = long_rows({0: {"siglip_within_shot_change": 0.02}})
    assert "SigLIP frame embeddings also show no content change" in next(
        r for r in build_recommendations(sig, [], None, None, 7.0) if r["rule"] == "no_early_visual_change")["why"]
    failed = long_rows({1: {"visual_status": "failed"}})
    assert "no_early_visual_change" not in rules(build_recommendations(failed, [], None, None, 7.0))
    short = rows({3: {"scene_cut_at_start": False}}, n=11)  # 5.5 s: the opening is most of the clip
    assert "no_early_visual_change" not in rules(build_recommendations(short, [], None, None, 5.5))


def test_long_shot_needs_a_measured_scene_of_at_least_ten_seconds() -> None:
    one_scene = [{"segment_index": i, "start_s": i * 0.5, "end_s": (i + 1) * 0.5, "scene_index": 0, "visual_status": "available",
                  "audio_status": "available", "speech_status": "available", "motion_mean": 5.0} for i in range(24)]
    recs = build_recommendations(one_scene, [], None, None, 12.0)
    long = next(r for r in recs if r["rule"] == "long_shot")
    assert (long["start_s"], long["end_s"]) == (0.0, 12.0) and len(long["evidence"]) == 24
    assert "long_shot" not in rules(build_recommendations(one_scene[:18], [], None, None, 9.0))  # 9 s only
    gap = [dict(r) for r in one_scene]
    gap[5]["visual_status"] = "failed"
    assert "long_shot" not in rules(build_recommendations(gap, [], None, None, 12.0))
    assert invalid_evidence(recs, evidence_index(one_scene, [], None, None)) == []


def test_repeated_text_cites_both_ocr_events() -> None:
    events = [{"event_id": 0, "text": "SALE", "first_seen_s": 0.5, "visible_until_s": 1.5, "repeat_of": None},
              {"event_id": 1, "text": "SALE", "first_seen_s": 4.0, "visible_until_s": 5.0, "repeat_of": 0}]
    o = {"measured": True, "status": "available", "frames": [], "events": events, "sampling": {"interval_s": 1.0}}
    recs = build_recommendations(rows(), [], o, None, 6.0)
    rep = next(r for r in recs if r["rule"] == "repeated_text")
    assert (rep["start_s"], rep["end_s"]) == (4.0, 5.0) and "'SALE'" in rep["why"]
    assert [e["id"] for e in rep["evidence"] if e["type"] == "text_event"] == [0, 1]
    assert invalid_evidence(recs, evidence_index(rows(), [], o, None)) == []
    assert "repeated_text" not in rules(build_recommendations(rows(), [], {**o, "measured": False, "status": "failed"}, None, 6.0))
    assert "repeated_text" not in rules(build_recommendations(rows(), [], None, None, 6.0))
    words = [{"event_id": 10 + i, "text": w, "first_seen_s": float(i), "visible_until_s": i + 1.0, "repeat_of": None} for i, w in enumerate(["the", "bag", "is", "big", "and", "light"])]
    caps = {**o, "events": words + [{"event_id": 20, "text": "the", "first_seen_s": 4.0, "visible_until_s": 5.0, "repeat_of": 10}]}
    assert "repeated_text" not in rules(build_recommendations(rows(), [], caps, None, 6.0))  # word-by-word captions repeat common words
