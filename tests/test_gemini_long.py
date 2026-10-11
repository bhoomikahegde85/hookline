"""Gemini on videos longer than the short-reel case: which seconds are sent (cost cap), original timestamps, evidence limited to the sent clips,
validation against them and caching. A fake transport / SDK stands in for Google: no network, no key, no cost."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from test_gemini_edit import ENV, NO_DOTENV, FakeTransport, fake_sdk

from hookline.config import GeminiConfig
from hookline.dropoffs import write_dropoffs
from hookline.gemini_edit import GenaiTransport, build_evidence_pack, cached_gemini_edit, extract_gemini_edit, select_ranges, validate, write_gemini_edit

CFG = GeminiConfig(mode="remote", allow_remote=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def curve(n: int, drops: dict[tuple[int, int], float]) -> list[float]:
    """R(0..n): a small steady loss plus extra hazard in the given (start, end) second ranges (a stand-in for TTCC's prediction)."""
    r = [1.0]
    for t in range(1, n + 1):
        extra = sum(h for (a, b), h in drops.items() if a < t <= b)
        r.append(r[-1] * math.exp(-(0.004 + extra)))
    return r


def make_long(tmp_path: Path, n: int, drops: dict, name: str = "long") -> tuple[Path, Path]:
    d = tmp_path / "features" / name
    d.mkdir(parents=True)
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"VIDEO" * 100)
    r = curve(n, drops)
    (d / "ttcc_retention.json").write_text(json.dumps({
        "status": "success", "video_id": name, "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "source_path": str(video),
        "cache_key": "k", "horizon_s": n, "duration_s": float(n), "partial_coverage": False, "stitched": True, "model": {"checkpoint_repo": "m"},
        "timestamps_s": list(range(n + 1)), "retention": r, "inference": {"deviations_from_upstream_eval": []}}))
    (d / "speech.json").write_text(json.dumps({"measured": True, "status": "available", "language": "en"}))
    rows = "".join(f"{name},{i},{i},{t}.2,{t}.6,word{t},0.9,0,\n" for i, t in enumerate(range(0, n, 5)))
    (d / "words.csv").write_text("video_id,word_index,segment_index,start_s,end_s,word,probability,asr_segment_id,asr_flags\n" + rows)
    (d / "ocr.json").write_text(json.dumps({"measured": True, "events": [
        {"event_id": i, "text": f"TEXT{t}", "first_seen_s": float(t), "visible_until_s": t + 2.0} for i, t in enumerate(range(0, n, 20))]}))
    (d / "librosa.json").write_text(json.dumps({"measured": True, "low_energy_intervals": []}))
    (d / "report.json").write_text(json.dumps({"scene_cut_times": [float(t) for t in range(3, n, 25)], "segments": []}))
    (d / "recommendations.json").write_text(json.dumps({"recommendations": []}))
    assert write_dropoffs(d)
    return d, video


DROPS = {(0, 1): 0.25, (100, 106): 0.20, (200, 210): 0.22, (330, 333): 0.20}  # steep seconds inside the 12 even segments of a 400 s video (33 s each)


def flagged(pack: dict) -> list[dict]:
    return [s for s in pack["ttcc"]["segments"] if s["kind"] in ("steep_drop", "sustained_decline")]


def answer_for(pack: dict, edits: dict[str, tuple[float, float]] | None = None) -> dict:
    segs = []
    for sid in pack["analysis_scope"]["analysed_segments"]:
        s = next(x for x in pack["ttcc"]["segments"] if x["id"] == sid)
        a, b = (edits or {}).get(sid, (s["start_s"], min(s["end_s"], s["start_s"] + 2)))
        segs.append({"segment_id": sid, "summary": "s", "hypothesis": "may be slow", "hypothesis_confidence": 0.4,
                     "observations": [{"kind": "observation", "text": "seen", "evidence_refs": [sid]}],
                     "suggestions": [{"category": "adjust_pacing", "action": "Tighten this stretch.", "edit": {"type": "review", "start_s": a, "end_s": b},
                                      "rationale": "r", "expected_effect": "may help", "confidence": 0.4, "evidence_refs": [sid]}]})
    return {"segments": segs, "overall": {"hook_assessment": "x", "top_priorities": pack["analysis_scope"]["analysed_segments"][:2]}}


def run(d, video, cfg=CFG, transport=None):
    t = transport or FakeTransport(answer_for(build_evidence_pack(d, cfg)))
    return extract_gemini_edit(d, video, cfg, lambda key: t, env=ENV, dotenv=NO_DOTENV, root=PROJECT_ROOT), t


# ---------------------------------------------------------------- which seconds are sent

def test_a_video_that_fits_the_budget_is_sent_whole(tmp_path) -> None:
    d, video = make_long(tmp_path, 60, {(0, 1): 0.25, (30, 36): 0.1})
    art, t = run(d, video)
    assert t.calls[0]["ranges"] == [(0.0, 60.0)] and art["analysis_scope"]["whole_predicted_span"] is True
    assert art["analysis_scope"]["analysed_s"] == 60.0 and art["analysis_scope"]["not_analysed"] == []
    assert "MULTI" not in t.calls[0]["prompt"] and "separate clips" not in t.calls[0]["prompt"] and "seconds 0 to 60" in t.calls[0]["prompt"]


def test_a_long_video_is_sent_as_the_opening_plus_flagged_segments_within_the_budget(tmp_path) -> None:
    d, video = make_long(tmp_path, 400, DROPS)
    pack = build_evidence_pack(d, CFG)
    scope = pack["analysis_scope"]
    assert scope["whole_predicted_span"] is False and scope["clips_s"][0][0] == 0.0
    assert sum(b - a for a, b in scope["clips_s"]) <= CFG.max_analysed_s
    for s in flagged(pack):                                                    # every flagged segment fits the budget here: a clip covers it with margin
        assert s["analysed"] is True and any(a <= max(0.0, s["start_s"] - CFG.context_margin_s) + 1e-6 and min(400.0, s["end_s"] + CFG.context_margin_s) <= b + 1e-6
                                             for a, b in scope["clips_s"])
    assert len(scope["clips_s"]) >= 3 and scope["not_analysed"] and all("budget" in x["reason"] or "outside" in x["reason"] for x in scope["not_analysed"])
    art, t = run(d, video)
    sent = t.calls[0]["ranges"]
    assert sent == [tuple(r) for r in pack["_ranges"]] and art["analysis_scope"]["analysed_s"] < 400 * 0.5
    assert "separate clips" in t.calls[0]["prompt"] and "ORIGINAL" in t.calls[0]["prompt"]
    assert art["status"] in ("available", "uncertain") and art["analysis_scope"]["frames_sampled_approx"] == round(art["analysis_scope"]["analysed_s"] * CFG.fps)
    assert art["usage"] == {"total_token_count": 1234}                           # the real cost is recorded with the result


def test_budget_goes_to_the_largest_predicted_falls_first(tmp_path) -> None:
    d, _ = make_long(tmp_path, 400, DROPS)
    tight = replace(CFG, max_analysed_s=80.0, hook_s=5.0, context_margin_s=3.0)  # room for the opening and one 33 s segment, not two
    pack = build_evidence_pack(d, tight)
    scope = pack["analysis_scope"]
    assert sum(b - a for a, b in scope["clips_s"]) <= 80.0
    kept = [s for s in flagged(pack) if s["analysed"]]
    dropped = [s for s in flagged(pack) if not s["analysed"]]
    assert kept and dropped
    assert min(s["rank_by_predicted_fall"] for s in kept) < min(s["rank_by_predicted_fall"] for s in dropped)  # the biggest fall made it
    assert {x["id"] for x in scope["not_analysed"]} >= {s["id"] for s in dropped}
    assert all("analysis budget" in x["reason"] for x in scope["not_analysed"] if x["id"] in {s["id"] for s in dropped})
    # the retention trend of the WHOLE video is still given, so the model sees where the falls are even where it sees no frames
    assert len(pack["ttcc"]["retention_by_second"]) == 401


def test_evidence_is_limited_to_the_clips_that_are_sent(tmp_path) -> None:
    d, _ = make_long(tmp_path, 400, DROPS)
    pack = build_evidence_pack(d, CFG)
    ranges = pack["_ranges"]
    inside = lambda a, b: any(a < r1 and b >= r0 for r0, r1 in ranges)  # noqa: E731
    assert pack["words"] and all(inside(w["t0"], w["t1"]) for w in pack["words"])
    assert all(inside(e["t0"], e["t1"]) for e in pack["on_screen_text"]) and all(inside(c["t"], c["t"]) for c in pack["scene_cuts"])
    everything = build_evidence_pack(d, replace(CFG, max_analysed_s=1000.0))
    assert len(everything["words"]) > len(pack["words"]) and len(json.dumps(pack["words"])) < len(json.dumps(everything["words"]))


def test_select_ranges_edges() -> None:
    segs = [{"id": "s0", "kind": "steep_drop", "start_s": 0, "end_s": 2, "rank_by_predicted_loss": 1},
            {"id": "s1", "kind": "unflagged", "start_s": 2, "end_s": 50, "rank_by_predicted_loss": 3},
            {"id": "s2", "kind": "sustained_decline", "start_s": 50, "end_s": 200, "rank_by_predicted_loss": 2}]
    sel = select_ranges(segs, 200.0, replace(CFG, max_analysed_s=60.0))
    assert sel["ranges"] == [[0.0, 5.0]] and sel["analysed"] == ["s0"] and [x["id"] for x in sel["not_analysed"]] == ["s1", "s2"]  # 150 s segment never fits
    assert "budget" in sel["not_analysed"][1]["reason"]
    assert select_ranges(segs, 200.0, None)["full"] is True and select_ranges(segs, 60.0, CFG)["ranges"] == [[0.0, 60.0]]
    merged = select_ranges([{"id": "a", "kind": "steep_drop", "start_s": 10, "end_s": 20, "rank_by_predicted_loss": 1},
                            {"id": "b", "kind": "steep_drop", "start_s": 23, "end_s": 30, "rank_by_predicted_loss": 2}], 500.0, CFG)
    assert merged["ranges"] == [[0.0, 5.0], [7.0, 33.0]]                        # nearby clips (<= 1 s apart) become one; the opening stays separate here


# ---------------------------------------------------------------- validation against the clips

def test_timestamps_must_be_original_seconds_inside_the_clips(tmp_path) -> None:
    d, _ = make_long(tmp_path, 400, DROPS)
    pack = build_evidence_pack(d, CFG)
    first = next(s for s in flagged(pack) if s["analysed"] and s["start_s"] > 50)
    outside = next(s for s in pack["ttcc"]["segments"] if not s["analysed"])
    good = {"type": "review", "start_s": float(first["start_s"]), "end_s": float(first["start_s"]) + 2}
    sug = lambda edit: {"category": "adjust_pacing", "action": "x", "edit": edit, "rationale": "r", "expected_effect": "may", "confidence": 0.5,  # noqa: E731
                        "evidence_refs": [first["id"]]}
    raw = {"segments": [
        {"segment_id": first["id"], "summary": "s", "hypothesis": "h", "hypothesis_confidence": 0.5, "observations": [],
         "suggestions": [sug(good), sug({"type": "review", "start_s": 2.0, "end_s": 4.0}),      # clip-relative time instead of original time
                         sug({"type": "review", "start_s": 399.0, "end_s": 400.0})]},
        {"segment_id": outside["id"], "summary": "invented", "hypothesis": "h", "hypothesis_confidence": 0.5, "observations": [], "suggestions": []}],
        "overall": {"hook_assessment": "x", "top_priorities": [first["id"], outside["id"]]}}
    clean, report = validate(raw, pack, CFG)
    assert [s["segment_id"] for s in clean["segments"]] == [first["id"]]
    assert [s["edit"]["start_s"] for s in clean["segments"][0]["suggestions"]] == [good["start_s"]]
    assert clean["overall"]["top_priorities"] == [first["id"]]
    assert any("not part of the analysed clips" in i for i in report["issues"]) and report["dropped"] >= 3
    assert not any("no usable entry" in i and outside["id"] in i for i in report["issues"])  # segments outside the clips are not 'missing'


def test_suggestions_keep_original_timestamps_end_to_end(tmp_path) -> None:
    d, video = make_long(tmp_path, 400, DROPS)
    pack = build_evidence_pack(d, CFG)
    target = next(s for s in flagged(pack) if s["analysed"] and s["start_s"] > 150)
    art, _ = run(d, video, transport=FakeTransport(answer_for(pack, {target["id"]: (target["start_s"] + 1.0, target["start_s"] + 3.0)})))
    seg = next(s for s in art["analysis"]["segments"] if s["segment_id"] == target["id"])
    assert seg["suggestions"][0]["edit"] == {"type": "review", "start_s": target["start_s"] + 1.0, "end_s": target["start_s"] + 3.0}
    assert art["status"] == "available" and art["validation"]["dropped"] == 0
    assert write_gemini_edit(d, art)


# ---------------------------------------------------------------- caching and the transport

def test_cache_depends_on_what_is_sent(tmp_path) -> None:
    d, video = make_long(tmp_path, 400, DROPS)
    art, _ = run(d, video)
    assert write_gemini_edit(d, art) and cached_gemini_edit(d, CFG)["cache_key"] == art["cache_key"]
    assert cached_gemini_edit(d, replace(CFG, max_analysed_s=60.0)) is None       # different clips -> different request
    assert cached_gemini_edit(d, replace(CFG, context_margin_s=5.0)) is None
    assert cached_gemini_edit(d, replace(CFG, timeout_s=9.0)) is not None


def test_multi_clip_request_has_labelled_parts_with_original_offsets(monkeypatch, tmp_path) -> None:
    log = fake_sdk(monkeypatch)
    t = GenaiTransport("key", CFG, sleep=lambda s: None)
    t.generate(tmp_path / "v.mp4", "sys", "PROMPT", {"type": "object"}, [(0.0, 5.0), (97.0, 109.0), (197.0, 213.0)])
    parts = log["contents"][0].parts
    texts = [p.text for p in parts if getattr(p, "text", None)]
    clips = [p for p in parts if getattr(p, "video_metadata", None)]
    assert len(clips) == 3 and [(c.video_metadata.start_offset, c.video_metadata.end_offset) for c in clips] == [("0s", "5s"), ("97s", "109s"), ("197s", "213s")]
    assert all(c.video_metadata.fps == 1.0 for c in clips) and log["uploads"] == 1          # one upload, three clips
    assert texts[0].startswith("CLIP 1 of 3: original video seconds 0 to 5") and "97 to 109" in texts[1] and texts[-1] == "PROMPT"
    assert parts.index(clips[1]) == parts.index(next(p for p in parts if getattr(p, "text", "").startswith("CLIP 2"))) + 1  # label right before its clip
    single = fake_sdk(monkeypatch)
    GenaiTransport("key", CFG, sleep=lambda s: None).generate(tmp_path / "v.mp4", "s", "P", {}, [(0.0, 25.0)])
    assert [getattr(p, "text", None) for p in single["contents"][0].parts] == [None, "P"]  # one clip: no label, as before


# ---------------------------------------------------------------- review fixes

def test_a_kept_gemini_analysis_is_not_attached_to_a_newer_ttcc_curve(tmp_path) -> None:
    from hookline.segment_recommendations import build_segment_recommendations
    d, video = make_long(tmp_path, 60, {(0, 1): 0.25, (30, 36): 0.1})
    art, _ = run(d, video)
    assert write_gemini_edit(d, art) and art["ttcc_cache_key"] == "k"
    recs = build_segment_recommendations(d)
    assert recs["gemini"]["used"] is True and recs["gemini"]["stale"] is False and any(s["gemini"] for s in recs["segments"])
    ttcc = json.loads((d / "ttcc_retention.json").read_text())
    (d / "ttcc_retention.json").write_text(json.dumps(ttcc | {"cache_key": "a-newer-run"}))       # TTCC was rerun: segments may be renumbered
    stale = build_segment_recommendations(d)
    assert stale["gemini"]["used"] is False and stale["gemini"]["stale"] is True and "different TTCC curve" in stale["gemini"]["reason"]
    assert all(s["gemini"] is None and not any(p["source"] == "gemini" for p in s["proposals"]) for s in stale["segments"])


def test_recommendations_say_which_segments_gemini_never_saw(tmp_path) -> None:
    from hookline.segment_recommendations import build_segment_recommendations
    d, video = make_long(tmp_path, 400, DROPS)
    art, _ = run(d, video, replace(CFG, max_analysed_s=80.0))
    assert write_gemini_edit(d, art)
    recs = build_segment_recommendations(d)
    scope = recs["gemini"]["scope"]
    assert scope["whole_predicted_span"] is False and scope["not_analysed"] and scope["max_analysed_s"] == 80.0
    skipped = {x["id"] for x in scope["not_analysed"]}
    for s in recs["segments"]:
        assert s["gemini_scope"]["analysed"] is (s["id"] not in skipped)
        if s["id"] in skipped:
            assert s["gemini"] is None and s["gemini_scope"]["reason"]                              # 'not seen', not 'Gemini found nothing'


def test_a_truncated_transcript_is_flagged_and_the_budget_scales_with_the_clips(tmp_path) -> None:
    d, video = make_long(tmp_path, 107, {(0, 1): 0.25, (95, 103): 0.1})
    rows = "".join(f"long,{i},{i},{i * 0.2:.2f},{i * 0.2 + 0.15:.2f},w{i},0.9,0,\n" for i in range(530))      # 5 words/s for the whole video
    (d / "words.csv").write_text("video_id,word_index,segment_index,start_s,end_s,word,probability,asr_segment_id,asr_flags\n" + rows)
    pack = build_evidence_pack(d, CFG)
    cut = pack["evidence_truncated"]["words"]
    assert cut["total"] == 530 and cut["kept"] == 428 and len(pack["words"]) == 428 and cut["last_kept_s"] < 107   # 4 words per analysed second
    art, t = run(d, video)
    assert "evidence_truncated" in t.calls[0]["prompt"] and "not included" in t.calls[0]["prompt"]
    short, _ = make_long(tmp_path, 40, {(0, 1): 0.25}, name="short")
    assert "evidence_truncated" not in build_evidence_pack(short, CFG)                                    # nothing cut, nothing claimed


def test_an_error_message_that_straddles_the_cut_does_not_leak_the_key(tmp_path) -> None:
    d, video = make_long(tmp_path, 40, {(0, 1): 0.25})
    key = ENV["GEMINI_API_KEY"]
    art, _ = run(d, video, transport=FakeTransport(exc=RuntimeError("x" * 285 + f"?key={key} tail")))
    assert art["status"] == "failed" and "AIza" not in art["reason"] and key[:8] not in art["reason"] and "***" in art["reason"]
