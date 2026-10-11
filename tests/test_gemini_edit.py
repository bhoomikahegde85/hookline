"""Gemini segment interpretation: evidence pack, strict validation, permission gate, transport, caching, the combined recommendations and the
page. A fake transport / fake SDK stands in for Google: no network, no key, no cost."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import types
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import gemini_edit as ge
from hookline.config import GeminiConfig
from hookline.dropoffs import write_dropoffs
from hookline.gemini_edit import (
    ARTIFACT, GeminiUnavailable, GenaiTransport, build_evidence_pack, cached_gemini_edit, extract_gemini_edit, remote_block_reason, validate,
    write_gemini_edit,
)
from hookline.review_page import build_review
from hookline.segment_recommendations import build_segment_recommendations, write_segment_recommendations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURVE = [1.0, 0.5, 0.3, 0.285, 0.27, 0.256, 0.243, 0.243, 0.2425, 0.2422, 0.2420]
CFG = GeminiConfig(mode="remote", allow_remote=True)
ENV = {"GEMINI_API_KEY": "AIza-test-key-not-real-123"}
NO_DOTENV = Path("/nonexistent/.env")


def make_dir(tmp_path: Path, name: str = "v") -> tuple[Path, Path]:
    d = tmp_path / "features" / name
    d.mkdir(parents=True)
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"VIDEO" * 100)
    import hashlib
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    (d / "ttcc_retention.json").write_text(json.dumps({
        "status": "success", "video_id": name, "source_sha256": sha, "source_path": str(video), "cache_key": "k", "horizon_s": len(CURVE) - 1,
        "duration_s": 10.0, "partial_coverage": False, "model": {"checkpoint_repo": "m"}, "timestamps_s": list(range(len(CURVE))), "retention": CURVE,
        "inference": {"deviations_from_upstream_eval": []}}))
    (d / "speech.json").write_text(json.dumps({"measured": True, "status": "available", "language": "hi"}))
    (d / "words.csv").write_text("video_id,word_index,segment_index,start_s,end_s,word,probability,asr_segment_id,asr_flags\n"
                                 "v,0,0,0.2,0.5,hello,0.9,0,\nv,1,1,5.0,5.4,later,0.9,0,\nv,2,2,40.0,41.0,beyond,0.9,0,\n")
    (d / "ocr.json").write_text(json.dumps({"measured": True, "events": [{"event_id": 0, "text": "SALE", "first_seen_s": 1.0, "visible_until_s": 2.0}]}))
    (d / "librosa.json").write_text(json.dumps({"measured": True, "low_energy_intervals": [{"interval_id": 0, "start_s": 6.0, "end_s": 8.0}]}))
    (d / "report.json").write_text(json.dumps({"scene_cut_times": [1.2, 7.5], "segments": [{"start_s": 0.0, "end_s": 10.0, "visual_status": "available", "motion_mean": 3.0}]}))
    (d / "recommendations.json").write_text(json.dumps({"recommendations": [
        {"rule": "static_stretch", "start_s": 0.0, "end_s": 2.0, "suggestion": "Consider a visual change.", "evidence": [{"type": "segment", "id": 0}]}]}))
    write_dropoffs(d)
    return d, video


def good_answer(pack: dict) -> dict:
    first = pack["ttcc"]["segments"][0]
    return {"segments": [{"segment_id": first["id"], "summary": "Opening with a spoken greeting.",
                          "observations": [{"kind": "observation", "text": "The word 'hello' is spoken at 0.2 s.", "evidence_refs": ["w0", first["id"]]},
                                           {"kind": "interpretation", "text": "The opening may feel slow.", "evidence_refs": []}],
                          "hypothesis": "Viewers may leave because nothing changes visually in the first seconds.", "hypothesis_confidence": 0.6,
                          "suggestions": [{"category": "remove_dead_space", "action": "Trim 0.0 to 1.0 s of lead-in before the greeting.",
                                           "edit": {"type": "trim", "start_s": 0.0, "end_s": 1.0}, "rationale": "Only one word in this stretch (w0).",
                                           "expected_effect": "May reach the content sooner.", "confidence": 0.5, "evidence_refs": ["w0"]}]}],
            "overall": {"hook_assessment": "The first 3 s contain a greeting and on-screen text.", "top_priorities": [first["id"]]}}


class FakeTransport:
    def __init__(self, answer=None, text=None, exc=None):
        self.answer, self.text, self.exc = answer, text, exc
        self.calls, self.closed, self.requests = [], False, [{"attempt": 1, "ok": True}]

    def generate(self, video_path, system, prompt, schema, ranges):
        self.calls.append({"video": video_path, "system": system, "prompt": prompt, "schema": schema, "ranges": ranges})
        if self.exc:
            raise self.exc
        return (self.text if self.text is not None else json.dumps(self.answer)), {"total_token_count": 1234}

    def close(self):
        self.closed = True


def run(d, video, cfg=CFG, transport=None, env=ENV, **kw):
    t = transport or FakeTransport(good_answer(build_evidence_pack(d)))
    made = []

    def factory(key):
        made.append(key)
        return t
    return extract_gemini_edit(d, video, cfg, factory, env=env, dotenv=NO_DOTENV, root=kw.get("root", PROJECT_ROOT)), t, made


# ---------------------------------------------------------------- evidence pack and prompt

def test_pack_has_citable_ids_and_stays_inside_the_predicted_span(tmp_path) -> None:
    d, _ = make_dir(tmp_path)
    pack = build_evidence_pack(d)
    ids = set(pack["_ids"])
    assert {"s0", "w0", "w1", "t0", "c0", "c1", "q0", "r0", "ttcc:t=0", "ttcc:t=10"} <= ids
    assert "w2" not in ids and all(w["t0"] < 10 for w in pack["words"])  # the word at 40 s is outside the 10 s span
    assert pack["predicted_span_s"] == [0.0, 10.0] and pack["ttcc"]["segments"][0]["rank_by_predicted_fall"] == 1
    assert "PREDICTION" in pack["ttcc"]["note"] and "_ids" not in ge.pack_for_prompt(pack)
    (d / "speech.json").unlink()
    write_dropoffs(d)
    assert build_evidence_pack(d)["words"] == "not run"  # a missing source is reported, not emptied
    assert build_evidence_pack(tmp_path / "nothing") is None


def test_prompt_and_schema_frame_gemini_as_explainer_with_hypotheses(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    out, t, _ = run(d, video)
    call = t.calls[0]
    assert "PREDICTS" in call["system"] and "hypothesis" in call["system"] and "validated" in call["system"]
    assert "0 to 10" in call["prompt"] and "hello" in call["prompt"] and "_ids" not in call["prompt"] and call["ranges"] == [(0.0, 10.0)]
    props = call["schema"]["properties"]["segments"]["items"]["properties"]["suggestions"]["items"]["properties"]
    assert set(props["category"]["enum"]) == set(ge.CATEGORIES) and set(props["edit"]["properties"]["type"]["enum"]) == {"trim", "add", "review"}


# ---------------------------------------------------------------- validation

def test_valid_answer_passes_and_is_labelled_a_hypothesis(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    art, t, _ = run(d, video)
    assert art["status"] == "available" and art["measured"] and art["sent"] and t.closed
    seg = art["analysis"]["segments"][0]
    assert seg["suggestions"][0]["status"].startswith("hypothesis") and seg["suggestions"][0]["edit"] == {"type": "trim", "start_s": 0.0, "end_s": 1.0}
    assert art["validation"]["dropped"] == 0 and art["validation"]["issues"] == ["no usable entry for segment(s) ['s1']"]  # said out loud
    assert art["usage"] == {"total_token_count": 1234}
    assert ENV["GEMINI_API_KEY"] not in json.dumps(art) and "validated" in art["note"]


def test_validation_drops_what_is_not_grounded(tmp_path) -> None:
    d, _ = make_dir(tmp_path)
    pack = build_evidence_pack(d)
    sug = lambda **kw: {"category": "adjust_pacing", "action": "Do x", "edit": {"type": "review", "start_s": 0.5, "end_s": 1.5}, "rationale": "r",  # noqa: E731
                        "expected_effect": "e", "confidence": 0.4, "evidence_refs": ["w0"]} | kw
    raw = {"segments": [
        {"segment_id": "ghost", "summary": "x", "observations": [], "hypothesis": "h", "hypothesis_confidence": 0.5, "suggestions": []},
        {"segment_id": "s0", "summary": "s", "hypothesis": "h", "hypothesis_confidence": 7.0, "observations": [
            {"kind": "observation", "text": "seen", "evidence_refs": ["w999"]},           # only an invented id: dropped
            {"kind": "interpretation", "text": "reading", "evidence_refs": ["w999"]}],   # kept, the bad ref removed
         "suggestions": [
             sug(),                                                                      # fine
             sug(evidence_refs=["w999"]),                                               # no valid evidence
             sug(edit={"type": "trim", "start_s": 0.0, "end_s": 50.0}),                 # outside the predicted span
             sug(edit={"type": "trim", "start_s": 1.0, "end_s": 1.1}),                  # shorter than 0.2 s
             sug(edit={"type": "trim", "start_s": 8.0, "end_s": 9.0}),                  # does not overlap the segment
             sug(edit={"type": "add", "start_s": 9.0, "end_s": 9.0}),                   # insert point outside the segment
             sug(category="make_it_viral"),                                              # unknown category
             sug(action="   ")]},                                                        # empty action
        {"segment_id": "s0", "summary": "dup", "observations": [], "hypothesis": "h", "hypothesis_confidence": 0.1, "suggestions": []}]}
    clean, v = validate(raw, pack, CFG)
    seg = clean["segments"][0]
    assert [s["segment_id"] for s in clean["segments"]] == ["s0"]  # ghost and duplicate dropped
    assert len(seg["suggestions"]) == 1 and seg["hypothesis_confidence"] == 1.0  # clamped
    assert [o["kind"] for o in seg["observations"]] == ["interpretation"] and seg["observations"][0]["evidence_refs"] == []
    assert v["dropped"] >= 9 and any("unknown segment_id" in i for i in v["issues"]) and any("outside the predicted span" in i for i in v["issues"])
    assert any("no usable entry" in i for i in v["issues"])  # the other segments got nothing: said out loud
    capped, v2 = validate({"segments": [{"segment_id": "s0", "summary": "", "observations": [], "hypothesis": "", "hypothesis_confidence": None,
                                         "suggestions": [sug() for _ in range(5)]}]}, pack, replace(CFG, max_suggestions_per_segment=2))
    assert len(capped["segments"][0]["suggestions"]) == 2 and any("beyond the cap" in i for i in v2["issues"])
    assert validate("nope", pack, CFG)[0] == {"segments": [], "overall": None, "audio": None}


def test_audio_read_is_kept_only_when_well_formed_and_inside_the_clips() -> None:
    from hookline.gemini_edit import _clean_audio
    good = {"kind": "song", "description": "A pop song under a spoken line.", "energy": "high", "familiarity": "May be a generic upbeat track.",
            "changes": [{"at_s": 2.0, "what": "music drops"}, {"at_s": 99.0, "what": "outside"}, {"at_s": 3.0, "what": ""}]}
    a = _clean_audio(good, [[0.0, 10.0]])
    assert a["kind"] == "song" and a["energy"] == "high" and a["changes"] == [{"at_s": 2.0, "what": "music drops"}] and "not a statement that any sound is trending" in a["status"]
    assert _clean_audio(good | {"fit": {"verdict": "clashes", "reason": "calm visuals, loud techno"}}, [[0.0, 10.0]])["fit"]["verdict"] == "clashes"
    assert _clean_audio(good | {"fit": {"verdict": "great"}}, [[0.0, 10.0]])["fit"] is None
    assert _clean_audio({"kind": "banger"}, [[0.0, 10.0]]) is None and _clean_audio("x", [[0.0, 10.0]]) is None
    assert _clean_audio(good | {"energy": "extreme"}, [[0.0, 10.0]])["energy"] is None


# ---------------------------------------------------------------- gate, failures, key handling

def test_nothing_is_sent_without_permission_key_or_a_matching_video(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    art, t, made = run(d, video, cfg=replace(CFG, allow_remote=False))
    assert not art["sent"] and "--gemini-allow-remote" in art["reason"] and made == []
    art, _, made = run(d, video, env={})
    assert not art["sent"] and "GEMINI_API_KEY is not set" in art["reason"] and made == []
    art, _, made = run(d, tmp_path / "gone.mp4")
    assert "video not found" in art["reason"] and made == []
    video.write_bytes(b"changed")
    art, _, made = run(d, video)
    assert "changed since its TTCC prediction" in art["reason"] and made == []
    art, _, made = run(make_dir(tmp_path, "big")[0], tmp_path / "big.mp4", cfg=replace(CFG, max_upload_mb=0.0001))
    assert "exceeds max_upload_mb" in art["reason"] and made == []
    assert extract_gemini_edit(d, video, GeminiConfig(), env=ENV) is None  # disabled: nothing at all


def test_company_reels_need_explicit_approval(tmp_path) -> None:
    company = tmp_path / "01-Hookline" / "face" / "videos" / "FC002.mp4"
    company.parent.mkdir(parents=True)
    company.write_bytes(b"x")
    assert "--gemini-approve FC002" in remote_block_reason("FC002", company, CFG, tmp_path)
    assert remote_block_reason("FC002", company, replace(CFG, approved_ids=("FC002",)), tmp_path) is None
    assert remote_block_reason("mine", tmp_path / "mine.mp4", CFG, tmp_path) is None
    assert "off" in remote_block_reason("x", company, GeminiConfig(), tmp_path)


def test_failures_become_reasons_and_never_leak_the_key(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    art, t, _ = run(d, video, transport=FakeTransport(exc=RuntimeError(f"401 bad key {ENV['GEMINI_API_KEY']}")))
    assert art["status"] == "failed" and art["sent"] and ENV["GEMINI_API_KEY"] not in art["reason"] and "***" in art["reason"] and t.closed
    art, t, _ = run(d, video, transport=FakeTransport(text="this is not json"))
    assert art["status"] == "failed" and "not valid JSON" in art["reason"] and t.closed
    art, _, _ = run(d, video, transport=FakeTransport(answer={"segments": [{"segment_id": "s0", "suggestions": [{"category": "x"}], "observations": []}]}))
    assert art["status"] == "failed" and "nothing that passed validation" in art["reason"]
    art = extract_gemini_edit(d, video, CFG, lambda k: (_ for _ in ()).throw(GeminiUnavailable("google-genai is not installed")), env=ENV, dotenv=NO_DOTENV)
    assert not art["sent"] and "not installed" in art["reason"]
    art = extract_gemini_edit(tmp_path / "empty", video, CFG, env=ENV, dotenv=NO_DOTENV)
    assert "no TTCC drop-off analysis" in art["reason"]


def test_partly_valid_answers_are_marked_uncertain(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    answer = good_answer(build_evidence_pack(d))
    answer["segments"][0]["suggestions"].append(dict(answer["segments"][0]["suggestions"][0], evidence_refs=["w999"]))
    art, _, _ = run(d, video, transport=FakeTransport(answer))
    assert art["status"] == "uncertain" and art["measured"] and art["validation"]["dropped"] == 1


# ---------------------------------------------------------------- cache and artifact handling

def test_cache_matches_only_the_same_video_settings_and_evidence(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    art, _, _ = run(d, video)
    assert write_gemini_edit(d, art) and cached_gemini_edit(d, CFG)["cache_key"] == art["cache_key"]
    assert cached_gemini_edit(d, replace(CFG, model="another-model")) is None  # model change
    assert cached_gemini_edit(d, replace(CFG, fps=2.0)) is None  # sampling change
    assert cached_gemini_edit(d, replace(CFG, timeout_s=5.0)) is not None  # transport settings are not in the key
    (d / "ocr.json").write_text(json.dumps({"measured": True, "events": [{"event_id": 0, "text": "NEW", "first_seen_s": 1.0, "visible_until_s": 2.0}]}))
    assert cached_gemini_edit(d, CFG) is None  # different evidence


def test_a_failure_never_replaces_an_earlier_success(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    good, _, _ = run(d, video)
    write_gemini_edit(d, good)
    bad, _, _ = run(d, video, transport=FakeTransport(exc=RuntimeError("boom")))
    assert write_gemini_edit(d, bad) is False
    assert json.loads((d / ARTIFACT).read_text())["measured"] is True


# ---------------------------------------------------------------- the SDK glue (fake google-genai)

def fake_sdk(monkeypatch, states=("PROCESSING", "ACTIVE"), fail_times=0, fail_code=503):
    ns = lambda name: type(name, (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})  # noqa: E731
    types_stub = types.SimpleNamespace(**{n: ns(n) for n in ("HttpOptions", "UploadFileConfig", "Part", "FileData", "VideoMetadata", "Content", "GenerateContentConfig")})
    log = {"deleted": [], "generate": 0, "uploads": 0, "polls": 0}

    class Files:
        def upload(self, file, config):
            log["uploads"] += 1
            return types.SimpleNamespace(name="files/abc", uri="https://u/abc", mime_type="video/mp4", state=states[0])

        def get(self, name):
            i = log["polls"]
            log["polls"] += 1
            return types.SimpleNamespace(name=name, uri="https://u/abc", mime_type="video/mp4", state=states[min(i, len(states) - 1)])

        def delete(self, name):
            log["deleted"].append(name)

    class Models:
        def generate_content(self, model, contents, config):
            log["generate"] += 1
            if log["generate"] <= fail_times:
                err = RuntimeError("busy")
                err.code = fail_code
                raise err
            log["config"], log["contents"], log["model"] = config, contents, model
            return types.SimpleNamespace(text='{"ok": 1}', usage_metadata=types.SimpleNamespace(prompt_token_count=10, candidates_token_count=5,
                                                                                               thoughts_token_count=2, total_token_count=17))

    client = types.SimpleNamespace(files=Files(), models=Models())
    monkeypatch.setattr(ge, "_sdk", lambda: (types.SimpleNamespace(Client=lambda **kw: client), types_stub))
    return log


def test_genai_transport_uploads_waits_retries_and_deletes(monkeypatch, tmp_path) -> None:
    log = fake_sdk(monkeypatch, fail_times=2)
    sleeps = []
    t = GenaiTransport("key", replace(CFG, max_retries=3), sleep=sleeps.append)
    text, usage = t.generate(tmp_path / "v.mp4", "sys", "prompt", {"type": "object"}, 25.0)
    assert text == '{"ok": 1}' and usage["total_token_count"] == 17 and log["uploads"] == 1 and log["generate"] == 3
    assert sleeps.count(2.0) >= 2 and 4.0 in sleeps  # waited for ACTIVE, then backed off 2 s, 4 s
    part = log["contents"][0].parts[0]
    assert part.video_metadata.end_offset == "25s" and part.video_metadata.start_offset == "0s" and part.video_metadata.fps == 1.0
    assert log["config"].response_mime_type == "application/json" and log["config"].media_resolution == "MEDIA_RESOLUTION_MEDIUM"
    assert [r["ok"] for r in t.requests] == [False, False, True]
    t.close()
    assert log["deleted"] == ["files/abc"]  # the upload does not stay on Google's side


def test_genai_transport_does_not_retry_client_errors_or_failed_uploads(monkeypatch, tmp_path) -> None:
    log = fake_sdk(monkeypatch, fail_times=5, fail_code=400)
    t = GenaiTransport("key", CFG, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="busy"):
        t.generate(tmp_path / "v.mp4", "s", "p", {}, 5.0)
    assert log["generate"] == 1  # a 400 is not retried
    fake_sdk(monkeypatch, states=("FAILED",))
    with pytest.raises(GeminiUnavailable, match="FAILED"):
        GenaiTransport("key", CFG, sleep=lambda s: None).generate(tmp_path / "v.mp4", "s", "p", {}, 5.0)


def test_missing_sdk_is_a_clear_unavailable_error(monkeypatch) -> None:
    import builtins
    real = builtins.__import__

    def no_genai(name, *a, **k):
        if name.startswith("google"):
            raise ImportError("no google")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_genai)
    with pytest.raises(GeminiUnavailable, match="google-genai is not installed"):
        ge._sdk()


# ---------------------------------------------------------------- combined recommendations and the page

def test_combined_recommendations_keep_the_two_models_apart(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    plain = build_segment_recommendations(d)
    assert plain["gemini"]["used"] is False and all(p["source"] == "rule" for s in plain["segments"] for p in s["proposals"])
    art, _, _ = run(d, video)
    write_gemini_edit(d, art)
    both = build_segment_recommendations(d)
    s0 = both["segments"][0]
    assert both["gemini"]["used"] and "validated" in both["status"] and s0["gemini"]["hypothesis"].startswith("Viewers may")
    sources = [p["source"] for p in s0["proposals"]]
    assert "rule" in sources and "gemini" in sources
    g = next(p for p in s0["proposals"] if p["source"] == "gemini")
    assert g["rule"] == "gemini:remove_dead_space" and g["edit"] == {"action": "trim", "start_s": 0.0, "end_s": 1.0} and g["evidence_refs"] == ["w0"]
    assert s0["ttcc"]["meaning"] == "TTCC model prediction, not audience data" and "score" not in s0  # nothing merged into one number
    assert write_segment_recommendations(d).name == "segment_recommendations.json"
    assert build_segment_recommendations(tmp_path / "x") is None


CHROME = next((c for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"), shutil.which("chromium"))
               if c and Path(c).exists()), None)


@pytest.mark.skipif(CHROME is None, reason="Chrome not installed")
def test_page_shows_gemini_and_its_trims_work_in_chrome(tmp_path) -> None:
    d, video = make_dir(tmp_path)
    art, _, _ = run(d, video)
    write_gemini_edit(d, art)
    page = build_review(d, video)
    html = page.read_text()
    assert ENV["GEMINI_API_KEY"] not in html and "Gemini overview" in html and "Gemini (model interpretation, hypothesis)" in html
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert data["gemini_info"]["used"] and any(c["source"] == "gemini" and c["edit"]["action"] == "trim" for iv in data["intervals"] for c in iv["candidate_causes"])
    out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000", "--dump-dom", page.resolve().as_uri() + "#selftest"],
                         capture_output=True, text=True, timeout=120).stdout
    m = re.search(r'<pre id="selftest">(.*?)</pre>', out, re.S)
    assert m and m.group(1).startswith("SELFTEST PASS") and "accepted trim reaches the export" in m.group(1), m.group(1) if m else "page script did not run"
    assert "Gemini overview" in out and "Gemini: what it observed" in out  # rendered into the DOM, not only present in the source


def test_e2e_gemini_step_is_required_and_gated(tmp_path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_e2e", PROJECT_ROOT / "scripts" / "run_e2e.py")
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    base = dict(allow_remote=True, approve="FC002", siglip=False, skip_local=False, open_page=False)
    steps = e2e.plan(tmp_path / "FC002.mp4", tmp_path, gemini_allow_remote=True, **base)
    assert len(steps) == 9  # Gemini is always part of the plan
    g = steps[3]
    assert "Gemini" in g["name"] and not g.get("optional") and "--gemini-allow-remote" in g["cmd"] and g["cmd"][g["cmd"].index("--gemini-approve") + 1] == "FC002"
    assert "run_gemini_edit.py" in g["cmd"][1] and ".venv-models/" in g["cmd"][0] and steps[4]["name"].startswith("music") and steps[5]["name"].startswith("popularity") and steps[6]["name"].startswith("scores") and steps[7]["name"].startswith("explain") and steps[8]["name"].startswith("segments")
    assert e2e.plan(tmp_path / "v.mp4", tmp_path, **base)[3]["skip"]  # no upload permission, no step
    calls = []

    def run(cmd, cwd=None, capture_output=True, text=True):
        calls.append(cmd)
        if "run_ttcc.py" in cmd[1]:
            d = tmp_path / "features" / "FC002"
            d.mkdir(parents=True, exist_ok=True)
            (d / "ttcc_status.json").write_text(json.dumps({"status": "success"}))
        return subprocess.CompletedProcess(cmd, 1 if "run_gemini_edit.py" in cmd[1] else 0, stdout="boom", stderr="")
    page = tmp_path / "features" / "FC002" / "review.html"
    page.parent.mkdir(parents=True)
    page.write_text("x")
    logs = []
    code = e2e.run_e2e(tmp_path / "FC002.mp4", tmp_path, run=run, log=logs.append, gemini_allow_remote=True, **base)
    assert code == 1 and not any("build_review" in " ".join(map(str, c)) for c in calls) and not any("continuing without it" in line for line in logs)  # Gemini failing stops the analysis: it is required
