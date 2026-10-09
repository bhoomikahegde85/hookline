"""Gemini interpretation with fake transports: validation, permissions, credentials, retries, caching keys.

No test here may touch the network: the ``no_network`` fixture makes any socket connection fail.
"""

from __future__ import annotations

import copy
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import gemini as gm
from hookline.config import GeminiConfig
from hookline.context import MediaContext


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def ctx(path: str = "/tmp/clip.mp4", **kw) -> MediaContext:
    segments = [{"segment_index": i, "start_s": i * 0.5, "end_s": (i + 1) * 0.5, "scene_cut_at_start": i == 4}
                for i in range(16)]
    base = dict(video_id="clip", source_path=path, source_sha256="abc", prep_key="p", duration=8.0,
                visual_stream="available", audio_stream="available", segments=segments)
    return MediaContext(**(base | kw))


def ev(s, e, **kw):
    return {"start_s": s, "end_s": e, "kind": "observation", "modality": "visual", "description": "a hand enters the frame",
            "confidence": 0.8} | kw


GOOD = {
    "video_summary": "A product demo on a desk.",
    "opening_hook": {"assessment": "Starts on a static wide shot.", "evidence": [ev(0, 2)]},
    "narrative_progression": "Setup, demo, result.",
    "pacing": "Slow first half.",
    "moments": [ev(2.0, 2.5, category="visual_change"), ev(0.0, 1.0, category="hook", kind="interpretation"),
                ev(5.0, 6.0, category="music", modality="music")],
    "strengths": [ev(5, 6, modality="audio")],
    "weaknesses": [ev(0, 2, kind="interpretation")],
    "edit_suggestions": [{"start_s": 0.0, "end_s": 2.0, "action": "Cut the first second.", "rationale": "Nothing moves.",
                          "expected_effect": "Reaches the product sooner.", "confidence": 0.6,
                          "evidence_moment_indices": [1, 0]}],
    "uncertainties": ["Audio is quiet."],
}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def analyse(self, video_path, prompt, schema, cfg):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return (json.dumps(r) if isinstance(r, dict) else r), {"prompt_token_count": 900, "candidates_token_count": 300,
                                                               "thoughts_token_count": None, "total_token_count": 1200}


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


REMOTE = GeminiConfig(mode="remote", allow_remote=True)
KEY_ENV = {"GEMINI_API_KEY": "sk-test-secret-123"}


def run(cfg, responses, c=None, env=KEY_ENV):
    t = FakeTransport(responses)
    sleeps = []
    art = gm.extract_gemini(c or ctx(), cfg, lambda: t, sleep=sleeps.append, env=env)
    return art, t, sleeps


# ------------------------------------------------------------- validation

def test_valid_response_gets_ids_links_and_sorted_moments() -> None:
    art, t, _ = run(REMOTE, [GOOD])
    assert art["status"] == "available" and art["measured"] and art["source"] == "gemini_api" and t.calls == 1
    a = art["analysis"]
    assert [(m["id"], m["start_s"]) for m in a["moments"]] == [("m1", 0.0), ("m0", 2.0), ("m2", 5.0)]
    assert a["edit_suggestions"][0]["id"] == "e0" and a["edit_suggestions"][0]["evidence_moment_ids"] == ["m1", "m0"]
    assert a["opening_hook"]["evidence"][0]["id"] == "h0"
    assert art["usage"]["total_token_count"] == 1200 and art["api_calls"] == 1 and art["cost"]["usd"] is None
    assert "retention" in art["note"]


def test_invalid_timestamps_are_dropped_not_trusted() -> None:
    bad = copy.deepcopy(GOOD)
    bad["moments"] += [ev(-1, 2, category="pacing"), ev(3, 2, category="pacing"), ev(7.9, 30, category="pacing"),
                       ev("1", 2, category="pacing"), ev(9.0, 9.5, category="pacing"), ev(float("nan"), 1, category="pacing")]
    bad["moments"].append(ev(7.5, 8.3, category="pacing"))  # within tolerance: clamped to 8.0
    bad["moments"].append(ev(4.0, 4.0, category="pacing"))  # point: widened
    bad["edit_suggestions"][0]["evidence_moment_indices"] = [0, 3, 99, True]  # 3 = dropped moment
    art, _, _ = run(REMOTE, [bad])
    a = art["analysis"]
    assert art["status"] == "uncertain" and art["validation"]["dropped"] == 6
    spans = sorted((m["start_s"], m["end_s"]) for m in a["moments"])
    assert (7.5, 8.0) in spans and (4.0, 4.5) in spans and all(0 <= s < e <= 8.0 for s, e in spans)
    assert a["edit_suggestions"][0]["evidence_moment_ids"] == ["m0"]
    assert sum("does not refer to a kept moment" in i for i in art["validation"]["issues"]) == 3


def test_unsupported_values_and_quotes() -> None:
    bad = copy.deepcopy(GOOD)
    bad["moments"] = [ev(0, 1, category="virality"), ev(0, 1, category="hook", kind="fact"),
                      ev(0, 1, category="hook", confidence=1.7), ev(0, 1, category="hook", modality="smell"),
                      ev(0, 1, category="hook", description=""),
                      ev(0, 1, category="speech", modality="speech", description='she says "this changed my whole life today"')]
    bad["retention_curve"] = [0.9, 0.5]
    art, _, _ = run(REMOTE, [bad])
    a = art["analysis"]
    assert len(a["moments"]) == 1 and a["moments"][0]["possible_quote"] is True
    issues = " ".join(art["validation"]["issues"])
    assert "ignored unexpected fields: ['retention_curve']" in issues and "quoted passage" in issues
    assert "retention_curve" not in json.dumps(a)


@pytest.mark.parametrize("payload,reason", [
    ("not json at all", "not valid JSON"),
    ('["a list"]', "not a JSON object"),
    (json.dumps({k: v for k, v in GOOD.items() if k != "moments"}), "missing required fields"),
    (json.dumps(GOOD | {"moments": "m"}), "moments is not a list"),
    (json.dumps(GOOD | {"opening_hook": {"assessment": 3}}), "opening_hook is malformed"),
])
def test_malformed_output_fails_cleanly(payload, reason) -> None:
    art, _, _ = run(REMOTE, [payload])
    assert art["status"] == "failed" and not art["measured"] and art["analysis"] is None and reason in art["reason"]


# ------------------------------------------------------------- permissions and credentials

def test_off_offline_and_remote_permissions(tmp_path) -> None:
    assert gm.extract_gemini(ctx(), GeminiConfig(), lambda: pytest.fail("transport created")) is None  # off
    art, t, _ = run(GeminiConfig(mode="remote"), [GOOD])
    assert art["status"] == "failed" and "--gemini-allow-remote" in art["reason"] and t.calls == 0
    company = str(gm.PROJECT_ROOT / "01-Hookline" / "face" / "videos" / "FC003.mp4")
    art, t, _ = run(REMOTE, [GOOD], ctx(company, video_id="FC003"))
    assert art["status"] == "failed" and "--gemini-approve FC003" in art["reason"] and t.calls == 0
    art, t, _ = run(replace(REMOTE, approved_ids=("FC003",)), [GOOD], ctx(company, video_id="FC003"))
    assert art["status"] == "available" and t.calls == 1


def test_missing_credentials_and_key_never_leaks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gm, "PROJECT_ROOT", tmp_path)  # no .env here
    art, t, _ = run(REMOTE, [GOOD], env={})
    assert art["status"] == "failed" and "GEMINI_API_KEY is not set" in art["reason"] and t.calls == 0
    (tmp_path / ".env").write_text("# comment\nexport GEMINI_API_KEY='from-dotenv'\nGEMINI_MODEL=x\n")
    assert gm.load_api_key({}) == "from-dotenv"
    assert gm.load_api_key({"GEMINI_API_KEY": " env-wins "}) == "env-wins"
    leaky = ApiError(400)
    leaky.args = ("bad request for key sk-test-secret-123",)
    art, _, _ = run(REMOTE, [leaky])
    assert "sk-test-secret-123" not in json.dumps(art) and "***" in art["reason"]
    art, _, _ = run(REMOTE, [GOOD])
    assert "sk-test-secret-123" not in json.dumps(art)


def test_offline_mode_uses_fixture_and_never_imports_the_sdk(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)  # importing it would raise
    fixture = tmp_path / "resp.json"
    fixture.write_text(json.dumps(GOOD))
    art = gm.extract_gemini(ctx(), GeminiConfig(mode="offline", offline_response=str(fixture)), env={})
    assert art["status"] == "available" and art["source"] == "offline_fixture" and art["api_calls"] == 0
    assert art["model"] is None  # offline output is never attributed to a model
    empty = gm.extract_gemini(ctx(), GeminiConfig(mode="offline"), env={})
    assert empty["status"] == "available" and empty["analysis"]["moments"] == []


def test_sdk_missing_in_remote_mode_is_a_clear_failure(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google", None)
    art = gm.extract_gemini(ctx(), REMOTE, env=KEY_ENV, sleep=lambda s: None)
    assert art["status"] == "failed" and "google-genai is not installed" in art["reason"] and art["api_calls"] == 0


def test_unreadable_or_streamless_video_is_not_sent() -> None:
    art, t, _ = run(REMOTE, [GOOD], ctx(visual_stream="uncertain", audio_stream="uncertain"))
    assert art["status"] == "uncertain" and t.calls == 0
    art, t, _ = run(REMOTE, [GOOD], ctx(visual_stream="absent", audio_stream="absent"))
    assert art["status"] == "absent" and t.calls == 0


# ------------------------------------------------------------- API failures

def test_rate_limits_and_timeouts_are_retried_with_backoff() -> None:
    art, t, sleeps = run(replace(REMOTE, max_retries=2), [ApiError(429), TimeoutError("slow"), GOOD])
    assert art["status"] == "available" and t.calls == 3 and art["api_calls"] == 3 and sleeps == [2.0, 4.0]
    art, t, sleeps = run(replace(REMOTE, max_retries=1), [ApiError(503), ApiError(503)])
    assert art["status"] == "failed" and "(503)" in art["reason"] and t.calls == 2


def test_client_errors_are_not_retried() -> None:
    art, t, sleeps = run(replace(REMOTE, max_retries=3), [ApiError(400)])
    assert art["status"] == "failed" and t.calls == 1 and sleeps == []


# ------------------------------------------------------------- caching key and timeline columns

def test_cache_key_covers_source_model_prompt_and_config(monkeypatch, tmp_path) -> None:
    c = ctx()
    key = gm.gemini_key(c, REMOTE)
    assert gm.gemini_key(c, replace(REMOTE, timeout_s=5, max_retries=0, allow_remote=False)) == key  # not content
    for changed in (replace(REMOTE, model="other-model"), replace(REMOTE, fps=2.0), replace(REMOTE, media_resolution="high"),
                    replace(REMOTE, mode="offline")):
        assert gm.gemini_key(c, changed) != key
    assert gm.gemini_key(replace(c, source_sha256="different"), REMOTE) != key
    import hookline.config as cfgmod
    monkeypatch.setattr(cfgmod, "GEMINI_PROMPT_VERSION", "999")
    assert gm.gemini_key(c, REMOTE) != key
    fixture = tmp_path / "r.json"
    fixture.write_text("{}")
    offline = GeminiConfig(mode="offline", offline_response=str(fixture))
    k1 = gm.gemini_key(c, offline)
    fixture.write_text('{"changed": 1}')
    assert gm.gemini_key(c, offline) != k1


def test_gemini_window_links_overlapping_items_only() -> None:
    art, _, _ = run(REMOTE, [GOOD])
    w, notes = gm.gemini_window(art, {"start_s": 0.5, "end_s": 1.0})
    assert w == {"gemini_status": "available", "gemini_moment_ids": ["m1"], "gemini_suggestion_ids": ["e0"]} and notes == []
    w, _ = gm.gemini_window(art, {"start_s": 3.0, "end_s": 3.5})
    assert w["gemini_moment_ids"] is None and w["gemini_suggestion_ids"] is None
    w, notes = gm.gemini_window(None, {"start_s": 0.0, "end_s": 0.5})
    assert w["gemini_status"] == "uncertain" and "disabled" in notes[0]
    failed, _, _ = run(REMOTE, ["nope"])
    w, _ = gm.gemini_window(failed, {"start_s": 0.0, "end_s": 0.5})
    assert w == {"gemini_status": "failed", "gemini_moment_ids": None, "gemini_suggestion_ids": None}


def test_schema_sent_to_the_model_matches_local_validator() -> None:
    schema = gm.RESPONSE_SCHEMA
    assert set(schema["required"]) == set(schema["properties"])
    moment = schema["properties"]["moments"]["items"]
    assert moment["properties"]["category"]["enum"] == gm.CATEGORIES and moment["properties"]["kind"]["enum"] == gm.KINDS
    assert "retention" in gm.PROMPT and "Do not transcribe" in gm.PROMPT


def test_config_validation() -> None:
    for bad in (dict(mode="on"), dict(media_resolution="ultra"), dict(fps=0), dict(max_retries=-1)):
        with pytest.raises(ValueError):
            GeminiConfig(**bad)
    assert Path(gm.PROJECT_ROOT, ".gitignore").read_text().count(".env") >= 1
