"""TTCC retention stage: configuration, input and output validation, caching, failure isolation,
plotting and reporting. Model inference is mocked (``FakeRunner``); the real checkpoint is only
exercised by ``test_ttcc_integration.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import ttcc
from hookline.config import TTCC_CHECKPOINT_FILES, PipelineConfig, TtccConfig, env_flag
from hookline.pipeline import VideoInput, run
from hookline.ttcc import (
    DISABLED, FAILED, PLOT_FILE, RAW_FILE, RAW_REJECTED_FILE, RETENTION_FILE, STATUS_FILE, SUCCESS, UNAVAILABLE,
    SubprocessRunner, TtccOutputError, horizon, parse_raw, retention_svg, run_ttcc,
)
from hookline.video_report import REPORT_FILE, write_video_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_V8_TOOL = PROJECT_ROOT / "go_viral" / "examples/custom/qwen2_5_omni_retention/tools/make_v8_from_hf.py"
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
ENABLED = TtccConfig(enabled=True, python="/fake/python")
QUIET = dict(log=lambda *_: None)


def curve(n: int = 60, start: float = 0.95, step: float = 0.01) -> list[float]:
    return [max(start - step * i, 0.01) for i in range(n)]


PROBE_OK = {"inference_possible": True, "problems": [], "libraries": {"torch": "2.8.0", "transformers": "4.56.2"},
            "cuda": {"available": True, "devices": [{"name": "Tesla T4"}]},
            "code": {"commit": "893da03b", "dirty": False, "plugin_sha256": "p", "runner_version": "1", "runner_sha256": "r"}}


class FakeRunner:
    """Stands in for the subprocess: writes raw outputs like scripts/ttcc_infer.py does."""

    def __init__(self, probe=None, values=None, fail=(), omit=(), raw_override=None):
        self.probe_info = PROBE_OK if probe is None else probe
        self.values = curve() if values is None else values
        self.fail, self.omit, self.raw_override = set(fail), set(omit), raw_override
        self.probes = self.infers = 0
        self.requests: list[dict] = []

    def python(self, cfg):
        return cfg.python

    def probe(self, cfg):
        self.probes += 1
        return self.probe_info

    def infer(self, cfg, request, workdir):
        self.infers += 1
        self.requests.append(request)
        out = []
        for job in request["jobs"]:
            if job["video_id"] in self.omit:
                continue
            if job["video_id"] in self.fail:
                out.append({"video_id": job["video_id"], "status": "failed", "stage": "inference",
                            "error": "OutOfMemoryError: CUDA out of memory"})
                continue
            raw = {"video_id": job["video_id"], "T": job["T"], "r_pred": list(self.values), "input_tokens": 1234,
                   "anchor": "</cot>", "inference_seconds": 1.5,
                   "provenance": {"runner_version": "1", "model_load_seconds": 20.0, "device": "cuda:0"}}
            if self.raw_override:
                raw = self.raw_override(raw)
            Path(job["raw_output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["raw_output"]).write_text(json.dumps(raw))
            out.append({"video_id": job["video_id"], "status": "ok", "raw_output": job["raw_output"]})
        return out


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    if not HAS_FFMPEG:
        pytest.skip("FFmpeg not installed")
    d = tmp_path_factory.mktemp("ttcc_clips")
    v = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"]
    out = {"six": d / "six.mp4", "seven": d / "seven.mp4", "short": d / "short.mp4", "empty": d / "empty.mp4",
           "long": d / "long.mp4"}
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x112:r=2:d=62.4", "-an", *v, str(out["long"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x284:r=10:d=6", "-f", "lavfi", "-i", "sine=duration=6", *v,
           "-c:a", "aac", "-shortest", str(out["six"]))
    ffmpeg("-f", "lavfi", "-i", "color=gray:s=160x284:r=10:d=7", "-an", *v, str(out["seven"]))
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x284:r=10:d=3", "-an", *v, str(out["short"]))
    out["empty"].write_bytes(b"")
    return out


def items(*paths: Path) -> list[VideoInput]:
    return [VideoInput(p.stem, p) for p in paths]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------- configuration

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_TTCC", raising=False)
    assert TtccConfig().enabled is False
    assert TtccConfig.from_env().enabled is False


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("", False)])
def test_enable_flag_from_environment(monkeypatch, value, expected):
    monkeypatch.setenv("ENABLE_TTCC", value)
    assert TtccConfig.from_env().enabled is expected
    assert TtccConfig.from_env(enabled=True).enabled is True  # explicit override wins


def test_invalid_enable_flag_is_an_error(monkeypatch):
    monkeypatch.setenv("ENABLE_TTCC", "maybe")
    with pytest.raises(ValueError, match="ENABLE_TTCC"):
        env_flag("ENABLE_TTCC")


def test_config_rejects_other_heads_and_matches_v8_protocol():
    with pytest.raises(ValueError):
        TtccConfig(head_type="sigmoid")
    env = TtccConfig().runner_env()
    assert env == {"FPS": "1.0", "FPS_MAX_FRAMES": "60", "VIDEO_MAX_PIXELS": "49152", "MAX_PIXELS": "49152",
                   "VIDEO_MAX_TOKEN_NUM": "8192", "USE_AUDIO_IN_VIDEO": "false", "ENABLE_AUDIO_OUTPUT": "False",
                   "RETENTION_HEAD_TYPE": "hazard"}
    assert TtccConfig().checkpoint_revision == "1686d5f00408bfd8f6a32be12e71248b6f8ff447"


@needs_ffmpeg
def test_disabled_run_touches_nothing(tmp_path, clips):
    runner = FakeRunner()
    res = run_ttcc(items(clips["six"]), tmp_path, TtccConfig(enabled=False), runner, **QUIET)
    assert [r["status"] for r in res["records"]] == [DISABLED]
    assert runner.probes == runner.infers == 0
    assert not (tmp_path / "features").exists()


@pytest.mark.skipif(not UPSTREAM_V8_TOOL.is_file(), reason="go_viral checkout not present")
def test_prompts_are_verbatim_upstream():
    src = UPSTREAM_V8_TOOL.read_text()
    block = re.search(r"SYSTEM_PROMPT = \((.*?)\)\n", src, re.S).group(1)
    upstream_system = "".join(re.findall(r'"([^"]*)"', block))
    assert upstream_system == ttcc.SYSTEM_PROMPT
    upstream_user = re.search(r'return f"(This ad is \{T\} seconds long\.[^"]*)"', src).group(1)
    assert upstream_user == ttcc.USER_PROMPT_TEMPLATE


# ---------------------------------------------------------------- input validation

@pytest.mark.parametrize("duration,expected", [(5.338, 5), (4.4, None), (4.5, None), (9.467, 9), (60.4, 60),
                                               (60.6, None), (143.2, None), (None, None), (float("nan"), None)])
def test_horizon_follows_upstream_rounding_and_range(duration, expected):
    assert horizon(duration) == expected


@needs_ffmpeg
def test_validate_video(clips, tmp_path):
    info, problem = ttcc.validate_video(clips["six"])
    assert problem is None and info["duration_s"] == pytest.approx(6.0, abs=0.1)
    assert ttcc.validate_video(clips["empty"])[1].startswith("input video could not be read")
    assert ttcc.validate_video(tmp_path / "missing.mp4")[1] is not None
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a video" * 100)
    assert ttcc.validate_video(garbage)[1] is not None


@needs_ffmpeg
def test_invalid_and_out_of_range_inputs(tmp_path, clips):
    runner = FakeRunner()
    res = run_ttcc(items(clips["empty"], clips["short"]), tmp_path, ENABLED, runner, **QUIET)
    by = {r["video_id"]: r for r in res["records"]}
    assert by["empty"]["status"] == FAILED and by["empty"]["stage"] == "validation"
    assert by["short"]["status"] == UNAVAILABLE and "shorter than TTCC's trained range" in by["short"]["reason"]
    assert runner.probes == runner.infers == 0  # nothing valid to run
    assert not (tmp_path / "features" / "short" / RETENTION_FILE).exists()
    assert load(tmp_path / "features" / "short" / STATUS_FILE)["status"] == UNAVAILABLE


# ---------------------------------------------------------------- output parsing

def test_parse_raw_normalises_to_horizon():
    raw = {"T": 6, "r_pred": curve()}
    c = parse_raw(raw, 6)
    assert c["timestamps_s"] == [0, 1, 2, 3, 4, 5, 6]
    assert c["retention"][0] == 1.0 and c["retention"][1:] == curve()[:6]
    assert c["summary"]["retention_at_horizon"] == curve()[5]
    assert c["summary"]["steepest_one_second_drop"]["from_s"] == 0  # 1.0 -> 0.95 is the largest drop


@pytest.mark.parametrize("raw,match", [
    ({"T": 6}, "no r_pred"),
    ({"T": 6, "r_pred": curve(59)}, "59 values"),
    ({"T": 6, "r_pred": curve()[:-1] + [float("nan")]}, "non-finite"),
    ({"T": 6, "r_pred": curve()[:-1] + [None]}, "missing"),
    ({"T": 6, "r_pred": [True] * 60}, "missing or non-finite"),
    ({"T": 6, "r_pred": [1.2] + curve()[1:]}, r"outside \(0, 1\]"),
    ({"T": 6, "r_pred": curve()[:-1] + [0.0]}, r"outside \(0, 1\]"),
    ({"T": 6, "r_pred": curve()[:10] + [0.99] + curve()[11:]}, "increases at t=\\[11\\]"),
    ({"T": 7, "r_pred": curve()}, "expected 6"),
    ("not a dict", "not a JSON object"),
])
def test_parse_raw_rejects_invalid_output(raw, match):
    with pytest.raises(TtccOutputError, match=match):
        parse_raw(raw, 6)


@pytest.mark.parametrize("T", [4, 61, None])
def test_parse_raw_rejects_out_of_range_horizon(T):
    with pytest.raises(TtccOutputError, match="trained range"):
        parse_raw({"T": T, "r_pred": curve()}, T)


# ---------------------------------------------------------------- end-to-end with a mocked model

@needs_ffmpeg
def test_success_writes_raw_normalised_plot_and_status(tmp_path, clips):
    runner = FakeRunner()
    res = run_ttcc(items(clips["six"]), tmp_path, ENABLED, runner, **QUIET)
    d = tmp_path / "features" / "six"
    assert res["records"][0]["status"] == SUCCESS and runner.infers == 1
    art, raw, st = load(d / RETENTION_FILE), load(d / RAW_FILE), load(d / STATUS_FILE)
    assert art["timestamps_s"] == list(range(7)) and art["retention"] == [1.0] + curve()[:6]
    assert raw["r_pred"] == curve()  # raw output keeps all 60 head values
    assert art["raw_output"]["sha256"] == hashlib.sha256((d / RAW_FILE).read_bytes()).hexdigest()
    assert art["source_sha256"] == hashlib.sha256(clips["six"].read_bytes()).hexdigest()
    assert art["model"]["checkpoint_repo"] == "liangyuch/ttcc-sft-qwen25omni-3b-v8"
    assert art["convention"]["unit"].startswith("fraction") and art["convention"]["resolution_s"] == 1.0
    assert st["status"] == SUCCESS and st["prediction_is_current"] is True
    assert (d / PLOT_FILE).read_text().startswith("<svg")
    request = runner.requests[0]
    assert request["prompts"]["assistant"] == "<cot></cot>" and request["jobs"][0]["T"] == 6
    assert request["checkpoint"]["expected_files"] == {k: list(v) for k, v in TTCC_CHECKPOINT_FILES.items()}
    assert request["checkpoint"]["allow_download"] is False


@needs_ffmpeg
def test_cache_hit_and_invalidation(tmp_path, clips):
    root = tmp_path
    runner = FakeRunner()
    run_ttcc(items(clips["six"]), root, ENABLED, runner, **QUIET)
    res = run_ttcc(items(clips["six"]), root, ENABLED, runner, **QUIET)
    assert runner.infers == 1 and res["records"][0]["cached"] is True
    assert load(root / "features" / "six" / STATUS_FILE)["stage"] == "cache"

    run_ttcc(items(clips["six"]), root, replace(ENABLED, attn_impl="eager"), runner, **QUIET)  # inference setting
    assert runner.infers == 2
    newer = FakeRunner(probe=PROBE_OK | {"libraries": {"torch": "2.8.0", "transformers": "4.57.0"}})
    run_ttcc(items(clips["six"]), root, replace(ENABLED, attn_impl="eager"), newer, **QUIET)  # library version
    assert newer.infers == 1
    patched = FakeRunner(probe=PROBE_OK | {"code": PROBE_OK["code"] | {"plugin_sha256": "changed"}})
    run_ttcc(items(clips["six"]), root, ENABLED, patched, **QUIET)  # upstream code
    assert patched.infers == 1
    run_ttcc(items(clips["six"]), root, ENABLED, patched, force=True, **QUIET)
    assert patched.infers == 2

    edited, other_root = tmp_path / "six_copy.mp4", tmp_path / "other"
    shutil.copyfile(clips["six"], edited)
    fresh = FakeRunner()
    run_ttcc([VideoInput("six", edited)], other_root, ENABLED, fresh, **QUIET)
    run_ttcc([VideoInput("six", edited)], other_root, ENABLED, fresh, **QUIET)
    assert fresh.infers == 1
    with edited.open("ab") as f:
        f.write(b"\0" * 16)  # source bytes changed, same video id
    run_ttcc([VideoInput("six", edited)], other_root, ENABLED, fresh, **QUIET)
    assert fresh.infers == 2


@needs_ffmpeg
def test_tampered_raw_output_is_not_reused(tmp_path, clips):
    runner = FakeRunner()
    run_ttcc(items(clips["six"]), tmp_path, ENABLED, runner, **QUIET)
    (tmp_path / "features" / "six" / RAW_FILE).write_text("{}")
    run_ttcc(items(clips["six"]), tmp_path, ENABLED, runner, **QUIET)
    assert runner.infers == 2


@needs_ffmpeg
def test_failure_keeps_earlier_prediction_and_other_videos(tmp_path, clips):
    run_ttcc(items(clips["six"]), tmp_path, ENABLED, FakeRunner(), **QUIET)
    d = tmp_path / "features" / "six"
    before = {f: (d / f).read_bytes() for f in (RETENTION_FILE, RAW_FILE, PLOT_FILE)}

    res = run_ttcc(items(clips["six"], clips["seven"]), tmp_path, ENABLED, FakeRunner(fail={"six"}), force=True, **QUIET)
    by = {r["video_id"]: r for r in res["records"]}
    assert by["six"]["status"] == FAILED and "out of memory" in by["six"]["reason"]
    assert by["seven"]["status"] == SUCCESS  # the batch continued
    assert {f: (d / f).read_bytes() for f in before} == before
    st = load(d / STATUS_FILE)
    assert st["status"] == FAILED and st["prediction_is_current"] is False and st["prediction_file"] == RETENTION_FILE


@needs_ffmpeg
def test_missing_result_and_invalid_prediction_fail_without_curve(tmp_path, clips):
    res = run_ttcc(items(clips["six"]), tmp_path, ENABLED, FakeRunner(omit={"six"}), **QUIET)
    assert res["records"][0]["status"] == FAILED and "no result" in res["records"][0]["reason"]

    increasing = curve()[:3] + [0.99] + curve()[4:]
    res = run_ttcc(items(clips["seven"]), tmp_path, ENABLED, FakeRunner(values=increasing), **QUIET)
    rec = res["records"][0]
    d = tmp_path / "features" / "seven"
    assert rec["status"] == FAILED and rec["stage"] == "validation_of_output"
    assert (d / RAW_REJECTED_FILE).exists()  # raw output kept for inspection
    assert not (d / RETENTION_FILE).exists() and not (d / PLOT_FILE).exists()

    res = run_ttcc(items(clips["seven"]), tmp_path, ENABLED, FakeRunner(raw_override=lambda r: r | {"r_pred": curve(30)}), **QUIET)
    assert res["records"][0]["status"] == FAILED and "30 values" in res["records"][0]["reason"]


@needs_ffmpeg
def test_unavailable_environment(tmp_path, clips):
    runner = FakeRunner(probe={"inference_possible": False, "problems": ["torch is not installed in this environment"]})
    res = run_ttcc(items(clips["six"]), tmp_path, ENABLED, runner, **QUIET)
    rec = res["records"][0]
    assert rec["status"] == UNAVAILABLE and rec["stage"] == "environment" and "torch" in rec["reason"]
    assert runner.infers == 0 and not (tmp_path / "features" / "six" / RETENTION_FILE).exists()

    no_gpu = FakeRunner(probe=PROBE_OK | {"cuda": {"available": False, "mps_available": True}})
    rec = run_ttcc(items(clips["six"]), tmp_path, replace(ENABLED, device_map="cuda"), no_gpu, **QUIET)["records"][0]
    assert rec["status"] == UNAVAILABLE and "needs a CUDA GPU" in rec["reason"] and no_gpu.infers == 0


MAC = PROBE_OK | {"cuda": {"available": False, "mps_available": True}, "machine": "Apple M2", "video_reader": "decord",
                  "libraries": {"torch": "2.11.0", "eva-decord": "0.6.1"}}


@pytest.mark.parametrize("probe,cfg,expected,problem", [
    (MAC, {}, {"device_map": "mps", "attn_impl": "sdpa"}, None),
    (MAC | {"cuda": {"available": False, "mps_available": False}}, {}, {"device_map": "cpu", "attn_impl": "sdpa"}, None),
    (PROBE_OK, {}, {"device_map": "cuda", "attn_impl": "sdpa"}, None),  # T4 without flash_attn
    (PROBE_OK | {"libraries": {"flash_attn": "2.8.3"}, "cuda": {"available": True, "devices": [{"name": "L4", "capability": "8.9"}]}},
     {}, {"device_map": "cuda", "attn_impl": "flash_attn"}, None),
    (MAC, {"attn_impl": "flash_attn"}, {"device_map": "mps", "attn_impl": "flash_attn"}, "flash_attn"),
    (MAC | {"cuda": {"available": False, "mps_available": False}}, {"device_map": "mps"}, {"device_map": "mps", "attn_impl": "sdpa"}, "MPS"),
])
def test_resolve_runtime(probe, cfg, expected, problem):
    runtime, problems = ttcc.resolve_runtime(TtccConfig(**cfg), probe)
    assert runtime == expected
    assert (problem in " ".join(problems)) if problem else problems == []


def test_config_rejects_unknown_device():
    with pytest.raises(ValueError, match="device_map"):
        TtccConfig(device_map="tpu")


@needs_ffmpeg
def test_cuda_less_run_uses_mps_and_records_deviations(tmp_path, clips):
    runner = FakeRunner(probe=MAC)
    rec = run_ttcc(items(clips["six"]), tmp_path, ENABLED, runner, **QUIET)["records"][0]
    assert rec["status"] == SUCCESS
    assert runner.requests[0]["settings"]["device_map"] == "mps" and runner.requests[0]["settings"]["attn_impl"] == "sdpa"
    art = load(tmp_path / "features" / "six" / RETENTION_FILE)
    devs = art["inference"]["deviations_from_upstream_eval"]
    assert any("mps" in d for d in devs) and any("sdpa" in d for d in devs) and any("eva-decord" in d for d in devs)
    assert art["inference"]["accelerator"] == "mps:Apple M2"
    cuda_key = ttcc.cache_key("s", 6, ENABLED, PROBE_OK)
    assert ttcc.cache_key("s", 6, ENABLED, MAC) != cuda_key  # predictions from different devices never share a cache entry


@needs_ffmpeg
def test_existing_feature_outputs_are_untouched(tmp_path, clips):
    run([clips["six"]], PipelineConfig(), root=tmp_path, **QUIET)
    d = tmp_path / "features" / "six"
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    run_ttcc(items(clips["six"]), tmp_path, ENABLED, FakeRunner(fail={"six"}), **QUIET)
    run_ttcc(items(clips["six"]), tmp_path, ENABLED, FakeRunner(), **QUIET)
    assert {name: (d / name).read_bytes() for name in before} == before


# ---------------------------------------------------------------- subprocess boundary

def test_probe_reports_missing_environment(tmp_path):
    info = SubprocessRunner().probe(TtccConfig(python=str(tmp_path / "nope" / "python")))
    assert info["inference_possible"] is False and "TTCC environment not found" in info["problems"][0]


def test_subprocess_crash_fails_unfinished_jobs(tmp_path):
    fake = tmp_path / "fake_runner.py"
    fake.write_text(textwrap.dedent("""
        import json, sys
        args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        req = json.load(open(args["--request"]))
        with open(args["--results"], "a") as f:
            f.write(json.dumps({"video_id": req["jobs"][0]["video_id"], "status": "ok", "raw_output": "x"}) + "\\n")
        print("Segmentation fault (simulated)")
        sys.exit(139)
    """))
    request = {"jobs": [{"video_id": "a"}, {"video_id": "b"}]}
    res = SubprocessRunner(script=fake).infer(TtccConfig(python=sys.executable), request, tmp_path / "work")
    by = {r["video_id"]: r for r in res}
    assert by["a"]["status"] == "ok"
    assert by["b"]["status"] == "failed" and "exited with code 139" in by["b"]["error"] and "simulated" in by["b"]["error"]


def _runner_module():
    spec = importlib.util.spec_from_file_location("ttcc_infer", PROJECT_ROOT / "scripts" / "ttcc_infer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not need torch
    return mod


def test_runner_row_matches_v8_schema():
    mod = _runner_module()
    row = mod.build_row({"video_id": "FC001", "video_path": "/v/FC001.mp4", "T": 15}, ttcc.PROMPTS)
    assert row["audios"] == [] and row["videos"] == ["/v/FC001.mp4"] and row["T"] == 15
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][1]["content"] == "This ad is 15 seconds long. Estimate the per-second retention curve."
    assert row["messages"][2]["content"] == "<cot></cot>" and "R" not in row  # no label in the input
    assert mod.contains_subsequence([1, 2, 3, 4], [3, 4]) and not mod.contains_subsequence([1, 2], [2, 3])


def test_runner_checkpoint_verification(tmp_path):
    mod = _runner_module()
    for name in mod.CHECKPOINT_REQUIRED:
        (tmp_path / name).write_text("x")
    (tmp_path / "config.json").write_text("{}")
    expected = {"config.json": [2, hashlib.sha256(b"{}").hexdigest()]}
    assert mod.verify_checkpoint_files(tmp_path, expected, False)["config.json"]["hash_verified"] is True
    with pytest.raises(mod.RunnerError, match="sha256"):
        mod.verify_checkpoint_files(tmp_path, {"config.json": [2, "0" * 64]}, False)
    with pytest.raises(mod.RunnerError, match="bytes, expected"):
        mod.verify_checkpoint_files(tmp_path, {"config.json": [3, "0" * 64]}, False)
    (tmp_path / "spk_dict.pt").unlink()
    with pytest.raises(mod.RunnerError, match="missing"):
        mod.verify_checkpoint_files(tmp_path, expected, False)


# ---------------------------------------------------------------- plot

def _success_art(T: int = 6) -> dict:
    c = parse_raw({"T": T, "r_pred": curve()}, T)
    return {"status": SUCCESS, "video_id": "v", "model": {"checkpoint_revision": "1686d5f0"}, **c}


def test_plot_has_labelled_axes_and_only_real_points():
    svg = retention_svg(_success_art())
    assert "Time (s)" in svg and "Predicted retention R(t) (fraction of viewers)" in svg
    assert svg.count("<circle") == 7  # t = 0..6, nothing interpolated
    assert "not measured retention" in svg


@pytest.mark.parametrize("art", [
    None,
    {"status": FAILED},
    {"status": UNAVAILABLE, "timestamps_s": [0, 1], "retention": [1.0, 0.9]},
    _success_art() | {"retention": [1.0, 0.9]},  # partial: lengths differ
    _success_art() | {"retention": [1.0] + [math.nan] * 6},
    _success_art() | {"timestamps_s": [0], "retention": [1.0]},
])
def test_plot_refuses_missing_partial_or_failed(art):
    assert retention_svg(art) is None


# ---------------------------------------------------------------- per-video report

def test_report_keeps_four_kinds_of_output_apart(tmp_path):
    write_video_report(tmp_path, "v")
    text = (tmp_path / REPORT_FILE).read_text()
    for heading in ("## 1. Measured multimodal features", "## 2. Heuristic attention and moment-quality scores",
                    "## 3. TTCC-predicted retention", "## 4. Evidence-backed editing recommendations"):
        assert heading in text
    assert "TTCC has not been run" in text and "No TTCC prediction is available" in text

    (tmp_path / STATUS_FILE).write_text(json.dumps({"status": FAILED, "stage": "inference", "reason": "boom"}))
    write_video_report(tmp_path, "v")
    text = (tmp_path / REPORT_FILE).read_text()
    assert "**failed** (inference): boom" in text and "| R(t) |" not in text

    art = _success_art() | {"horizon_s": 6, "summary": parse_raw({"T": 6, "r_pred": curve()}, 6)["summary"],
                            "note": ttcc.NOTE, "convention": ttcc.CONVENTION}
    (tmp_path / RETENTION_FILE).write_text(json.dumps(art))
    (tmp_path / STATUS_FILE).write_text(json.dumps({"status": SUCCESS, "prediction_is_current": True}))
    write_video_report(tmp_path, "v")
    text = (tmp_path / REPORT_FILE).read_text()
    assert "| R(t) | 1.000 | 0.950 |" in text and "not confirmed current" not in text
    assert "not measured retention" in text


@needs_ffmpeg
def test_run_ttcc_script_disabled_and_unavailable(tmp_path, clips, monkeypatch):
    monkeypatch.delenv("ENABLE_TTCC", raising=False)
    spec = importlib.util.spec_from_file_location("run_ttcc", PROJECT_ROOT / "scripts" / "run_ttcc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = ["--input", str(clips["six"]), "--artifacts", str(tmp_path), "--features", "skip"]
    assert mod.main(base) == 0
    summary = load(tmp_path / "features" / "ttcc_run_summary.json")
    assert summary["last_run"]["counts"][DISABLED] == 1
    assert (tmp_path / "features" / "six" / REPORT_FILE).exists()
    assert not (tmp_path / "features" / "six" / STATUS_FILE).exists()

    code = mod.main(base + ["--enable-ttcc", "--ttcc-python", str(tmp_path / "missing" / "python")])
    assert code == 3
    st = load(tmp_path / "features" / "six" / STATUS_FILE)
    assert st["status"] == UNAVAILABLE and "TTCC environment not found" in st["reason"]


# ---------------------------------------------------------------- videos longer than the trained range

@pytest.mark.parametrize("duration,mode,expected", [
    (25.0, "first_60s", (25, None)), (60.4, "first_60s", (60, None)), (71.399, "first_60s", (60, 60.0)),
    (143.2, "first_60s", (60, 60.0)), (71.4, "skip", (None, None)), (3.0, "first_60s", (None, None)), (None, "first_60s", (None, None)),
])
def test_plan_window(duration, mode, expected):
    T, end, why = ttcc.plan_window(duration, mode)
    assert (T, end) == expected
    assert (why is None) == (T is not None)


def test_runner_row_limits_long_videos_to_the_window():
    mod = _runner_module()
    row = mod.build_row({"video_id": "FC001", "video_path": "/v/FC001.mp4", "T": 60, "video_end": 60.0}, ttcc.PROMPTS)
    assert row["chat_template_kwargs"] == {"video_start": 0.0, "video_end": 60.0}
    assert "60 seconds long" in row["messages"][1]["content"]
    assert "chat_template_kwargs" not in mod.build_row({"video_id": "a", "video_path": "/a.mp4", "T": 9, "video_end": None}, ttcc.PROMPTS)


@needs_ffmpeg
def test_long_video_predicts_first_60s_only(tmp_path, clips):
    runner = FakeRunner()
    legacy = replace(ENABLED, long_videos="first_60s")  # the pre-windowing policy; the default is now "windows" (test_ttcc_long.py)
    rec = run_ttcc(items(clips["long"]), tmp_path, legacy, runner, **QUIET)["records"][0]
    assert rec["status"] == SUCCESS and rec["window_end_s"] == 60.0
    job = runner.requests[0]["jobs"][0]
    assert job["T"] == 60 and job["video_end"] == 60.0
    d = tmp_path / "features" / "long"
    art = load(d / RETENTION_FILE)
    assert art["out_of_trained_range"] is True and art["partial_coverage"] is True and art["timestamps_s"] == list(range(61))
    assert art["coverage"]["predicted_until_s"] == 60.0 and art["coverage"]["unpredicted_s"][0] == 60.0
    assert art["coverage"]["unpredicted_s"][1] == pytest.approx(62.4, abs=0.2)
    assert any("first 60 s" in x for x in art["inference"]["deviations_from_upstream_eval"])
    svg = (d / PLOT_FILE).read_text()
    assert svg.count("<circle") == 61 and "not predicted" in svg  # tail shown empty, not filled
    write_video_report(d, "long")
    assert "Partial prediction" in (d / REPORT_FILE).read_text()

    skipped = run_ttcc(items(clips["long"]), tmp_path / "skip", replace(ENABLED, long_videos="skip"), FakeRunner(), **QUIET)
    assert skipped["records"][0]["status"] == UNAVAILABLE and "longer than" in skipped["records"][0]["reason"]
    full = ttcc.cache_key("s", 60, ENABLED, PROBE_OK)
    assert ttcc.cache_key("s", 60, ENABLED, PROBE_OK, window_end=60.0) != full  # a 60 s window is not a 60 s video


def test_runner_shrinks_window_to_fit_max_length():
    mod = _runner_module()
    assert mod.shrink_horizon(60, 35962, 32768) == 54
    assert mod.shrink_horizon(54, 32800, 32768) == 53  # always at least one second shorter
    assert mod.shrink_horizon(30, None, 32768) == 29

    class TooLong(ValueError):
        pass

    class Template:  # tokens grow 600 per second of window, like the real encoder (~36k for 60 s)
        def encode(self, inputs):
            n = 600 * inputs["T"] + 200
            if n > 32768:
                raise TooLong(f"Current length of row({n}) is larger than the max_length(32768).")
            return {"n": n}

    class Inputs:
        @staticmethod
        def from_dict(row):
            return row

    job = {"video_id": "FC001", "video_path": "/v.mp4", "T": 60, "video_end": 60.0, "shrink_to_fit": True}
    row, enc, window = mod.encode_fitting(job, ttcc.PROMPTS, Template(), Inputs, TooLong, 32768)
    assert row["T"] == 54 and row["chat_template_kwargs"]["video_end"] == 54.0 and enc["n"] <= 32768
    assert window["requested_T"] == 60 and window["T"] == 54 and "max_length" in window["reason"]
    with pytest.raises(TooLong):
        mod.encode_fitting(job | {"shrink_to_fit": False}, ttcc.PROMPTS, Template(), Inputs, TooLong, 32768)


@needs_ffmpeg
def test_shrunk_window_from_runner_is_recorded(tmp_path, clips):
    def shrink(raw):
        return raw | {"T": 5, "window": {"requested_T": 6, "T": 5, "video_end_s": 5.0, "reason": "exceeded max_length"}}
    rec = run_ttcc(items(clips["six"]), tmp_path, ENABLED, FakeRunner(raw_override=shrink), **QUIET)["records"][0]
    assert rec["status"] == SUCCESS and rec["window_end_s"] == 5.0
    art = load(tmp_path / "features" / "six" / RETENTION_FILE)
    assert art["timestamps_s"] == list(range(6)) and art["partial_coverage"] is True and art["out_of_trained_range"] is False
    assert any("too many tokens" in d for d in art["inference"]["deviations_from_upstream_eval"])


def test_runner_retries_with_a_shorter_window_after_out_of_memory() -> None:
    mod = _runner_module()
    assert mod.next_window_after_oom(60) == 42 and mod.next_window_after_oom(42) == 29 and mod.next_window_after_oom(29) == 20
    assert mod.next_window_after_oom(7) is None and mod.next_window_after_oom(5) is None  # nothing shorter than 5 s
    assert mod.next_window_after_oom(6) is None
    assert all(mod.next_window_after_oom(t) < t for t in range(8, 61))  # always strictly shorter, so the retry loop ends

    class OutOfMemoryError(RuntimeError):
        pass
    assert mod.is_oom(OutOfMemoryError("x")) and mod.is_oom(RuntimeError("CUDA out of memory. Tried to allocate 8 GiB"))
    assert not mod.is_oom(ValueError("shape mismatch"))
