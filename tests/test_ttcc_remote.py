"""TTCC in a Hugging Face Space: the Space service (with a fake runner script), the remote client,
upload permissions, and a real localhost round trip. No model, no GPU and no network beyond localhost."""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import ttcc
from hookline.config import TTCC_CHECKPOINT_FILES, TtccConfig
from hookline.pipeline import VideoInput
from hookline.tribe import JobTransport, TribeRequestError
from hookline.ttcc import RETENTION_FILE, SUCCESS, UNAVAILABLE, RemoteRunner, run_ttcc, upload_block_reason

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = PROJECT_ROOT / "endpoint" / "ttcc_space" / "app.py"
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
REMOTE = TtccConfig(enabled=True, remote=True, allow_remote=True)
QUIET = dict(log=lambda *_: None)
ENV = {"TTCC_ENDPOINT_URL": "https://example.invalid", "HF_TOKEN": "hf_secret_token_456"}
PROBE_OK = {"inference_possible": True, "problems": [], "libraries": {"torch": "2.6.0", "transformers": "4.56.2"},
            "cuda": {"available": True, "devices": [{"name": "NVIDIA L4", "capability": "8.9"}]},
            "code": {"commit": "893da03b", "dirty": False, "plugin_sha256": "p", "runner_version": "1", "runner_sha256": "r"},
            "video_reader": "decord"}

# A stand-in for scripts/ttcc_infer.py: same CLI and output files, no model.
FAKE_RUNNER = textwrap.dedent('''
    import json, sys
    cmd, args = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2]))
    if cmd == "probe":
        print(json.dumps(%(probe)r)); sys.exit(0)
    req = json.load(open(args["--request"]))
    job = req["jobs"][0]
    if job["video_id"] == "boom":
        open(args["--results"], "a").write(json.dumps({"video_id": "boom", "status": "failed", "stage": "inference", "error": "CUDA out of memory"}) + "\\n")
        sys.exit(1)
    values = [max(0.95 - 0.01 * i, 0.01) for i in range(60)]
    raw = {"video_id": job["video_id"], "T": job["T"], "r_pred": values, "input_tokens": 999, "anchor": "</cot>",
           "inference_seconds": 2.0, "provenance": {"device": "cuda:0", "settings": req["settings"],
           "video_seen": open(job["video_path"], "rb").read().decode("latin1")}}
    json.dump(raw, open(job["raw_output"], "w"))
    open(args["--results"], "a").write(json.dumps({"video_id": job["video_id"], "status": "ok", "raw_output": job["raw_output"]}) + "\\n")
''')


def load_space():
    spec = importlib.util.spec_from_file_location("ttcc_space_app", SPACE_APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def service(tmp_path):
    mod = load_space()
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER % {"probe": PROBE_OK})
    svc = mod.TtccService(runner_cmd=[sys.executable, str(script)], repo_dir=tmp_path / "go_viral", ckpt_dir=tmp_path / "ckpt")
    return mod, svc


def payload(video_id="six", T=6, video=b"VIDEOBYTES", **over):
    job = {"video_id": video_id, "T": T, "video_end": None, "shrink_to_fit": True}
    req = {"settings": {"device_map": "cuda", "attn_impl": "sdpa"}, "prompts": ttcc.PROMPTS,
           "checkpoint": {"repo_id": "liangyuch/ttcc-sft-qwen25omni-3b-v8", "revision": "1686d5f00408bfd8f6a32be12e71248b6f8ff447",
                          "expected_files": {k: list(v) for k, v in TTCC_CHECKPOINT_FILES.items()}, "verify_weights_sha256": False},
           "job": job}
    return {"request": req | over.pop("req", {}), "video_b64": base64.b64encode(video).decode(), "filename": "six.mp4"} | over


# ---------------------------------------------------------------- the Space service

def test_service_prepares_then_runs_a_video(service):
    mod, svc = service
    assert svc.handle(payload())["error"].startswith("the Space is still preparing")  # not ready yet
    svc.prepare(download=lambda: None)
    assert svc.ready and svc.probe["inference_possible"] is True
    out = svc.handle(payload())
    assert out["result"]["status"] == "ok" and len(out["raw"]["r_pred"]) == 60
    assert out["raw"]["provenance"]["video_seen"] == "VIDEOBYTES"  # the uploaded bytes reached the runner
    assert out["raw"]["provenance"]["settings"] == {"device_map": "cuda", "attn_impl": "sdpa"}


def test_service_rejects_bad_requests_and_reports_runner_failures(service):
    mod, svc = service
    svc.prepare(download=lambda: None)
    assert "bad request" in svc.handle({"request": {}, "video_b64": "AA=="})["error"]
    assert "bad request" in svc.handle(payload(video_b64="@@@"))["error"]
    assert "unsupported video suffix" in svc.handle(payload(filename="x.txt"))["error"]
    wrong = payload()
    wrong["request"]["checkpoint"]["revision"] = "0" * 40
    assert "this Space serves" in svc.handle(wrong)["error"]
    failed = svc.handle(payload(video_id="boom"))
    assert failed["result"]["status"] == "failed" and "out of memory" in failed["result"]["error"]


def test_service_checkpoint_failure_is_reported(service):
    mod, svc = service

    def broken():
        raise OSError("no space left on device")
    svc.prepare(download=broken)
    assert not svc.ready and "no space left" in svc.error
    assert "no space left" in svc.handle(payload())["error"]


# ---------------------------------------------------------------- the remote client

class SpaceTransport:
    """JobTransport stand-in: hands the payload to the service directly (what the Space's job store returns)."""

    posted: list = []  # shared across instances: RemoteRunner makes a fresh transport per call

    def __init__(self, svc, health=None, fail=None, fail_probe=False):
        self.svc, self.health, self.fail, self.fail_probe = svc, health, fail, fail_probe

    def request(self, method, path, payload=None):
        if self.fail and self.fail_probe:
            raise self.fail
        return self.health

    def post(self, payload):
        SpaceTransport.posted.append(payload)
        if self.fail:
            raise self.fail
        return self.svc.handle(payload)


def make_runner(svc, **kw):
    holder = {}
    SpaceTransport.posted = []

    def factory(url, token, cfg, sleep=None):
        holder["t"] = SpaceTransport(svc, **kw)
        return holder["t"]
    return RemoteRunner("https://x", "tok", transport_cls=factory), holder


def test_remote_probe_reports_readiness_and_unreachable_space(service):
    mod, svc = service
    health = {"probe": PROBE_OK, "checkpoint_ready": True, "checkpoint_error": None, "jobs": {}}
    assert make_runner(svc, health=health)[0].probe(REMOTE)["inference_possible"] is True
    starting = make_runner(svc, health=health | {"checkpoint_ready": False})[0].probe(REMOTE)
    assert starting["inference_possible"] is False and "still downloading" in starting["problems"][0]
    down = make_runner(svc, fail=TribeRequestError("HTTP 404 Not Found: x"), fail_probe=True)[0].probe(REMOTE)
    assert down["inference_possible"] is False and "not reachable" in down["problems"][0]
    assert "TTCC probe" in make_runner(svc, health={"nope": 1})[0].probe(REMOTE)["problems"][0]


@needs_ffmpeg
def test_remote_end_to_end_with_fake_space(tmp_path, service):
    from test_ttcc import ffmpeg
    mod, svc = service
    svc.prepare(download=lambda: None)
    clip = tmp_path / "six.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x284:r=10:d=6", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(clip))
    runner, holder = make_runner(svc, health={"probe": PROBE_OK, "checkpoint_ready": True})
    res = run_ttcc([VideoInput("six", clip)], tmp_path / "root", REMOTE, runner, **QUIET)
    rec = res["records"][0]
    assert rec["status"] == SUCCESS, rec
    d = tmp_path / "root" / "features" / "six"
    art = json.loads((d / RETENTION_FILE).read_text())
    assert art["timestamps_s"] == list(range(7)) and art["retention"][1] == pytest.approx(0.95)
    assert art["inference"]["accelerator"].startswith("cuda:NVIDIA L4")  # from the Space's probe, not this machine
    assert SpaceTransport.posted[0]["request"]["job"]["T"] == 6
    assert (d / "ttcc_retention.svg").exists()
    again = run_ttcc([VideoInput("six", clip)], tmp_path / "root", REMOTE, runner, **QUIET)["records"][0]
    assert again["cached"] is True and len(SpaceTransport.posted) == 1  # cache hit: nothing sent again


@needs_ffmpeg
def test_remote_failures_stay_failures(tmp_path, service):
    from test_ttcc import ffmpeg
    mod, svc = service
    svc.prepare(download=lambda: None)
    clip = tmp_path / "boom.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x284:r=10:d=6", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(clip))
    health = {"probe": PROBE_OK, "checkpoint_ready": True}
    rec = run_ttcc([VideoInput("boom", clip)], tmp_path / "r1", REMOTE, make_runner(svc, health=health)[0], **QUIET)["records"][0]
    assert rec["status"] == "failed" and "out of memory" in rec["reason"]
    err = RemoteRunner("https://x", "tok", transport_cls=lambda *a, **k: SpaceTransport(svc, health=health, fail=TribeRequestError(
        "HTTP 401 Unauthorized: bad token tok")))
    rec = run_ttcc([VideoInput("boom", clip)], tmp_path / "r2", REMOTE, err, **QUIET)["records"][0]
    assert rec["status"] == "failed" and "tok" not in rec["reason"].replace("token", "")
    assert not (tmp_path / "r2" / "features" / "boom" / RETENTION_FILE).exists()


# ---------------------------------------------------------------- upload permission and configuration

def test_company_reels_need_approval_and_nothing_is_sent_without_it(tmp_path):
    company = tmp_path / "01-Hookline" / "face" / "videos" / "FC002.mp4"
    company.parent.mkdir(parents=True)
    company.write_bytes(b"x")
    assert "--ttcc-allow-remote" in upload_block_reason("FC002", company, TtccConfig(remote=True), tmp_path)
    assert "--ttcc-approve FC002" in upload_block_reason("FC002", company, REMOTE, tmp_path)
    assert upload_block_reason("FC002", company, replace(REMOTE, approved_ids=("FC002",)), tmp_path) is None
    assert upload_block_reason("mine", tmp_path / "mine.mp4", REMOTE, tmp_path) is None


@needs_ffmpeg
def test_unapproved_company_reel_is_blocked_before_any_upload(tmp_path, service):
    mod, svc = service
    company = PROJECT_ROOT / "01-Hookline" / "face" / "videos"
    sample = next(company.glob("*.mp4"), None)
    if sample is None:
        pytest.skip("company dataset not present")
    runner, holder = make_runner(svc, health={"probe": PROBE_OK, "checkpoint_ready": True})
    rec = run_ttcc([VideoInput(sample.stem, sample)], tmp_path, REMOTE, runner, **QUIET)["records"][0]
    assert rec["status"] == UNAVAILABLE and rec["stage"] == "permission" and "--ttcc-approve" in rec["reason"]
    assert SpaceTransport.posted == []


def test_remote_requires_url_and_token(tmp_path):
    runner, why = ttcc.remote_runner_or_reason(REMOTE, env={}, dotenv=Path("/nonexistent/.env"))
    assert runner is None and "TTCC_ENDPOINT_URL and HF_TOKEN not set" in why
    runner, why = ttcc.remote_runner_or_reason(REMOTE, env=ENV, dotenv=Path("/nonexistent/.env"))
    assert isinstance(runner, RemoteRunner) and why is None


def test_remote_off_by_default_and_not_in_cache_key():
    assert TtccConfig().remote is False and TtccConfig().allow_remote is False
    local, remote = TtccConfig(enabled=True), TtccConfig(enabled=True, remote=True, allow_remote=True, poll_s=1)
    assert ttcc.cache_key("s", 6, local, PROBE_OK) == ttcc.cache_key("s", 6, remote, PROBE_OK)  # same model and settings, same key


# ---------------------------------------------------------------- real HTTP round trip (needs fastapi)

def test_space_app_over_real_http(tmp_path, service):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    mod, svc = service
    svc.prepare(download=lambda: None)
    server = uvicorn.Server(uvicorn.Config(mod.create_app(svc, start=False), host="127.0.0.1", port=0, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started
    url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    try:
        runner = RemoteRunner(url, "tok", transport_cls=JobTransport)
        probe = runner.probe(replace(REMOTE, poll_s=0.05))
        assert probe["inference_possible"] is True and probe["cuda"]["devices"][0]["name"] == "NVIDIA L4"
        clip = tmp_path / "six.mp4"
        clip.write_bytes(b"VIDEOBYTES")
        cfg = replace(REMOTE, poll_s=0.05, timeout_s=60)
        request = {"settings": {"device_map": "cuda"}, "prompts": ttcc.PROMPTS,
                   "checkpoint": payload()["request"]["checkpoint"],
                   "jobs": [{"video_id": "six", "video_path": str(clip), "T": 6, "video_end": None, "shrink_to_fit": True,
                             "raw_output": str(tmp_path / "raw.json")}]}
        results = runner.infer(cfg, request, tmp_path / "work")
        assert results[0]["status"] == "ok", results
        raw = json.loads((tmp_path / "raw.json").read_text())
        assert len(raw["r_pred"]) == 60 and raw["provenance"]["video_seen"] == "VIDEOBYTES"
    finally:
        server.should_exit = True
