"""TRIBE v2 remote-endpoint stage: configuration, upload permission, HTTP client, response validation,
caching, failure isolation, and the endpoint handler's contract. No model runs and no network leaves
this machine: the endpoint is faked (in-process objects or a localhost HTTP server)."""

from __future__ import annotations

import base64
import http.server
import importlib.util
import io
import json
import shutil
import sys
import threading
import types
import urllib.error
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hookline import tribe
from hookline.config import TRIBE_MODEL_REVISION, TribeConfig
from hookline.context import MediaContext
from hookline.status import ABSENT, AVAILABLE, FAILED
from hookline.tribe import (
    ARTIFACT, FSAVERAGE5_VERTICES, PREDS_FILE, HttpTransport, TribeOutputError, TribeRequestError, cached_tribe,
    extract_tribe, parse_response, remote_block_reason, write_tribe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
ENV = {"TRIBE_ENDPOINT_URL": "https://example.invalid/tribe", "HF_TOKEN": "hf_secret_token_123"}
ON = TribeConfig(enabled=True, allow_remote=True)


def response(n: int = 4, vertices: int = FSAVERAGE5_VERTICES, **over) -> dict:
    preds = (np.arange(n * vertices, dtype="<f4").reshape(n, vertices) % 7) - 3
    return {"handler_version": "1", "model_repo": "facebook/tribev2", "model_revision": TRIBE_MODEL_REVISION,
            "tr_s": 1.0, "mesh": "fsaverage5", "shape": [n, vertices], "dtype": "float32",
            "preds_b64": base64.b64encode(preds.tobytes()).decode(), "language": "english",
            "segments": [{"start_s": float(i), "duration_s": 1.0} for i in range(n)],
            "events": {"Video": 1, "Audio": 1, "Word": 12}, "timing": {"predict_seconds": 3.0},
            "versions": {"tribev2": "0.1.0"}} | over


class FakeTransport:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.payloads = resp, exc, []
        self.requests = [{"attempt": 1, "http_status": 200}]

    def post(self, payload):
        self.payloads.append(payload)
        if self.exc:
            raise self.exc
        return self.resp


NO_DOTENV = Path("/nonexistent/.env")  # tests never read the developer's real .env


def extract(*args, **kwargs):
    return extract_tribe(*args, dotenv=NO_DOTENV, **kwargs)


def ctx_for(path: Path, video_id: str | None = None, **kw) -> MediaContext:
    return MediaContext(video_id=video_id or path.stem, source_path=str(path), source_sha256="abc", prep_key=None,
                        duration=4.0, visual_stream=kw.pop("visual", AVAILABLE), audio_stream=kw.pop("audio", AVAILABLE))


@pytest.fixture
def video(tmp_path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 1000)
    return p


# ---------------------------------------------------------------- configuration and permission

def test_disabled_by_default(monkeypatch, video):
    monkeypatch.delenv("ENABLE_TRIBE", raising=False)
    assert TribeConfig().enabled is False and TribeConfig.from_env().enabled is False
    called = FakeTransport(response())
    assert extract(ctx_for(video), TribeConfig(), lambda u, t: called, env=ENV) == (None, None)
    assert called.payloads == []
    monkeypatch.setenv("ENABLE_TRIBE", "true")
    assert TribeConfig.from_env().enabled is True


def test_invalid_config():
    with pytest.raises(ValueError, match="language"):
        TribeConfig(language="hindi")
    with pytest.raises(ValueError):
        TribeConfig(timeout_s=0)


def test_company_videos_need_explicit_approval(tmp_path):
    company = tmp_path / "01-Hookline" / "face" / "videos" / "FC001.mp4"
    company.parent.mkdir(parents=True)
    company.write_bytes(b"x")
    ctx = ctx_for(company)
    assert "--tribe-allow-remote" in remote_block_reason(ctx, TribeConfig(enabled=True), tmp_path)
    assert "--tribe-approve FC001" in remote_block_reason(ctx, ON, tmp_path)
    assert remote_block_reason(ctx, replace(ON, approved_ids=("FC001",)), tmp_path) is None
    assert remote_block_reason(ctx_for(tmp_path / "mine.mp4"), ON, tmp_path) is None


def test_blocked_or_unconfigured_sends_nothing(video, tmp_path):
    t = FakeTransport(response())
    art, preds = extract(ctx_for(video), TribeConfig(enabled=True), lambda u, k: t, env=ENV)
    assert art["status"] == FAILED and not art["sent"] and "not sent" in art["reason"] and preds is None
    art, _ = extract(ctx_for(video), ON, lambda u, k: t, env={"HF_TOKEN": "x"}, root=tmp_path)
    assert "TRIBE_ENDPOINT_URL not set" in art["reason"]
    art, _ = extract(ctx_for(video), ON, lambda u, k: t, env={}, root=tmp_path)
    assert "TRIBE_ENDPOINT_URL and HF_TOKEN not set" in art["reason"]
    art, _ = extract(ctx_for(video), replace(ON, max_upload_mb=0.0001), lambda u, k: t, env=ENV, root=tmp_path)
    assert "exceeds max_upload_mb" in art["reason"]
    art, _ = extract(ctx_for(video, visual=ABSENT, audio=ABSENT), ON, lambda u, k: t, env=ENV, root=tmp_path)
    assert art["status"] == ABSENT
    assert t.payloads == []


def test_secrets_from_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER_KEY=g\nexport HF_TOKEN='hf_from_file'\nTRIBE_ENDPOINT_URL=https://x.endpoints.huggingface.cloud\n")
    assert tribe.load_secret("HF_TOKEN", env={}, dotenv=env_file) == "hf_from_file"
    assert tribe.load_secret("HF_TOKEN", env={"HF_TOKEN": "from_env"}, dotenv=env_file) == "from_env"
    assert tribe.load_secret("MISSING", env={}, dotenv=env_file) is None


# ---------------------------------------------------------------- request, success and failures

def test_success_returns_actual_outputs(video, tmp_path):
    t = FakeTransport(response(n=5))
    art, preds = extract(ctx_for(video), ON, lambda u, k: t, env=ENV, root=tmp_path)
    assert art["status"] == AVAILABLE and art["measured"] and art["sent"]
    assert preds.shape == (5, FSAVERAGE5_VERTICES) and preds.dtype == np.float32
    expected = (np.arange(5 * FSAVERAGE5_VERTICES, dtype="<f4").reshape(5, -1) % 7) - 3
    assert np.array_equal(preds, expected)  # exactly what the endpoint returned
    assert art["timestamps_s"] == [0.0, 1.0, 2.0, 3.0, 4.0] and art["tr_s"] == 1.0
    assert "not a retention" in art["output_meaning"]["not"]
    sent = t.payloads[0]
    assert base64.b64decode(sent["inputs"]["video_b64"]) == video.read_bytes() and sent["inputs"]["filename"] == "clip.mp4"
    assert ENV["HF_TOKEN"] not in json.dumps(art) and ENV["TRIBE_ENDPOINT_URL"] not in json.dumps(art)


@pytest.mark.parametrize("resp,match", [
    ({"error": "CUDA out of memory"}, "endpoint error: CUDA out of memory"),
    ({"shape": [4, 20484]}, "lacks"),
    (response(model_revision="0" * 40), "expected facebook/tribev2@"),
    (response(n=2, vertices=100), "fsaverage5 has 20484"),
    (response(mesh="fsaverage6"), "unexpected mesh"),
    (response(shape=[3, FSAVERAGE5_VERTICES]), "bytes"),
    (response(preds_b64="not base64!"), "not base64"),
    (response(segments=[{"start_s": 0.0, "duration_s": 1.0}]), "segments for 4 prediction rows"),
    (response(n=2, segments=[{"start_s": 1.0, "duration_s": 1.0}, {"start_s": 0.0, "duration_s": 1.0}]), "time order"),
    (response(tr_s=0), "tr_s"),
    ("text", "not a JSON object"),
])
def test_invalid_responses_fail(resp, match):
    with pytest.raises(TribeOutputError, match=match):
        parse_response(resp, ON)


def test_nan_predictions_fail():
    bad = np.full((2, FSAVERAGE5_VERTICES), np.nan, dtype="<f4")
    with pytest.raises(TribeOutputError, match="NaN"):
        parse_response(response(n=2, preds_b64=base64.b64encode(bad.tobytes()).decode()), ON)


def test_request_errors_become_reasons_with_token_scrubbed(video, tmp_path):
    exc = TribeRequestError(f"HTTP 401 Unauthorized: bad token {ENV['HF_TOKEN']} (check HF_TOKEN)")
    art, preds = extract(ctx_for(video), ON, lambda u, k: FakeTransport(exc=exc), env=ENV, root=tmp_path)
    assert art["status"] == FAILED and art["sent"] and preds is None
    assert ENV["HF_TOKEN"] not in art["reason"] and "***" in art["reason"]
    art, _ = extract(ctx_for(video), ON, lambda u, k: FakeTransport(exc=KeyError("boom")), env=ENV, root=tmp_path)
    assert art["status"] == FAILED and "KeyError" in art["reason"]  # unexpected errors never escape


# ---------------------------------------------------------------- HTTP client

class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://e", code, {503: "Service Unavailable", 401: "Unauthorized"}.get(code, "Err"),
                                  {}, io.BytesIO(body.encode()))


def test_http_retries_cold_start_then_succeeds():
    calls, sleeps = [], []

    def opener(req, timeout):
        calls.append(req)
        if len(calls) < 3:
            raise http_error(503, "Model is initializing")
        return FakeResponse(json.dumps({"ok": 1}).encode())

    t = HttpTransport("https://e", "tok", replace(ON, retry_wait_s=10), sleep=sleeps.append, opener=opener)
    assert t.post({"a": 1}) == {"ok": 1}
    assert sleeps == [10, 20] and [r["http_status"] for r in t.requests] == [503, 503, 200]
    assert calls[0].get_header("Authorization") == "Bearer tok" and json.loads(calls[0].data) == {"a": 1}


def test_http_client_errors_do_not_retry():
    sleeps = []
    t = HttpTransport("https://e", "tok", ON, sleep=sleeps.append, opener=lambda r, timeout: (_ for _ in ()).throw(http_error(401)))
    with pytest.raises(TribeRequestError, match="check HF_TOKEN"):
        t.post({})
    assert sleeps == [] and len(t.requests) == 1


def test_http_gives_up_after_retries():
    def opener(req, timeout):
        raise urllib.error.URLError("connection refused")
    t = HttpTransport("https://e", "tok", replace(ON, max_retries=2, retry_wait_s=0), sleep=lambda s: None, opener=opener)
    with pytest.raises(TribeRequestError, match="gave up after 3 attempts"):
        t.post({})


def test_http_against_local_server():
    """Real urllib round trip: headers and JSON body reach a server, JSON comes back."""
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["auth"] = self.headers["Authorization"]
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps(response(n=2)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        resp = HttpTransport(url, "tok", ON).post({"inputs": {"video_b64": "AA==", "filename": "a.mp4"}})
    finally:
        server.shutdown()
    preds, segs = parse_response(resp, ON)
    assert preds.shape == (2, FSAVERAGE5_VERTICES) and seen["auth"] == "Bearer tok" and seen["body"]["inputs"]["filename"] == "a.mp4"


# ---------------------------------------------------------------- artifacts and caching

def test_failure_never_replaces_success_and_cache_checks_file(video, tmp_path):
    ctx = ctx_for(video)
    art, preds = extract(ctx, ON, lambda u, k: FakeTransport(response()), env=ENV, root=tmp_path)
    write = lambda p, d: p.write_text(json.dumps(d))  # noqa: E731
    assert write_tribe(tmp_path, art, preds, write)
    assert np.array_equal(np.load(tmp_path / PREDS_FILE), preds)
    key = tribe.tribe_key(ctx, ON)
    assert cached_tribe(tmp_path, key)["measured"] is True
    assert cached_tribe(tmp_path, tribe.tribe_key(ctx, replace(ON, language="french"))) is None  # setting change

    failed, _ = extract(ctx, ON, lambda u, k: FakeTransport({"error": "boom"}), env=ENV, root=tmp_path)
    assert write_tribe(tmp_path, failed, None, write) is False
    assert json.loads((tmp_path / ARTIFACT).read_text())["measured"] is True

    np.save(tmp_path / PREDS_FILE, np.zeros((1, 1), dtype=np.float32))  # tampered predictions
    assert cached_tribe(tmp_path, key) is None


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
def test_pipeline_outputs_unchanged_when_tribe_fails(tmp_path, monkeypatch):
    from test_multimodal import CFG, OCFG, SCFG, build_features, ffmpeg, make_backends  # shared fixtures/helpers
    from hookline.multimodal import process_video_multimodal
    from hookline.pipeline import VideoInput

    clip = tmp_path / "talk.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=gray:s=180x320:r=30:d=3", "-f", "lavfi", "-i", "sine=duration=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip))
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    outputs = {}
    for label, cfg, transport in (("off", TribeConfig(), None), ("fail", ON, FakeTransport(exc=TribeRequestError("HTTP 503"))),
                                  ("ok", ON, FakeTransport(response(n=3)))):
        root = tmp_path / label
        build_features([clip], root)
        backends, _, _ = make_backends()
        backends.tribe = (lambda u, k, t=transport: t) if transport else None
        res = process_video_multimodal(VideoInput("talk", clip), root, CFG, SCFG, OCFG, backends, tribe_cfg=cfg)
        d = root / "features" / "talk"
        outputs[label] = {p.name: p.read_bytes() for p in d.iterdir() if p.suffix == ".csv"}
        outputs[label + "_status"] = res["status"]
        outputs[label + "_tribe"] = res["tribe"]
    assert outputs["off"] == outputs["fail"] == outputs["ok"]  # CSV outputs identical whatever TRIBE does
    assert outputs["off_status"] == outputs["fail_status"] == outputs["ok_status"]
    assert outputs["off_tribe"] is None and outputs["fail_tribe"]["status"] == FAILED
    assert outputs["ok_tribe"]["status"] == AVAILABLE and (tmp_path / "ok" / "features" / "talk" / PREDS_FILE).exists()
    assert not (tmp_path / "off" / "features" / "talk" / ARTIFACT).exists()


# ---------------------------------------------------------------- endpoint handler contract

def load_handler(monkeypatch, predict=None):
    """Import endpoint/tribev2/handler.py with stand-ins for tribev2 and huggingface_hub (no model)."""
    class Seg:
        def __init__(self, start):
            self.start, self.duration = start, 1.0

    class FakeModel:
        data = types.SimpleNamespace(TR=1.0)

        @classmethod
        def from_pretrained(cls, path, cache_folder=None):
            cls.loaded_from = path
            return cls()

        def get_events_dataframe(self, video_path):
            assert Path(video_path).read_bytes() == b"VIDEO"
            counts = {"Word": 2, "Video": 1, "Audio": 1}
            return {"type": types.SimpleNamespace(value_counts=lambda: counts)}  # the one pandas call the handler makes

        def predict(self, events, verbose=True):
            if predict:
                return predict()
            return np.ones((3, FSAVERAGE5_VERTICES), dtype=np.float32) * 0.5, [Seg(0.0), Seg(1.0), Seg(2.0)]

    monkeypatch.setitem(sys.modules, "tribev2", types.SimpleNamespace(TribeModel=FakeModel))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=lambda repo, revision, allow_patterns: f"/snap/{repo}@{revision}"))
    spec = importlib.util.spec_from_file_location("tribe_handler", PROJECT_ROOT / "endpoint" / "tribev2" / "handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, FakeModel


def test_handler_round_trip_matches_client(monkeypatch):
    mod, model = load_handler(monkeypatch)
    assert mod.MODEL_REVISION == TRIBE_MODEL_REVISION  # handler and client pin the same revision
    h = mod.EndpointHandler()
    assert model.loaded_from == f"/snap/facebook/tribev2@{TRIBE_MODEL_REVISION}"
    resp = h({"inputs": {"video_b64": base64.b64encode(b"VIDEO").decode(), "filename": "x.mp4"}})
    preds, segs = parse_response(json.loads(json.dumps(resp)), ON)
    assert preds.shape == (3, FSAVERAGE5_VERTICES) and float(preds[0, 0]) == 0.5 and segs["starts"] == [0.0, 1.0, 2.0]
    assert resp["events"] == {"Word": 2, "Video": 1, "Audio": 1}


def test_handler_rejects_bad_requests_and_reports_errors(monkeypatch):
    mod, _ = load_handler(monkeypatch, predict=lambda: (_ for _ in ()).throw(RuntimeError("CUDA OOM")))
    h = mod.EndpointHandler()
    assert "expected JSON" in h({"inputs": "x"})["error"]
    assert "suffix" in h({"inputs": {"video_b64": "AA==", "filename": "a.txt"}})["error"]
    assert "language" in h({"inputs": {"video_b64": "AA==", "filename": "a.mp4"}, "parameters": {"language": "hindi"}})["error"]
    assert "base64" in h({"inputs": {"video_b64": "@@@", "filename": "a.mp4"}})["error"]
    err = h({"inputs": {"video_b64": base64.b64encode(b"VIDEO").decode(), "filename": "a.mp4"}})
    assert err == {"error": "RuntimeError: CUDA OOM"}
    with pytest.raises(TribeOutputError, match="CUDA OOM"):
        parse_response(err, ON)


# ---------------------------------------------------------------- job API (Space): client and server

SPACE_DIR = PROJECT_ROOT / "endpoint" / "tribev2_space"


def load_space_app():
    spec = importlib.util.spec_from_file_location("tribe_space_app", SPACE_DIR / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # imports no web or model library until create_app() runs
    return mod


def wait_for(store, job_id, status, seconds=5.0):
    import time
    end = time.time() + seconds
    while time.time() < end:
        rep = store.report(job_id)
        if rep["status"] in status:
            return rep
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}: {store.report(job_id)}")


def test_job_store_runs_jobs_in_order_and_reports_failures():
    mod = load_space_app()
    seen = []

    def handle(req):
        seen.append(req["n"])
        if req["n"] == 2:
            return {"error": "RuntimeError: CUDA OOM"}
        if req["n"] == 3:
            raise ValueError("boom")
        return {"echo": req["n"]}

    store = mod.JobStore(handle)
    ids = [store.submit({"n": n}) for n in (1, 2, 3)]
    done = [wait_for(store, i, ("done", "failed")) for i in ids]
    assert seen == [1, 2, 3]
    assert [d["status"] for d in done] == ["done", "failed", "failed"]
    assert done[0]["result"] == {"echo": 1} and "request" not in done[0] and done[0]["timing"]["run_s"] >= 0
    assert done[1]["error"] == "RuntimeError: CUDA OOM" and "ValueError: boom" in done[2]["error"]
    assert store.report("nope") is None and store.counts()["done"] == 1


class Seq:
    """Opener stand-in returning scripted JSON bodies for successive requests."""

    def __init__(self, *bodies):
        self.bodies, self.calls = list(bodies), []

    def __call__(self, req, timeout):
        self.calls.append((req.get_method(), req.full_url, req.data))
        body = self.bodies.pop(0)
        if isinstance(body, Exception):
            raise body
        return FakeResponse(json.dumps(body).encode())


def job_transport(opener, **cfg):
    clock = iter(range(0, 10_000, 5))  # advances 5 s per call
    return tribe.JobTransport("https://owner-space.hf.space", "tok", replace(ON, **cfg), sleep=lambda s: None,
                              opener=opener, clock=lambda: next(clock))


def test_job_client_submits_then_polls_until_done():
    result = response(n=2)
    op = Seq({"job_id": "abc"}, {"status": "queued"}, {"status": "running"}, {"status": "done", "result": result})
    assert job_transport(op).post({"inputs": {"video_b64": "AA=="}}) == result
    assert [(m, u) for m, u, _ in op.calls] == [
        ("POST", "https://owner-space.hf.space/analyze"), ("GET", "https://owner-space.hf.space/report/abc"),
        ("GET", "https://owner-space.hf.space/report/abc"), ("GET", "https://owner-space.hf.space/report/abc")]
    assert json.loads(op.calls[0][2]) == {"inputs": {"video_b64": "AA=="}} and op.calls[1][2] is None
    preds, _ = parse_response(result, ON)
    assert preds.shape == (2, FSAVERAGE5_VERTICES)


def test_job_client_failures():
    assert job_transport(Seq({"job_id": "a"}, {"status": "failed", "error": "CUDA OOM"})).post({}) == {"error": "CUDA OOM"}
    with pytest.raises(TribeOutputError, match="no job_id"):
        job_transport(Seq({"nope": 1})).post({})
    with pytest.raises(TribeOutputError, match="unexpected job status"):
        job_transport(Seq({"job_id": "a"}, {"status": "weird"})).post({})
    with pytest.raises(TribeOutputError, match="no result"):
        job_transport(Seq({"job_id": "a"}, {"status": "done"})).post({})
    with pytest.raises(TribeRequestError, match="still running after 20 s"):
        job_transport(Seq({"job_id": "a"}, *[{"status": "running"}] * 10), job_wait_s=20).post({})
    with pytest.raises(TribeRequestError, match="job .* is gone: the Space no longer knows it"):
        job_transport(Seq({"job_id": "a"}, {"status": "running"}, http_error(404, "unknown job_id"))).post({})
    cold = Seq(http_error(503, "Space is starting"), {"job_id": "a"}, {"status": "done", "result": {"ok": 1}})
    assert job_transport(cold).post({}) == {"ok": 1}  # a sleeping Space answers 503 first, then accepts


def test_job_api_config_and_dispatch(video, tmp_path):
    with pytest.raises(ValueError, match="api"):
        TribeConfig(api="rest")
    with pytest.raises(ValueError, match="positive"):
        TribeConfig(poll_s=0)
    assert tribe.tribe_key(ctx_for(video), ON) == tribe.tribe_key(ctx_for(video), replace(ON, api="job"))  # same model, same key
    made = {}

    def factory(url, token):
        made["url"] = url
        return FakeTransport(response())

    art, _ = extract(ctx_for(video), replace(ON, api="job"), factory, env=ENV, root=tmp_path)
    assert art["status"] == AVAILABLE and made["url"] == ENV["TRIBE_ENDPOINT_URL"]


def test_space_app_end_to_end_over_real_http(monkeypatch):
    """The Space's FastAPI app on localhost, called by the real client (JobTransport over urllib)."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    mod = load_space_app()
    app = mod.create_app(lambda: (lambda req: response(n=2) if req["inputs"]["filename"] == "ok.mp4"
                                  else {"error": "RuntimeError: bad video"}))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    cfg = replace(ON, api="job", poll_s=0.05, retry_wait_s=0.1, job_wait_s=30)
    try:
        t = tribe.JobTransport(f"http://127.0.0.1:{port}", "tok", cfg)
        payload = {"inputs": {"video_b64": base64.b64encode(b"x").decode(), "filename": "ok.mp4"}, "parameters": {"language": "english"}}
        preds, segs = parse_response(t.post(payload), cfg)
        assert preds.shape == (2, FSAVERAGE5_VERTICES) and segs["starts"] == [0.0, 1.0]
        bad = t.post(payload | {"inputs": {"video_b64": "AA==", "filename": "bad.mp4"}})
        with pytest.raises(TribeOutputError, match="bad video"):
            parse_response(bad, cfg)
        with pytest.raises(TribeRequestError, match="HTTP 400"):
            t.post({"inputs": "nope"})
        with pytest.raises(TribeRequestError, match="HTTP 404"):
            t.request("GET", "/report/doesnotexist")
        assert t.request("GET", "/health")["model_loaded"] is True
    finally:
        server.should_exit = True
        thread.join(timeout=5)
