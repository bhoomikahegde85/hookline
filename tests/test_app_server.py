"""Upload page and job runner: safety checks and the flow, with a fake pipeline (no model, no Space, no network)."""

from __future__ import annotations

import http.client
import importlib.util
import json
import time
from pathlib import Path

import pytest

from hookline.app_server import ID_RE, STEP_RE, App, Job, start_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def req(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{port}"} | (headers or {})
    try:
        c.request(method, path, body=body, headers=h)
    except (BrokenPipeError, ConnectionResetError):  # the server answered (e.g. 413) before reading the whole body
        pass
    r = c.getresponse()
    data = r.read()
    out = (r.status, data, dict(r.getheaders()))
    c.close()
    return out


@pytest.fixture
def served(tmp_path):
    calls = []

    def runner(job: Job) -> int:
        calls.append(job.options | {"video": job.video.name})
        job.add("[1/3] features ...")
        job.add("[1/3] features: ok in 1 s")
        job.add("[2/3] retention (TTCC, remote Space) ...")
        feats = tmp_path / "features" / job.id
        feats.mkdir(parents=True, exist_ok=True)
        (feats / "review.html").write_text("<html>REVIEW " + job.id + "</html>")
        job.add("[2/3] retention (TTCC, remote Space): ok in 5 s")
        job.add("[3/3] explanation (Gemini): skipped (not requested)")
        return 0 if not job.options.get("fail") else 1

    app = App(tmp_path, runner, max_upload_mb=1)
    server, port = start_server(app, 0)
    yield app, port, calls
    server.shutdown()


H = {"X-Hookline": "1", "X-Filename": "my clip.mp4"}


def wait(port, job_id, until=("done", "failed")):
    for _ in range(100):
        st, data, _ = req(port, "GET", f"/api/job/{job_id}")
        j = json.loads(data)
        if j["status"] in until:
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_index_and_the_whole_flow(served) -> None:
    app, port, calls = served
    st, body, hdr = req(port, "GET", "/")
    assert st == 200 and b"Try a video" in body and b"model prediction" in body and b"will or will not go viral" in body and "text/html" in hdr["Content-Type"]
    st, body, _ = req(port, "POST", "/api/upload?remote=1&gemini=1", b"VIDEO" * 100, H)
    assert st == 200
    jid = json.loads(body)["job_id"]
    assert ID_RE.match(jid)
    j = wait(port, jid)
    assert j["status"] == "done" and j["review_url"] == f"/review/{jid}/" and [s["state"] for s in j["steps"]] == ["ok", "ok", "skipped"]
    assert calls == [{"remote": True, "gemini": True, "original_name": "my clip.mp4", "video": f"{jid}.mp4"}]
    assert (app.uploads / f"{jid}.mp4").read_bytes() == b"VIDEO" * 100                 # saved under the generated id, not the user's name
    st, body, _ = req(port, "GET", f"/review/{jid}/")
    assert st == 200 and f"REVIEW {jid}".encode() in body
    assert req(port, "GET", f"/review/{jid}")[0] == 301
    st, body, hdr = req(port, "GET", f"/review/{jid}/video", headers={"Range": "bytes=5-9"})
    assert st == 206 and body == b"VIDEO"[:5] and hdr["Content-Range"] == "bytes 5-9/500"
    assert req(port, "GET", f"/review/{jid}/video", headers={"Range": "bytes=9999-"})[0] == 416
    assert json.loads(req(port, "GET", "/api/recent")[1])["recent"][0]["id"] == jid


def test_uploads_always_run_the_full_pipeline_with_no_consent_options(served) -> None:
    app, port, calls = served
    jid = json.loads(req(port, "POST", "/api/upload", b"x" * 10, H)[1])["job_id"]
    wait(port, jid)
    assert calls[0]["gemini"] is True and calls[0]["remote"] is True


def test_uploads_need_consent_the_custom_header_and_a_known_file_type(served) -> None:
    app, port, calls = served
    assert req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, {"X-Filename": "a.mp4"})[0] == 403            # no custom header: a cross-site form cannot do this
    assert req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H | {"Origin": "http://evil.example"})[0] == 403
    assert req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H | {"X-Filename": "a.exe"})[0] == 415
    assert req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H | {"X-Filename": "../../etc/passwd"})[0] == 415
    assert req(port, "POST", "/api/other", b"x", H)[0] == 404
    assert req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * (2 * 2**20), H)[0] == 413                         # over the size limit (1 MB here)
    assert calls == [] and not list(app.uploads.iterdir())                                                    # nothing was kept or run


def test_path_names_are_never_trusted(served) -> None:
    app, port, calls = served
    jid = json.loads(req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H | {"X-Filename": "../../evil.mp4"})[1])["job_id"]
    wait(port, jid)
    assert [p.name for p in app.uploads.iterdir()] == [f"{jid}.mp4"] and calls[0]["original_name"] == "evil.mp4"
    for bad in ("/review/../../etc/passwd/", "/review/UP12/../../x/", "/api/job/..%2f..", "/uploads/", "/features/x/review.html", "/video"):
        assert req(port, "GET", bad)[0] in (404, 301, 403), bad
    assert req(port, "GET", "/review/UPzzzzzzz/")[0] == 404


def test_host_header_is_checked(served) -> None:
    app, port, _ = served
    assert req(port, "GET", "/", host="evil.example.com")[0] == 403
    assert req(port, "POST", "/api/upload?remote=1&gemini=1", b"x", H, host="evil.example.com")[0] == 403
    assert req(port, "GET", "/", host=f"localhost:{port}")[0] == 200


def test_one_analysis_at_a_time_and_failures_are_reported(served, tmp_path) -> None:
    app, port, calls = served
    release = []
    real = app.runner

    def slow(job):
        while not release:
            time.sleep(0.02)
        return real(job)
    app.runner = slow
    first = json.loads(req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H)[1])["job_id"]
    st, body, _ = req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H)
    assert st == 409 and "another analysis" in json.loads(body)["error"]
    release.append(1)
    wait(port, first)
    app.runner = lambda job: (_ for _ in ()).throw(RuntimeError("boom"))
    j = wait(port, json.loads(req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H)[1])["job_id"])
    assert j["status"] == "failed" and "RuntimeError: boom" in j["error"]


def test_step_lines_are_parsed() -> None:
    job = Job("UPabcd12", Path("x.mp4"), {})
    for line in ("[1/6] features ...", "[1/6] features: ok in 3 s", "[3/6] retention (TTCC, remote Space): FAILED (exit 1) in 2 s",
                 "[4/6] explanation (Gemini, uploads the video to Google): not run: the video is uploaded to Google only with --gemini-allow-remote", "noise"):
        job.add(line)
    assert [(s["n"], s["state"]) for s in job.view()["steps"]] == [(1, "ok"), (3, "failed"), (4, "skipped")] and STEP_RE.match("[2/6] x ...")


def test_serve_app_runner_passes_the_choices_to_the_pipeline(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("serve_app", PROJECT_ROOT / "scripts" / "serve_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}

    class FakeE2E:
        @staticmethod
        def run_e2e(video, root, **kw):
            seen.update(kw | {"video": video, "root": root})
            return 0
    runner = mod.make_runner(tmp_path, 10, FakeE2E)
    assert runner(Job("UPaaaa11", tmp_path / "v.mp4", {"gemini": True})) == 0
    assert seen["allow_remote"] is True and seen["approve"] is None and "gemini" not in seen and seen["gemini_allow_remote"] is True and seen["hook_s"] == 10 and seen["full_curve"] is True and seen["siglip"] is True


# ---------------------------------------------------------------- what-if

def test_whatif_endpoints_run_one_job_and_report_the_result(served) -> None:
    app, port, calls = served
    jid = json.loads(req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H)[1])["job_id"]
    wait(port, jid)
    seen = []
    app.whatif_runner = lambda job_id, trims: (seen.append((job_id, trims)), {"verdict": "no clear difference", "metrics": [], "caveats": ["c"]})[1]
    body = json.dumps({"trims": [{"start_s": 3, "end_s": 5}]}).encode()
    assert req(port, "POST", f"/api/whatif/{jid}", body, {"Content-Type": "application/json"})[0] == 403            # no custom header
    assert req(port, "POST", f"/api/whatif/{jid}", b"not json", {"X-Hookline": "1"})[0] == 400
    assert req(port, "POST", f"/api/whatif/{jid}", json.dumps({"trims": []}).encode(), {"X-Hookline": "1"})[0] == 400
    assert req(port, "POST", f"/api/whatif/{jid}", b"x" * 70000, {"X-Hookline": "1"})[0] == 413
    assert req(port, "POST", "/api/whatif/UPnotreal", body, {"X-Hookline": "1"})[0] == 404
    assert req(port, "GET", f"/api/whatif/{jid}")[0] == 404                                                          # nothing yet
    st, resp, _ = req(port, "POST", f"/api/whatif/{jid}", body, {"X-Hookline": "1"})
    assert st == 200 and json.loads(resp) == {"started": True}
    for _ in range(100):
        view = json.loads(req(port, "GET", f"/api/whatif/{jid}")[1])
        if view["status"] != "running":
            break
        time.sleep(0.05)
    assert view["status"] == "done" and view["result"]["verdict"] == "no clear difference" and seen == [(jid, [{"start_s": 3, "end_s": 5}])]
    app.whatif_runner = lambda job_id, trims: (_ for _ in ()).throw(ValueError("remove less than 80%"))
    req(port, "POST", f"/api/whatif/{jid}", body, {"X-Hookline": "1"})
    for _ in range(100):
        view = json.loads(req(port, "GET", f"/api/whatif/{jid}")[1])
        if view["status"] != "running":
            break
        time.sleep(0.05)
    assert view["status"] == "failed" and "ValueError: remove less than 80%" in view["error"]


def test_whatif_runner_cuts_the_video_predicts_it_and_compares(tmp_path) -> None:
    import shutil
    import subprocess
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("FFmpeg not installed")
    spec = importlib.util.spec_from_file_location("serve_app", PROJECT_ROOT / "scripts" / "serve_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jid = "UPtest0001"
    (tmp_path / "uploads").mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x112:r=10:d=20", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(tmp_path / "uploads" / f"{jid}.mp4")], check=True)
    feats = tmp_path / "features" / jid
    feats.mkdir(parents=True)
    orig = [1.0] + [max(0.05, 0.9 - 0.04 * t) for t in range(1, 21)]
    (feats / "ttcc_retention.json").write_text(json.dumps({"status": "success", "retention": orig, "duration_s": 20.0, "stitched": True}))
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        out = Path(cmd[cmd.index("--artifacts") + 1]) / "features" / f"{jid}_edit"
        out.mkdir(parents=True)
        edited = [1.0] + [max(0.05, 0.9 - 0.04 * t) for t in range(1, 16)]
        (out / "ttcc_retention.json").write_text(json.dumps({"status": "success", "retention": edited, "stitched": True}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    res = mod.make_whatif_runner(tmp_path, 10, fake_run)(jid, [{"start_s": 8, "end_s": 13}])
    cmd = cmds[0]
    assert cmd[cmd.index("--input") + 1].endswith(f"{jid}_edit.mp4") and "--ttcc-remote" in cmd and cmd[cmd.index("--window-s") + 1] == "10" and cmd[cmd.index("--long-videos") + 1] == "windows"
    assert abs(res["edited_duration_s"] - 15.0) < 0.6 and res["removed_s"] == 5.0 and res["trims"] == [[8.0, 13.0]]
    assert res["original_retention"] == orig and len(res["edited_retention"]) == 16 and any("approximate" in c for c in res["caveats"])
    assert json.loads((tmp_path / "whatif" / jid / "whatif.json").read_text())["video_id"] == jid                       # kept, so the page can show it again
    with pytest.raises(ValueError):
        mod.make_whatif_runner(tmp_path, 10, fake_run)(jid, [{"start_s": 0, "end_s": 19}])                              # would remove nearly everything


def test_a_saved_whatif_is_served_after_a_restart(served) -> None:
    app, port, _ = served
    jid = json.loads(req(port, "POST", "/api/upload?remote=1&gemini=1", b"x" * 10, H)[1])["job_id"]
    wait(port, jid)
    d = app.root / "whatif" / jid
    d.mkdir(parents=True)
    (d / "whatif.json").write_text(json.dumps({"verdict": "no clear difference", "metrics": []}))
    st, body, _ = req(port, "GET", f"/api/whatif/{jid}")
    assert st == 200 and json.loads(body)["result"]["verdict"] == "no clear difference"


def test_indented_output_lines_are_not_mistaken_for_steps() -> None:
    job = Job("UPabcd12", Path("x.mp4"), {})
    for line in ("[1/9] features ...", "      [1/1] UPabcd12: ok (visual=available)", "[1/9] features: ok in 7 s"):
        job.add(line)
    assert [(s["n"], s["name"], s["state"]) for s in job.view()["steps"]] == [(1, "features", "ok")]
