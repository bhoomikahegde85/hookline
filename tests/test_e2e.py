"""One-command pipeline (scripts/run_e2e.py) and the local review page. Subprocesses are faked; no model, GPU or network."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hookline.dropoffs import write_dropoffs
from hookline.review_page import REVIEW_FILE, build_review

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURVE = [1.0, 0.5, 0.3, 0.285, 0.27, 0.256, 0.243, 0.243, 0.2425, 0.2422, 0.2420]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRun:
    def __init__(self, fail_at: int | None = None, out: str = "", ttcc_status: str = "success", ttcc_reason: str | None = None, fail_script: str | None = None):
        self.calls, self.fail_at, self.out, self.fail_script = [], fail_at, out, fail_script
        self.ttcc_status, self.ttcc_reason = ttcc_status, ttcc_reason

    def __call__(self, cmd, cwd=None, capture_output=True, text=True):
        self.calls.append(cmd)
        if any(str(c).endswith("run_ttcc.py") for c in cmd):  # the real script writes ttcc_status.json and exits 0 even when TTCC was unavailable
            d = Path(cmd[cmd.index("--artifacts") + 1]) / "features" / Path(cmd[cmd.index("--input") + 1]).stem
            d.mkdir(parents=True, exist_ok=True)
            (d / "ttcc_status.json").write_text(json.dumps({"status": self.ttcc_status, "reason": self.ttcc_reason}))
        code = 1 if (self.fail_at == len(self.calls) or (self.fail_script and any(self.fail_script in str(c) for c in cmd))) else 0
        return subprocess.CompletedProcess(cmd, code, stdout=self.out if code else "done\n", stderr="")


def opts(**over):
    return dict(allow_remote=True, approve=None, siglip=False, skip_local=False, open_page=False, gemini_allow_remote=True) | over


def test_plan_orders_the_five_steps_and_passes_flags(tmp_path) -> None:
    e2e = load_script("run_e2e")
    video = tmp_path / "FC002.mp4"
    steps = e2e.plan(video, tmp_path / "root", **opts(approve="FC002", siglip=True))
    assert [s["name"].split(" ")[0] for s in steps] == ["features", "speech", "retention", "explanation", "music", "popularity", "scores:", "explain:", "segments"]
    assert steps[0]["cmd"][1].endswith("run_pipeline.py") and ".venv/" in steps[0]["cmd"][0]
    assert steps[1]["cmd"][1].endswith("run_speech_ocr.py") and ".venv-models/" in steps[1]["cmd"][0] and "--siglip" in steps[1]["cmd"]
    ttcc = steps[2]["cmd"]
    assert "--ttcc-remote" in ttcc and "--ttcc-allow-remote" in ttcc and ttcc[ttcc.index("--ttcc-approve") + 1] == "FC002"
    assert steps[3]["cmd"][1].endswith("run_gemini_edit.py") and not steps[3].get("optional") and steps[3]["skip"] is None  # Gemini is required
    assert steps[4]["cmd"][1].endswith("run_stems.py") and not steps[4].get("optional")  # Demucs music/voice levels: local, required
    assert steps[5]["cmd"][1].endswith("run_tribe.py") and not steps[5].get("optional")  # TRIBE popularity: a separate required step, never feeds TTCC
    assert steps[6]["cmd"][1].endswith("score_video.py") and not steps[6].get("optional") and steps[7]["cmd"][1].endswith("explain_video.py") and not steps[7].get("optional") \
        and steps[8]["cmd"][1].endswith("build_review.py") and "--open" not in steps[8]["cmd"]
    assert e2e.plan(video, tmp_path / "root", **opts(gemini_allow_remote=False))[3]["skip"]
    none = e2e.plan(video, tmp_path / "root", **opts(allow_remote=False, skip_local=True))
    assert none[0]["skip"] and none[1]["skip"] and "--allow-remote" in none[2]["skip"] and "--ttcc-approve" not in none[2]["cmd"]


def test_remote_is_never_run_without_permission(tmp_path) -> None:
    e2e = load_script("run_e2e")
    fake, logs = FakeRun(), []
    code = e2e.run_e2e(tmp_path / "v.mp4", tmp_path / "root", run=fake, log=logs.append, **opts(allow_remote=False))
    assert code == 1 and not any("run_ttcc" in " ".join(c) or "run_gemini" in " ".join(c) or "build_review" in " ".join(c) for c in fake.calls)  # nothing is uploaded, and nothing after the stop ran
    assert any("--allow-remote" in line for line in logs) and any("stopping" in line for line in logs)


def test_a_failed_step_stops_the_run_with_a_hint(tmp_path) -> None:
    e2e = load_script("run_e2e")
    fake, logs = FakeRun(fail_script="run_ttcc.py", out="TTCC unavailable - Space not reachable: HTTP 404"), []
    code = e2e.run_e2e(tmp_path / "v.mp4", tmp_path / "root", run=fake, log=logs.append, **opts())
    assert code == 1 and not any("run_gemini" in " ".join(c) or "build_review" in " ".join(c) for c in fake.calls)  # the later steps never ran
    assert any("Space is not ready" in line for line in logs) and any("FAILED" in line for line in logs)
    fake = FakeRun(fail_script="run_pipeline.py", out="boom")
    assert e2e.run_e2e(tmp_path / "v.mp4", tmp_path / "root", run=fake, log=lambda *_: None, **opts()) == 1 and len(fake.calls) == 1


def test_success_prints_the_review_page_path(tmp_path) -> None:
    e2e = load_script("run_e2e")
    video = tmp_path / "v.mp4"
    page = tmp_path / "root" / "features" / "v" / "review.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html></html>")
    logs = []
    assert e2e.run_e2e(video, tmp_path / "root", run=FakeRun(), log=logs.append, **opts()) == 0
    assert any(str(page.resolve()) in line for line in logs)


def test_ttcc_unavailable_is_a_failed_step_with_its_reason(tmp_path) -> None:
    """run_ttcc exits 0 when it did not run (video too short, not approved, ...); the pipeline must say so instead of reporting 'ok'."""
    e2e = load_script("run_e2e")
    fake, logs = FakeRun(ttcc_status="unavailable", ttcc_reason="duration 0.40 s is shorter than TTCC's trained range"), []
    code = e2e.run_e2e(tmp_path / "v.mp4", tmp_path / "root", run=fake, log=logs.append, **opts())
    assert code == 1 and not any("run_gemini" in " ".join(c) or "build_review" in " ".join(c) for c in fake.calls)  # nothing after the retention wave ran
    assert any("TTCC unavailable: duration 0.40 s is shorter" in line for line in logs)
    assert any("FAILED" in line and "retention" in line for line in logs)


# ---------------------------------------------------------------- the review page

def make_video_dir(tmp_path: Path, ocr_text: str = "SALE") -> tuple[Path, Path]:
    d = tmp_path / "features" / "v"
    d.mkdir(parents=True)
    video = tmp_path / "my clip & more.mp4"
    video.write_bytes(b"x")
    (d / "ttcc_retention.json").write_text(json.dumps({
        "status": "success", "video_id": "v", "source_sha256": "abc", "source_path": str(video), "cache_key": "k", "horizon_s": len(CURVE) - 1,
        "duration_s": 10.0, "partial_coverage": False, "model": {"checkpoint_repo": "m"}, "timestamps_s": list(range(len(CURVE))),
        "retention": CURVE, "inference": {"deviations_from_upstream_eval": ["attention kernel sdpa instead of flash_attn"]}}))
    (d / "ocr.json").write_text(json.dumps({"measured": True, "events": [
        {"event_id": 0, "text": ocr_text, "first_seen_s": 1.0, "visible_until_s": 2.0}]}))
    (d / "report.json").write_text(json.dumps({"scene_cut_times": [], "segments": [
        {"start_s": 0.0, "end_s": 10.0, "visual_status": "available", "motion_mean": 3.0}]}))
    write_dropoffs(d)
    return d, video


def test_review_page_is_self_contained_and_carries_the_data(tmp_path) -> None:
    d, video = make_video_dir(tmp_path)
    page = build_review(d, video)
    assert page.name == REVIEW_FILE
    html = page.read_text()
    assert not re.search(r"https?://", html.replace("http://www.w3.org/2000/svg", ""))  # the only URL in the page is the SVG namespace id
    assert "<script src" not in html and "<link" not in html and "XMLHttpRequest" not in html  # no external resource
    calls = re.findall(r"fetch\(([^,)]*)", html)                                              # the page talks only to its own local server (what-if), by relative /api/ path
    assert calls and all(c.strip().startswith(("'/api/", '"/api/')) for c in calls), calls
    assert "innerHTML" not in html and "document.write" not in html and "eval(" not in html  # data is rendered as text only
    assert video.resolve().as_uri() in html and "%20" in html  # file:// URL, percent-encoded
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert data["video_id"] == "v" and data["retention"] == CURVE and data["intervals"][0]["kind"] == "steep_drop"
    assert "not measured" in data["note"] or "not measured viewer behaviour" in data["note"]
    assert data["deviations"] == ["attention kernel sdpa instead of flash_attn"] and data["confidence_note"].startswith("none")
    for needle in ("Jump to", "Export edit decisions (JSON)", "Export edit decisions (CSV)", "Accept", "Reject", "Unsure",
                   "Play edited version", "Preview trim", "PREDICTED", "not been verified"):
        assert needle in html


def test_review_page_cannot_be_broken_by_hostile_text(tmp_path) -> None:
    d, video = make_video_dir(tmp_path, ocr_text='</script><img src=x onerror=alert(1)><!--')
    html = build_review(d, video).read_text()
    assert html.count("</script>") == 2  # only the page's own two script tags close; the OCR text cannot
    assert "<!--" not in html.split('type="application/json">')[1].split("</script>")[0]
    assert html.count("<script") == 2


def test_review_page_needs_a_prediction_and_survives_a_missing_video(tmp_path) -> None:
    empty = tmp_path / "x"
    empty.mkdir()
    assert build_review(empty) is None
    d, _ = make_video_dir(tmp_path)
    html = build_review(d, tmp_path / "gone.mp4").read_text()
    assert '"video_url": null' in html  # the page says the file was not found instead of failing


def test_build_review_script_reports_missing_prediction(tmp_path, capsys, monkeypatch) -> None:
    script = load_script("build_review")
    monkeypatch.setattr("sys.argv", ["build_review.py", "nope", "--artifacts", str(tmp_path)])
    assert script.main() == 1
    assert "no successful TTCC prediction" in capsys.readouterr().err


# ---------------------------------------------------------------- the page's own logic, in a real browser (skipped without Chrome)

CHROME = next((c for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"), shutil.which("chromium"))
               if c and Path(c).exists()), None)


@pytest.mark.skipif(CHROME is None, reason="Chrome not installed")
def test_page_logic_in_headless_chrome_with_trim_proposals(tmp_path) -> None:
    d, video = make_video_dir(tmp_path)
    (d / "recommendations.json").write_text(json.dumps({"recommendations": [
        {"rule": "static_stretch", "start_s": 0.0, "end_s": 2.0, "suggestion": "Consider a visual change.", "evidence": [{"type": "segment", "id": 0}]},
        {"rule": "quiet_stretch", "start_s": 0.5, "end_s": 1.5, "suggestion": "Trim the quiet part.", "evidence": []}]}))
    write_dropoffs(d)
    page = build_review(d, video)
    data = json.loads(page.read_text().split('type="application/json">')[1].split("</script>")[0].replace("<\\/", "</"))
    actions = [c["edit"]["action"] for iv in data["intervals"] for c in iv["candidate_causes"]]
    assert "trim" in actions  # the browser test below really exercises the trim path
    out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000", "--dump-dom", page.resolve().as_uri() + "#selftest"],
                         capture_output=True, text=True, timeout=120).stdout
    result = re.search(r'<pre id="selftest">(.*?)</pre>', out, re.S)
    assert result, "the page script did not run to the end"
    text = result.group(1)
    assert text.startswith("SELFTEST PASS"), text
    assert "accepted trim reaches the export" in text and "CSV has header + rows" in text  # the trim checks ran, not skipped


# ---------------------------------------------------------------- the local review server

def http_get(port: int, path: str, headers: dict | None = None, host: str | None = None, method: str = "GET"):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    conn.endheaders()
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), body


def test_review_server_serves_only_the_page_and_the_video_with_ranges(tmp_path) -> None:
    from hookline.review_server import start_server
    page = tmp_path / "review.html"
    page.write_text("<html>page</html>")
    video = tmp_path / "clip.mp4"
    video.write_bytes(bytes(range(256)) * 40)  # 10,240 bytes
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve")
    server, port = start_server(page, video)
    try:
        assert http_get(port, "/")[0:3:2] == (200, b"<html>page</html>")
        status, hdr, body = http_get(port, "/video")
        assert status == 200 and len(body) == 10240 and hdr["Accept-Ranges"] == "bytes" and hdr["Content-Type"] == "video/mp4"
        status, hdr, body = http_get(port, "/video", {"Range": "bytes=100-199"})
        assert status == 206 and body == video.read_bytes()[100:200] and hdr["Content-Range"] == "bytes 100-199/10240"
        status, hdr, body = http_get(port, "/video", {"Range": "bytes=10000-"})
        assert status == 206 and len(body) == 240
        assert http_get(port, "/video", {"Range": "bytes=-50"})[2] == video.read_bytes()[-50:]  # suffix range
        assert http_get(port, "/video", {"Range": "bytes=99999-"})[0] == 416
        assert http_get(port, "/video", method="HEAD")[1]["Content-Length"] == "10240"
        for bad in ("/secret.txt", "/../secret.txt", "/%2e%2e/secret.txt", "/video/../secret.txt", "/clip.mp4", "/ttcc_retention.json"):
            assert http_get(port, bad)[0] == 404, bad
        assert http_get(port, "/", host="evil.example.com")[0] == 403  # DNS-rebinding guard
        assert http_get(port, "/video", host=f"127.0.0.1:{port + 1}")[0] == 403
        assert server.server_address[0] == "127.0.0.1"  # not reachable from other machines
    finally:
        server.shutdown()


def test_review_server_without_a_video_serves_the_page_only(tmp_path) -> None:
    from hookline.review_server import start_server
    page = tmp_path / "review.html"
    page.write_text("x")
    server, port = start_server(page, None)
    try:
        assert http_get(port, "/")[0] == 200 and http_get(port, "/video")[0] == 404
    finally:
        server.shutdown()


@pytest.mark.skipif(CHROME is None or not (PROJECT_ROOT / "01-Hookline" / "face" / "videos" / "FC002.mp4").is_file(),
                    reason="Chrome or the FC002 reel not available")
def test_real_video_loads_in_chrome_over_the_local_server(tmp_path) -> None:
    from hookline.review_server import start_server
    real = PROJECT_ROOT / "01-Hookline" / "face" / "videos" / "FC002.mp4"
    d, _ = make_video_dir(tmp_path)
    page = build_review(d, real)
    server, port = start_server(page, real)
    try:
        out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=8000", "--dump-dom", f"http://127.0.0.1:{port}/#videocheck"],
                             capture_output=True, text=True, timeout=120).stdout
    finally:
        server.shutdown()
    m = re.search(r'<pre id="videocheck">(.*?)</pre>', out, re.S)
    assert m and "error=none" in m.group(1) and "duration=25." in m.group(1), m.group(1) if m else "page script did not run"


@pytest.mark.skipif(CHROME is None, reason="Chrome not installed")
def test_page_shows_scores_with_their_accuracy_in_chrome(tmp_path) -> None:
    d, video = make_video_dir(tmp_path)
    scores = {"schema": "hookline.hook_scores", "video_id": "v", "status": "ok", "note": "Scores are positions among reference reels.",
              "popularity_score": {"score": 62, "reference_reels": 39, "accuracy": {"model": "tribe_only", "evaluated": True, "auc": 0.6, "auc_ci95": [0.46, 0.68],
                                                                                   "verdict": "not shown to beat chance on these reels (the 95% interval includes 0.5)"}},
              "hook_score": {"score": None, "reason": "only 0 reference reels have TTCC predictions; 30 are needed"}, "retention_pct": {"1s": 43, "3s": 19, "5s": 14, "10s": 8}}
    (d / "hook_scores.json").write_text(json.dumps(scores))
    page = build_review(d, video)
    out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000", "--dump-dom", page.resolve().as_uri() + "#selftest"],
                         capture_output=True, text=True, timeout=120).stdout
    m = re.search(r'<pre id="selftest">(.*?)</pre>', out, re.S)
    assert m and m.group(1).startswith("SELFTEST PASS") and "three score gauges" in m.group(1), m.group(1) if m else "page script did not run"
    assert "not shown to beat chance" in out and "POPULARITY (TRIBE, experimental)" in out and "needs TTCC on the reference set" in out
    assert "will go viral" not in out.lower().replace("not a statement that a video will or will not go viral", "")


def test_skip_local_does_not_stop_the_run(tmp_path) -> None:
    e2e = load_script("run_e2e")
    logs = []
    e2e.run_e2e(tmp_path / "v.mp4", tmp_path, run=FakeRun(), log=logs.append, **opts(skip_local=True, gemini_allow_remote=True))
    assert not any("stopping" in line for line in logs) and any("retention" in line and "..." in line for line in logs)


def test_spaces_are_woken_at_the_start_and_before_each_remote_step(tmp_path) -> None:
    e2e = load_script("run_e2e")
    woken = []
    e2e.run_e2e(tmp_path / "v.mp4", tmp_path, run=FakeRun(), log=lambda _l: None, wake=woken.append, **opts(gemini_allow_remote=True))
    assert woken[0] == "start" and sorted(woken[1:]) == ["tribe", "ttcc"]  # the two remote steps run at the same time, so their order is not fixed


def test_independent_steps_really_run_at_the_same_time(tmp_path) -> None:
    import threading
    e2e = load_script("run_e2e")
    gate = threading.Barrier(2, timeout=10)  # speech and the retention step must be in flight together, or this raises BrokenBarrierError
    inner = FakeRun()

    def run(cmd, cwd=None, capture_output=True, text=True):
        if any(str(c).endswith(("run_speech_ocr.py", "run_ttcc.py")) for c in cmd):
            gate.wait()
        return inner(cmd, cwd=cwd, capture_output=capture_output, text=text)
    page = tmp_path / "root" / "features" / "v" / "review.html"
    page.parent.mkdir(parents=True)
    page.write_text("x")
    assert e2e.run_e2e(tmp_path / "v.mp4", tmp_path / "root", run=run, log=lambda _l: None, **opts(gemini_allow_remote=True)) == 0
