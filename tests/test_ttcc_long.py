"""Videos shorter than, equal to and longer than one TTCC window (30 s), including 60 s and 82.5 s: planning, timestamps on the
original timeline, merging, caching, failures and the Space protocol. Inference is faked with a known hazard function, so these tests
check Hookline's windowing and bookkeeping, not model quality."""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import ttcc
from hookline.config import TTCC_CHECKPOINT_FILES, TtccConfig
from hookline.pipeline import VideoInput
from hookline.ttcc import FAILED, PLOT_FILE, RAW_FILE, RETENTION_FILE, STATUS_FILE, SUCCESS, UNAVAILABLE, run_ttcc
from test_ttcc import ENABLED as AUTO
from test_ttcc import PROBE_OK, QUIET
from test_ttcc_windows import true_hazard

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
DURATIONS = {"d25": 25.0, "d28": 28.0, "d30": 30.0, "d31": 31.0, "d60": 60.0, "d825": 82.5}


def head_values(start: float, artifact: float = 1.0) -> list[float]:
    """The 60 head outputs of a window starting at ``start`` s: hazards from the shared 'true' hazard, optionally inflated in the
    first 3 s of every window after the first (the 'fresh start' artifact)."""
    r, out = 1.0, []
    for tau in range(1, 61):
        r *= math.exp(-true_hazard(int(start) + tau) * (artifact if start > 0 and tau <= 3 else 1.0))
        out.append(r)
    return out


class WindowRunner:
    """Fake of the real runner: window-aware outputs, optional failures. ``fail_if(job)`` returns an error string or None."""

    def __init__(self, artifact=1.0, fail_if=None, ignore_window=False):
        self.artifact, self.fail_if, self.ignore_window = artifact, fail_if or (lambda job: None), ignore_window
        self.requests: list[dict] = []

    def python(self, cfg):
        return "fake"

    def probe(self, cfg):
        return PROBE_OK

    def infer(self, cfg, request, workdir):
        self.requests.append(request)
        out = []
        for job in request["jobs"]:
            error = self.fail_if(job)
            if error:
                out.append({"video_id": job["video_id"], "status": "failed", "stage": "inference", "error": error})
                continue
            start = job.get("video_start") or 0.0
            row = {"chat_template_kwargs": {"video_start": float(start), "video_end": float(job["video_end"])}} if job.get("video_end") is not None else {}
            if self.ignore_window:  # an outdated runner: always decodes from 0
                row = {"chat_template_kwargs": {"video_start": 0.0, "video_end": float(job["video_end"])}}
            raw = {"video_id": job["video_id"], "T": job["T"], "r_pred": head_values(start, self.artifact), "input_tokens": 600 * job["T"], "row": row,
                   "anchor": "</cot>", "inference_seconds": 1.0, "provenance": {"runner_version": "1", "model_load_seconds": 20.0, "device": "cuda:0"}}
            Path(job["raw_output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["raw_output"]).write_text(json.dumps(raw))
            out.append({"video_id": job["video_id"], "status": "ok", "raw_output": job["raw_output"]})
        return out

    @property
    def jobs(self) -> list[dict]:
        return [j for r in self.requests for j in r["jobs"]]


def oom_above(seconds: int):
    return lambda job: "OutOfMemoryError: CUDA out of memory" if job["T"] > seconds else None


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("ttcc_long_clips")
    out = {}
    for name, dur in DURATIONS.items():
        out[name] = d / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=64x112:r=2:d={dur}", "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(out[name])], check=True)
    return out


AUTO = replace(AUTO, long_videos="windows")  # these tests cover the (experimental) windowed policy; the default is now "hook"
ENABLED = replace(AUTO, window_mode="fixed")  # 30 s windows every 15 s; the auto sizing has its own tests below


def run(clip: Path, root: Path, runner, cfg: TtccConfig = ENABLED, **kw) -> dict:
    return run_ttcc([VideoInput(clip.stem, clip)], root, cfg, runner, **QUIET, **kw)["records"][0]


def art_of(root: Path, name: str) -> dict:
    return json.loads((root / "features" / name / RETENTION_FILE).read_text())


def truth(n: int) -> list[float]:
    r = [1.0]
    for t in range(1, n + 1):
        r.append(r[-1] * math.exp(-true_hazard(t)))
    return r


# ---------------------------------------------------------------- shorter than / equal to one window: unchanged single pass

@pytest.mark.parametrize("name,n", [("d25", 25), ("d30", 30)])
def test_up_to_one_window_is_a_single_native_pass(tmp_path, clips, name, n):
    runner = WindowRunner()
    rec = run(clips[name], tmp_path, runner)
    assert rec["status"] == SUCCESS
    assert len(runner.jobs) == 1 and runner.jobs[0]["T"] == n and runner.jobs[0].get("video_end") is None  # whole video, as before
    art = art_of(tmp_path, name)
    assert art["timestamps_s"] == list(range(n + 1)) and not art.get("stitched") and "windowing" not in art
    assert art["partial_coverage"] is False and art["retention"][1:] == pytest.approx(head_values(0)[:n])


# ---------------------------------------------------------------- longer than one window

@pytest.mark.parametrize("name,n,plan", [
    ("d31", 31, [(0, 30), (1, 31)]),
    ("d60", 60, [(0, 30), (15, 45), (30, 60)]),
    ("d825", 82, [(0, 30), (15, 45), (30, 60), (45, 75), (52, 82)]),  # 82.5 s rounds to 82 (as upstream's round(duration))
])
def test_longer_videos_are_windowed_and_cover_the_whole_video(tmp_path, clips, name, n, plan):
    runner = WindowRunner()
    rec = run(clips[name], tmp_path, runner)
    assert rec["status"] == SUCCESS and rec["windows"] == len(plan) and rec["windows_failed"] == 0
    jobs = runner.jobs
    assert [(j["video_start"], j["video_end"]) for j in jobs] == [(float(s), float(e)) for s, e in plan]
    assert all(j["T"] == e - s and j["shrink_to_fit"] is False for j, (s, e) in zip(jobs, plan))  # prompt T = window length
    art = art_of(tmp_path, name)
    assert art["timestamps_s"] == list(range(n + 1)) and art["horizon_s"] == n and art["stitched"] is True
    assert art["partial_coverage"] is False and art["coverage"]["unpredicted_s"] is None and art["coverage"]["predicted_until_s"] == n
    assert art["retention"] == pytest.approx(truth(n), abs=1e-9)      # consistent windows give back the underlying curve, second for second
    r = art["retention"]
    assert r[0] == 1.0 and all(b <= a + 1e-12 for a, b in zip(r, r[1:]))
    w = art["windowing"]
    assert w["complete"] is True and w["covered_until_s"] == n and [(x["start_s"], x["end_s"]) for x in w["windows"]] == plan
    assert len(w["per_second_coverage"]) == n and all(c for c in w["per_second_coverage"])  # every second is covered by some window
    assert art["out_of_trained_range"] is (n > 60) and "approximation" in w["limitation"].lower() and "not measured retention" in art["note"]
    assert any("stitched" in x for x in art["inference"]["deviations_from_upstream_eval"])
    svg = (tmp_path / "features" / name / PLOT_FILE).read_text()
    assert svg.count("<circle") == n + 1 and "stitched from overlapping windows" in svg


def test_artifact_at_each_window_start_does_not_become_a_jump_in_the_curve(tmp_path, clips):
    runner = WindowRunner(artifact=6.0)  # every later window predicts a steep opening, as a standalone 'ad' might
    run(clips["d60"], tmp_path, runner)
    r = art_of(tmp_path, "d60")["retention"]
    drops = [a - b for a, b in zip(r, r[1:])]
    t = truth(60)
    true_drops = [a - b for a, b in zip(t, t[1:])]
    for start in (15, 30):  # seconds start+1..start+3 sit right after a window boundary
        for k in range(start, start + 3):
            assert drops[k] < 2.6 * true_drops[k] + 1e-9   # vs 6x for a naive concatenation of the windows
    assert all(b <= a + 1e-12 for a, b in zip(r, r[1:])) and r[-1] > 0


def test_a_known_event_lands_on_its_original_timestamp(tmp_path, clips):
    class Spike(WindowRunner):
        def infer(self, cfg, request, workdir):
            out = super().infer(cfg, request, workdir)
            for job in request["jobs"]:  # an 'event' at absolute second 50 shows up in every window that contains it
                s = job.get("video_start") or 0.0
                p = Path(job["raw_output"])
                raw = json.loads(p.read_text())
                hz = []
                for tau in range(1, 61):
                    r_prev = 1.0 if tau == 1 else raw["r_pred"][tau - 2]
                    hz.append(-math.log(raw["r_pred"][tau - 1] / r_prev) + (0.6 if s + tau == 50 else 0.0))
                raw["r_pred"] = [math.exp(-sum(hz[:i + 1])) for i in range(60)]
                p.write_text(json.dumps(raw))
            return out

    run(clips["d60"], tmp_path, Spike())
    art = art_of(tmp_path, "d60")
    h = art["windowing"]["merged_hazard"]
    assert max(range(60), key=lambda i: h[i]) + 1 == 50                        # second 50 is the interval (49, 50]
    r = art["retention"]
    assert r[49] - r[50] == max(r[t - 1] - r[t] for t in range(15, 61))                  # the biggest one-second loss after the opening


# ---------------------------------------------------------------- caching and cost

def test_rerun_reuses_everything_and_a_new_stride_reuses_matching_windows(tmp_path, clips):
    runner = WindowRunner()
    run(clips["d60"], tmp_path, runner)
    assert len(runner.jobs) == 3
    again = run(clips["d60"], tmp_path, runner)
    assert again["cached"] is True and len(runner.jobs) == 3                   # nothing sent
    other = run(clips["d60"], tmp_path, runner, replace(ENABLED, stride_s=10))  # windows (0,30),(10,40),(20,50),(30,60): two are cached
    assert other["status"] == SUCCESS and [(j["video_start"], j["video_end"]) for j in runner.jobs[3:]] == [(10.0, 40.0), (20.0, 50.0)]
    forced = run(clips["d60"], tmp_path, runner, replace(ENABLED, stride_s=10), force=True)
    assert len(runner.jobs) == 3 + 2 + 4 and forced["status"] == SUCCESS


def test_windows_are_cached_individually_on_disk(tmp_path, clips):
    run(clips["d60"], tmp_path, WindowRunner())
    files = sorted(p.name for p in (tmp_path / "features" / "d60" / ttcc.WINDOWS_DIR).iterdir())
    assert files == ["00000-00030.json", "00015-00045.json", "00030-00060.json"]
    bundle = json.loads((tmp_path / "features" / "d60" / RAW_FILE).read_text())
    assert [(w["start"], w["end"]) for w in bundle["windows"]] == [(0, 30), (15, 45), (30, 60)]
    assert art_of(tmp_path, "d60")["raw_output"]["windows"] == 3


# ---------------------------------------------------------------- failures

def window_fails(start: float, error="RuntimeError: decode failed"):
    return lambda job: error if job.get("video_start") == start else None


def test_a_failed_middle_window_is_bridged_by_the_overlap_or_leaves_a_marked_gap(tmp_path, clips):
    bridged = run(clips["d825"], tmp_path / "a", WindowRunner(fail_if=window_fails(30.0)))       # (15,45) and (45,75) overlap the hole
    art = art_of(tmp_path / "a", "d825")
    assert bridged["status"] == SUCCESS and bridged["windows_failed"] == 1 and art["windowing"]["complete"] is False
    assert art["timestamps_s"] == list(range(83)) and art["partial_coverage"] is False
    assert art["windowing"]["failed_windows"][0]["start"] == 30 and "decode failed" in art["windowing"]["failed_windows"][0]["error"]
    assert "overlapping windows cover the whole video" in bridged["reason"]

    gap = run(clips["d60"], tmp_path / "b", WindowRunner(fail_if=window_fails(15.0)))
    art = art_of(tmp_path / "b", "d60")                                                          # (0,30) then (30,60): still no gap
    assert art["partial_coverage"] is False and gap["windows_failed"] == 1

    hole = run(clips["d825"], tmp_path / "c", WindowRunner(fail_if=lambda j: "boom" if j["video_start"] in (30.0, 45.0) else None))
    art = art_of(tmp_path / "c", "d825")                                                         # seconds 46..52 have no window
    assert hole["status"] == SUCCESS and art["partial_coverage"] is True and art["timestamps_s"] == list(range(46))
    assert art["coverage"]["unpredicted_s"] == [45.0, 82.5] and art["horizon_s"] == 45
    assert art["retention"] == pytest.approx(truth(45), abs=1e-9)                                # what is predicted is right; nothing is invented
    assert "2 window(s) failed" in hole["reason"]


def test_a_rerun_repeats_only_the_failed_windows(tmp_path, clips):
    flaky = WindowRunner(fail_if=lambda j: "boom" if j["video_start"] in (30.0, 45.0) else None)
    run(clips["d825"], tmp_path, flaky)
    fixed = WindowRunner()
    rec = run(clips["d825"], tmp_path, fixed)
    assert [j["video_start"] for j in fixed.jobs] == [30.0, 45.0] and rec["status"] == SUCCESS and rec["cached"] is False
    art = art_of(tmp_path, "d825")
    assert art["windowing"]["complete"] is True and art["partial_coverage"] is False and art["retention"] == pytest.approx(truth(82), abs=1e-9)


def test_without_the_first_window_there_is_no_curve_and_an_earlier_one_survives(tmp_path, clips):
    run(clips["d60"], tmp_path, WindowRunner())
    before = (tmp_path / "features" / "d60" / RETENTION_FILE).read_text()
    rec = run(clips["d60"], tmp_path, WindowRunner(fail_if=window_fails(0.0)), replace(ENABLED, stride_s=10), force=True)
    assert rec["status"] == FAILED and "no window starting at 0 s" in rec["reason"]
    assert (tmp_path / "features" / "d60" / RETENTION_FILE).read_text() == before
    st = json.loads((tmp_path / "features" / "d60" / STATUS_FILE).read_text())
    assert st["status"] == FAILED and st["prediction_is_current"] is False
    nothing = run(clips["d60"], tmp_path / "n", WindowRunner(fail_if=lambda j: "dead"))
    assert nothing["status"] == FAILED and "no window was predicted" in nothing["reason"]


def test_out_of_memory_replans_with_shorter_windows(tmp_path, clips):
    runner = WindowRunner(fail_if=oom_above(21))
    rec = run(clips["d60"], tmp_path, runner)
    assert rec["status"] == SUCCESS
    art = art_of(tmp_path, "d60")
    w = art["windowing"]
    assert [a["window_s"] for a in w["attempts"]] == [30, 21] and w["window_s"] == 21 and w["configured_window_s"] == 30
    assert all(x["end_s"] - x["start_s"] == 21 for x in w["windows"]) and w["windows"][-1]["end_s"] == 60
    assert art["timestamps_s"] == list(range(61)) and art["retention"] == pytest.approx(truth(60), abs=1e-9)
    assert w["attempts"][0]["failed"] == [[0, 30], [15, 45], [30, 60]]
    assert w["stride_s"] == 10 and w["configured_stride_s"] == 15                  # the stride scales with the window (21/30)


def test_shrinking_stops_at_the_minimum_window_and_can_be_turned_off(tmp_path, clips):
    hopeless = run(clips["d60"], tmp_path / "a", WindowRunner(fail_if=oom_above(8)))              # 30 -> 21 -> 14 -> 9 < min 10
    assert hopeless["status"] == FAILED and "out of memory" in hopeless["reason"].lower()
    st = json.loads((tmp_path / "a" / "features" / "d60" / STATUS_FILE).read_text())
    assert st["failed_windows"] and st["status"] == FAILED
    off = WindowRunner(fail_if=oom_above(21))
    rec = run(clips["d60"], tmp_path / "b", off, replace(ENABLED, auto_shrink_window=False))
    assert rec["status"] == FAILED and len({j["T"] for j in off.jobs}) == 1


def test_a_video_that_does_not_fit_in_one_pass_falls_back_to_windows(tmp_path, clips):
    runner = WindowRunner(fail_if=oom_above(20))                                    # 28 s fails as a single pass, 19 s windows fit
    rec = run(clips["d28"], tmp_path, runner)
    assert rec["status"] == SUCCESS
    art = art_of(tmp_path, "d28")
    assert art["stitched"] is True and art["timestamps_s"] == list(range(29)) and art["windowing"]["window_s"] == 19
    assert art["retention"] == pytest.approx(truth(28), abs=1e-9)
    assert runner.jobs[0]["T"] == 28 and runner.jobs[1]["video_start"] == 0.0     # first the whole video, then windows


def test_policies_and_limits(tmp_path, clips):
    legacy = run(clips["d60"], tmp_path / "a", WindowRunner(), replace(ENABLED, long_videos="first_60s"))
    assert legacy["status"] == SUCCESS and "windows" not in legacy                  # the older single-pass behaviour is still available
    big = WindowRunner()
    run(clips["d60"], tmp_path / "b", big, replace(ENABLED, window_s=60, stride_s=30))
    assert [(j["T"], j.get("video_start")) for j in big.jobs] == [(60, None)]       # a bigger GPU (window_s=60): one native pass
    skipped = run(clips["d825"], tmp_path / "c", WindowRunner(), replace(ENABLED, long_videos="skip"))
    assert skipped["status"] == UNAVAILABLE and "longer than" in skipped["reason"]
    for bad in (dict(window_s=61), dict(stride_s=31), dict(min_window_s=31), dict(stride_s=0), dict(warmup_s=-1), dict(long_videos="all")):
        with pytest.raises(ValueError):
            TtccConfig(enabled=True, **bad)


# ---------------------------------------------------------------- the runner and the Space

def test_runner_row_selects_the_window_of_the_file():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ttcc_infer_long", Path(__file__).resolve().parents[1] / "scripts" / "ttcc_infer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = mod.build_row({"video_id": "x", "video_path": "/v.mp4", "T": 30, "video_start": 45.0, "video_end": 75.0}, ttcc.PROMPTS)
    assert row["chat_template_kwargs"] == {"video_start": 45.0, "video_end": 75.0} and "30 seconds long" in row["messages"][1]["content"]
    assert mod.build_row({"video_id": "x", "video_path": "/v.mp4", "T": 30, "video_end": 30.0}, ttcc.PROMPTS)["chat_template_kwargs"]["video_start"] == 0.0

    class TooLong(ValueError):
        pass

    class Template:
        def encode(self, inputs):
            n = 600 * inputs["T"] + 200
            if n > 18000:
                raise TooLong(f"Current length of row({n}) is larger than the max_length(18000).")
            return {"n": n, "kw": inputs["chat_template_kwargs"]}

    class Inputs:
        @staticmethod
        def from_dict(row):
            return row

    job = {"video_id": "x", "video_path": "/v.mp4", "T": 40, "video_start": 20.0, "video_end": 60.0, "shrink_to_fit": True}
    row, enc, window = mod.encode_fitting(job, ttcc.PROMPTS, Template(), Inputs, TooLong, 18000)
    assert row["T"] == 29 and enc["kw"] == {"video_start": 20.0, "video_end": 49.0} and window["video_end_s"] == 49.0  # the shorter window still starts at 20 s


FAKE_MULTI_RUNNER = textwrap.dedent('''
    import json, sys
    cmd, args = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2]))
    req = json.load(open(args["--request"]))
    for job in req["jobs"]:
        if job["video_id"].endswith("w00015_00045"):
            open(args["--results"], "a").write(json.dumps({"video_id": job["video_id"], "status": "failed", "stage": "inference", "error": "CUDA out of memory"}) + "\\n")
            continue
        raw = {"video_id": job["video_id"], "T": job["T"], "r_pred": [0.9 - 0.01 * i for i in range(60)], "input_tokens": 1,
               "provenance": {"seen": [job.get("video_start"), job.get("video_end")], "bytes": len(open(job["video_path"], "rb").read())}}
        json.dump(raw, open(job["raw_output"], "w"))
        open(args["--results"], "a").write(json.dumps({"video_id": job["video_id"], "status": "ok", "raw_output": job["raw_output"]}) + "\\n")
''')


def test_space_handles_several_windows_of_one_upload(tmp_path):
    from test_ttcc_remote import load_space
    mod = load_space()
    script = tmp_path / "runner.py"
    script.write_text(FAKE_MULTI_RUNNER)
    svc = mod.TtccService(runner_cmd=[sys.executable, str(script)], repo_dir=tmp_path / "g", ckpt_dir=tmp_path / "c")
    svc.ready = True
    ck = {"repo_id": "liangyuch/ttcc-sft-qwen25omni-3b-v8", "revision": "1686d5f00408bfd8f6a32be12e71248b6f8ff447",
          "expected_files": {k: list(v) for k, v in TTCC_CHECKPOINT_FILES.items()}, "verify_weights_sha256": False}
    jobs = [{"video_id": f"v__w{s:05d}_{s + 30:05d}", "T": 30, "video_start": float(s), "video_end": float(s + 30), "shrink_to_fit": False} for s in (0, 15, 30)]
    body = {"request": {"settings": {}, "prompts": ttcc.PROMPTS, "checkpoint": ck, "jobs": jobs}, "video_b64": base64.b64encode(b"VIDEO").decode(), "filename": "v.mp4"}
    out = svc.handle(body)
    items = {i["video_id"]: i for i in out["items"]}
    assert len(items) == 3 and items[jobs[0]["video_id"]]["raw"]["provenance"] == {"seen": [0.0, 30.0], "bytes": 5}
    assert items[jobs[2]["video_id"]]["raw"]["provenance"]["seen"] == [30.0, 60.0]       # each window got its own start/end
    assert items[jobs[1]["video_id"]]["result"]["status"] == "failed" and "out of memory" in items[jobs[1]["video_id"]]["result"]["error"]
    too_many = body | {"request": body["request"] | {"jobs": jobs * 6}}
    assert "between 1 and" in svc.handle(too_many)["error"] and "bad request" in svc.handle({"request": {"jobs": jobs}, "video_b64": "AA=="})["error"]
    single = svc.handle(body | {"request": {k: v for k, v in body["request"].items() if k != "jobs"} | {"job": jobs[0]}})
    assert single["result"]["status"] == "ok" and "items" not in single                   # the original single-job shape is unchanged


def test_remote_client_sends_one_upload_for_all_windows_of_a_video(tmp_path, clips):
    from test_ttcc_remote import REMOTE, SpaceTransport, make_runner
    seen = []

    class Svc:
        def handle(self, payload):
            seen.append(payload)
            jobs = payload["request"].get("jobs") or [payload["request"]["job"]]
            items = [{"video_id": j["video_id"], "result": {"video_id": j["video_id"], "status": "ok", "raw_output": None},
                      "raw": {"video_id": j["video_id"], "T": j["T"], "r_pred": head_values(j.get("video_start") or 0.0), "input_tokens": 5,
                              "row": {"chat_template_kwargs": {"video_start": j.get("video_start") or 0.0, "video_end": j.get("video_end")}},
                              "provenance": {"model_load_seconds": 30.0}}} for j in jobs if j.get("video_start") != 30.0]
            return {"items": items}

    runner, _ = make_runner(Svc(), health={"probe": PROBE_OK, "checkpoint_ready": True})
    rec = run(clips["d60"], tmp_path, runner, replace(REMOTE, long_videos="windows", window_mode="fixed"))
    assert len(SpaceTransport.posted) == 1 and len(seen[0]["request"]["jobs"]) == 3          # one upload, three windows
    assert seen[0]["request"]["jobs"][1] == {"video_id": "d60__w00015_00045", "T": 30, "video_start": 15.0, "video_end": 45.0, "shrink_to_fit": False}
    assert rec["status"] == SUCCESS and rec["windows_failed"] == 1                           # (30,60) came back without a result; (0,30),(15,45) remain
    art = art_of(tmp_path, "d60")
    assert art["timestamps_s"] == list(range(46)) and "neither a raw output nor an error" in art["windowing"]["failed_windows"][0]["error"]


# ---------------------------------------------------------------- window size follows the video length (window_mode="auto")

def test_auto_window_size_by_length() -> None:
    from hookline.ttcc_windows import auto_window_s, window_params
    assert [auto_window_s(n, 30) for n in (5, 12, 25, 30)] == [5, 12, 25, 30]                      # up to the limit: the whole video
    assert [auto_window_s(n, 30) for n in (31, 45, 46, 60, 90, 91, 106, 600)] == [20, 20, 15, 15, 15, 10, 10, 10]
    assert auto_window_s(60, 12) == 12 and auto_window_s(100, 30, min_window_s=15) == 15            # bounded by the configured max and min
    assert all(5 <= auto_window_s(n, 30) <= 30 for n in range(5, 2000))
    assert window_params(106, 30, 15, 5.0, "auto") == (10, 5, 2.5)                                 # stride keeps the 50 % overlap; warm-up <= a quarter window
    assert window_params(60, 30, 15, 5.0, "auto") == (15, 8, 3.75) and window_params(40, 30, 15, 5.0, "auto") == (20, 10, 5.0)
    assert window_params(106, 30, 15, 5.0, "fixed") == (30, 15, 5.0) and window_params(106, 30, 15, 5.0, "auto", size=7) == (7, 4, 1.75)


@pytest.mark.parametrize("name,n,W,plan", [
    ("d31", 31, 20, [(0, 20), (10, 30), (11, 31)]),
    ("d60", 60, 15, [(0, 15), (8, 23), (16, 31), (24, 39), (32, 47), (40, 55), (45, 60)]),
])
def test_auto_mode_uses_shorter_windows_for_longer_videos(tmp_path, clips, name, n, W, plan) -> None:
    runner = WindowRunner()
    rec = run(clips[name], tmp_path, runner, AUTO)
    assert rec["status"] == SUCCESS and rec["windows"] == len(plan)
    assert [(j["video_start"], j["video_end"]) for j in runner.jobs] == [(float(s), float(e)) for s, e in plan]
    assert all(j["T"] == W for j in runner.jobs)                                                   # every window is a full window of the chosen size
    art = art_of(tmp_path, name)
    w = art["windowing"]
    assert w["window_mode"] == "auto" and w["window_s"] == W and w["configured_window_s"] == 30
    assert art["timestamps_s"] == list(range(n + 1)) and art["retention"] == pytest.approx(truth(n), abs=1e-9) and art["partial_coverage"] is False
    assert all(c for c in w["per_second_coverage"]) and w["warmup_s"] == min(5.0, max(1.0, W / 4))


def test_auto_mode_82s_video_and_the_unchanged_short_cases(tmp_path, clips) -> None:
    runner = WindowRunner()
    run(clips["d825"], tmp_path / "a", runner, AUTO)
    art = art_of(tmp_path / "a", "d825")
    assert art["windowing"]["window_s"] == 15 and art["windowing"]["stride_s"] == 8 and art["timestamps_s"] == list(range(83))
    assert art["retention"] == pytest.approx(truth(82), abs=1e-9) and runner.jobs[-1]["video_end"] == 82.0
    for name, n in (("d25", 25), ("d30", 30)):                                                      # up to 30 s: the full video, one pass, as before
        r2 = WindowRunner()
        run(clips[name], tmp_path / name, r2, AUTO)
        assert [(j["T"], j.get("video_start")) for j in r2.jobs] == [(n, None)] and "windowing" not in art_of(tmp_path / name, name)


def test_auto_mode_shrinks_further_on_out_of_memory_and_keys_differ_by_mode(tmp_path, clips) -> None:
    runner = WindowRunner(fail_if=oom_above(11))                                                    # 15 s windows fail; 10 s windows (0.7 x 15 = 10) fit
    rec = run(clips["d60"], tmp_path, runner, AUTO)
    art = art_of(tmp_path, "d60")
    assert rec["status"] == SUCCESS and [a["window_s"] for a in art["windowing"]["attempts"]] == [15, 10] and art["windowing"]["warmup_s"] == 2.5
    assert art["timestamps_s"] == list(range(61)) and art["retention"] == pytest.approx(truth(60), abs=1e-9)
    fixed = WindowRunner()
    run(clips["d60"], tmp_path, fixed, ENABLED)                                                     # another mode is a different request, not a cache hit
    assert len(fixed.jobs) == 3 and art_of(tmp_path, "d60")["windowing"]["window_mode"] == "fixed"
    with pytest.raises(ValueError):
        TtccConfig(enabled=True, window_mode="tiny")


# ---------------------------------------------------------------- review fixes

def test_an_outdated_runner_that_ignores_video_start_is_detected_and_never_cached(tmp_path, clips) -> None:
    stale = WindowRunner(ignore_window=True)
    rec = run(clips["d60"], tmp_path, stale)
    d = tmp_path / "features" / "d60"
    assert "outdated" in rec["reason"] and not (d / ttcc.WINDOWS_DIR / "00015-00045.json").exists() and not (d / ttcc.WINDOWS_DIR / "00030-00060.json").exists()
    art = art_of(tmp_path, "d60")                                       # only the first window (start 0) is honestly decoded by any runner
    assert art["partial_coverage"] is True and art["timestamps_s"] == list(range(31)) and art["windowing"]["failed_windows"][0]["stage"] == "protocol"
    fixed = WindowRunner()
    rec = run(clips["d60"], tmp_path, fixed)                             # a corrected runner repeats only what was refused
    assert [j["video_start"] for j in fixed.jobs] == [15.0, 30.0] and rec["status"] == SUCCESS
    assert art_of(tmp_path, "d60")["retention"] == pytest.approx(truth(60), abs=1e-9)
    assert ttcc.window_not_honoured({"row": {}}, 0, 30) and ttcc.window_not_honoured({}, 0, 30) and ttcc.window_not_honoured(None, 0, 30)


def test_a_batch_survives_one_video_falling_back_to_windows(tmp_path, clips) -> None:
    runner = WindowRunner(fail_if=oom_above(26))                         # d28 fails as a single pass; d25 fits
    out = run_ttcc([VideoInput("d28", clips["d28"]), VideoInput("d25", clips["d25"])], tmp_path, ENABLED, runner, **QUIET)["records"]
    assert [r["status"] for r in out] == [SUCCESS, SUCCESS]              # d25's already-finished raw output is not deleted under it
    assert art_of(tmp_path, "d25")["timestamps_s"] == list(range(26)) and art_of(tmp_path, "d28")["stitched"] is True
    assert not (tmp_path / "cache" / "_ttcc_runs").exists() or not list((tmp_path / "cache" / "_ttcc_runs").glob("*/raw"))


def test_a_fallback_to_windows_is_remembered_by_the_next_run(tmp_path, clips) -> None:
    runner = WindowRunner(fail_if=oom_above(20))
    run(clips["d28"], tmp_path, runner)
    sent = len(runner.jobs)
    again = run(clips["d28"], tmp_path, runner)
    assert again["status"] == SUCCESS and again["cached"] is True and len(runner.jobs) == sent      # no new single pass, no upload
    forced = run(clips["d28"], tmp_path, runner, force=True)
    assert forced["status"] == SUCCESS and runner.jobs[sent]["T"] == 28                              # --force really starts over
