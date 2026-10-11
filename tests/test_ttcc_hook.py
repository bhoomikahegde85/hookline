"""TTCC hook analysis (one start-anchored pass over the opening) and the sensitivity-check script. Inference is faked: these tests check
planning, metrics, bookkeeping and the variants, not the model."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import ttcc
from hookline.pipeline import VideoInput
from hookline.ttcc import RETENTION_FILE, SUCCESS, hook_metrics, plan_hook, run_ttcc
from test_ttcc import ENABLED, QUIET, FakeRunner

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("duration,hook_s,prompt,expected", [
    (25.0, 30, None, (25, None)), (30.0, 30, None, (30, None)), (30.4, 30, None, (30, None)), (31.0, 30, None, (30, 30.0)),
    (106.6, 30, None, (30, 30.0)), (143.2, 30, None, (30, 30.0)), (12.0, 30, None, (12, None)), (106.6, 25, None, (25, 25.0)),
    (106.6, 30, 60, (60, 30.0)), (25.0, 30, 45, (45, 25.0)), (3.0, 30, None, (None, None)), (None, 30, None, (None, None)),
])
def test_plan_hook(duration, hook_s, prompt, expected) -> None:
    T, shown, why = plan_hook(duration, hook_s, prompt)
    assert (T, shown) == expected and (why is None) == (T is not None)


def test_hook_metrics_reads_the_opening_only() -> None:
    r = [1.0, 0.6, 0.5, 0.45, 0.42, 0.40, 0.39, 0.385, 0.38, 0.378, 0.376, 0.3, 0.2]
    h = hook_metrics(r)
    assert (h["R1"], h["R3"], h["R5"], h["R10"]) == (0.6, 0.45, 0.40, 0.376)
    assert h["steepest_early_drop"] == {"from_s": 0, "to_s": 1, "drop": pytest.approx(0.4)} and h["early_window_s"] == 10  # the 0.1 drop at 12 s is not 'early'
    short = hook_metrics(r[:8])                                         # a 7 s video: R(10) does not exist, and is not extrapolated
    assert short["R10"] is None and short["R5"] == 0.40 and short["early_window_s"] == 7
    told_longer = hook_metrics(r, shown_s=4)                            # model saw 4 s: nothing after second 4 is read
    assert told_longer["R5"] is None and told_longer["R3"] == 0.45 and told_longer["seconds_shown"] == 4


@needs_ffmpeg
def test_hook_mode_is_one_pass_over_the_first_30_s(tmp_path) -> None:
    from test_ttcc import ffmpeg
    clip = tmp_path / "long.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x112:r=2:d=62.4", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(clip))
    runner = FakeRunner()
    rec = run_ttcc([VideoInput("long", clip)], tmp_path, ENABLED, runner, **QUIET)["records"][0]
    assert ENABLED.long_videos == "hook" and rec["status"] == SUCCESS
    jobs = [j for r in runner.requests for j in r["jobs"]]
    assert len(jobs) == 1 and jobs[0]["T"] == 30 and jobs[0]["video_end"] == 30.0 and jobs[0]["shrink_to_fit"] is True and "video_start" not in jobs[0]
    art = json.loads((tmp_path / "features" / "long" / RETENTION_FILE).read_text())
    assert art["policy"] == "hook" and not art.get("stitched") and art["timestamps_s"] == list(range(31))
    assert art["hook"]["R3"] == art["retention"][3] and art["hook"]["R10"] == art["retention"][10] and art["hook"]["seconds_shown"] == 30
    assert any("hook analysis" in d for d in art["inference"]["deviations_from_upstream_eval"])
    assert art["coverage"]["unpredicted_s"][0] == 30.0                 # the rest of the video is honestly 'not predicted'

    told = FakeRunner()
    run_ttcc([VideoInput("long", clip)], tmp_path / "p60", replace(ENABLED, hook_prompt_s=60), told, **QUIET)
    j = told.requests[0]["jobs"][0]
    assert (j["T"], j["video_end"], j["shrink_to_fit"]) == (60, 30.0, False)  # a sensitivity run never silently shrinks
    a60 = json.loads((tmp_path / "p60" / "features" / "long" / RETENTION_FILE).read_text())
    assert any("sensitivity run" in d for d in a60["inference"]["deviations_from_upstream_eval"]) and a60["hook"]["seconds_shown"] == 30
    from hookline.views_eval import ttcc_features
    assert ttcc_features(tmp_path / "p60" / "features" / "long")[0] is None   # never used as an evaluation feature
    assert ttcc_features(tmp_path / "features" / "long")[0]["ttcc_R3"] == art["hook"]["R3"]
    assert ttcc.cache_key("s", 60, ENABLED, {}, window_end=30.0) != ttcc.cache_key("s", 30, ENABLED, {}, window_end=30.0)


# ---------------------------------------------------------------- the sanity-check script

def _sanity():
    spec = importlib.util.spec_from_file_location("ttcc_sanity", PROJECT_ROOT / "scripts" / "ttcc_sanity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_dataset(tmp_path: Path, durations: dict[str, float]) -> Path:
    from test_ttcc import ffmpeg
    data = tmp_path / "data"
    for g in ("face", "faceless", "outliers"):
        (data / g / "videos").mkdir(parents=True)
        with (data / g / "metadata.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "instagram_handle", "follower_band", "views"])
            for vid, dur in durations.items():
                if (g == "face") == vid.startswith("FC"):
                    if g == "outliers":
                        continue
                    w.writerow([vid, f"h_{vid}", "50k-200k", 1000])
                    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=64x112:r=4:d={dur}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                           str(data / g / "videos" / f"{vid}.mp4"))
    return data


def _mean_luma(path: Path) -> float:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", "scale=16:16,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return sum(out) / len(out)


@needs_ffmpeg
def test_sanity_plan_builds_the_variants_without_sending_anything(tmp_path) -> None:
    mod = _sanity()
    data = fake_dataset(tmp_path, {"FC001": 8.0, "FL001": 35.0})
    out = tmp_path / "sanity"
    assert mod.main(["FC001", "FL001", "--data", str(data), "--out", str(out), "--plan-only", "--hook-s", "10"]) == 0
    run_dir = out / (out / "LATEST").read_text().strip() if (out / "LATEST").exists() else next(out.iterdir())
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["videos"]["FC001"]["seconds_shown"] == 8 and plan["videos"]["FL001"]["seconds_shown"] == 10
    assert plan["passes"] == 2 * 4 + 2 * 2                                # (original + 3 variants) + 2 told lengths, per reel
    v = run_dir / "variants"
    for name in ("FL001__reencoded", "FL001__black", "FL001__shuffled"):
        info, problem = ttcc.validate_video(v / f"{name}.mp4")
        assert problem is None and abs(info["duration_s"] - 10.0) < 0.6 and info["has_audio"] is not True
    assert _mean_luma(v / "FL001__black.mp4") < 20 < _mean_luma(v / "FL001__reencoded.mp4")
    order = json.loads((v / "FL001__shuffle_order.json").read_text())
    assert sorted(order) == list(range(10)) and order != list(range(10))
    with pytest.raises(SystemExit):
        mod.main(["FC001", "--data", str(data), "--out", str(out)])       # nothing is sent without --allow-remote
    with pytest.raises(SystemExit):
        mod.main(["NOPE", "--data", str(data), "--out", str(out), "--plan-only"])
    with pytest.raises(SystemExit):                                       # variants outside 01-Hookline/ still need their source approved
        mod.main(["FC001", "FL001", "--data", str(data), "--out", str(out), "--allow-remote", "--approve", "FC001"])


@needs_ffmpeg
def test_sanity_results_and_flags(tmp_path, monkeypatch) -> None:
    mod = _sanity()
    data = fake_dataset(tmp_path, {"FC001": 8.0, "FL001": 12.0})
    calls = []

    def fake_run(videos, root, cfg, log=print):
        calls.append((cfg.hook_prompt_s, cfg.approved_ids, [v.video_id for v in videos]))
        recs = []
        for v in videos:
            base = 0.30 if v.video_id.startswith("FC001") else 0.10        # the two reels differ; their variants copy them (content ignored)
            r = [1.0] + [max(0.05, 1.0 - base - 0.02 * t) for t in range(1, 13)]
            d = root / "features" / v.video_id
            d.mkdir(parents=True, exist_ok=True)
            (d / RETENTION_FILE).write_text(json.dumps({"status": "success", "retention": r, "hook": hook_metrics(r), "inference": {}}))
            recs.append({"video_id": v.video_id, "status": SUCCESS})
        return {"records": recs, "batch": {"runner_seconds": 10.0}}

    monkeypatch.setattr(mod, "run_ttcc", fake_run)
    out = tmp_path / "sanity"
    assert mod.main(["FC001", "FL001", "--data", str(data), "--out", str(out), "--allow-remote", "--approve", "FC001", "--approve", "FL001",
                     "--prompts", "45"]) == 0
    run_dir = out / (out / "LATEST").read_text().strip()
    res = json.loads((run_dir / "sanity_results.json").read_text())
    assert "FC001__black" in calls[0][1] and calls[1][0] == 45                # variants inherit their source's approval; prompt run told 45 s
    m = res["summary"]["mean_abs_differences"]
    assert m["black"]["abs_dR3"] == 0.0 and m["different_videos"]["abs_dR3"] == pytest.approx(0.2)
    assert any(f.startswith("black") and "not clearly larger" in f for f in res["summary"]["flags"])
    assert res["cost"]["runner_seconds"] == 20.0 and res["cost"]["estimated_usd_runner_time"] == round(20 / 3600 * 1.8, 2)
    assert (run_dir / "sanity_results.csv").read_text().startswith("video_id,variant,R1,R3,R5,R10")
    text = (run_dir / "sanity_results.json").read_text()
    assert "hf_" not in text and "AIza" not in text


# ---------------------------------------------------------------- codecs the runner cannot read (VP9: 4 of 110 reels failed with KeyError 'video_fps')

@needs_ffmpeg
def test_a_vp9_video_is_converted_to_h264_before_it_is_sent(tmp_path, monkeypatch) -> None:
    from test_ttcc import ffmpeg
    src = tmp_path / "vp9clip.webm"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x112:r=10:d=8", "-an", "-c:v", "libvpx-vp9", "-b:v", "200k", str(src))
    assert ttcc.video_codec(src) == "vp9"
    runner = FakeRunner()
    rec = run_ttcc([VideoInput("vp9clip", src)], tmp_path, ENABLED, runner, **QUIET)["records"][0]
    sent = Path(runner.requests[0]["jobs"][0]["video_path"])
    assert rec["status"] == SUCCESS and sent != src.resolve() and sent.name == "vp9clip.mp4" and ttcc.video_codec(sent) == "h264"
    art = json.loads((tmp_path / "features" / "vp9clip" / RETENTION_FILE).read_text())
    assert art["input"]["normalised"].startswith("vp9 -> h264") and art["source_path"] == str(src)             # the original stays the identity
    assert art["source_sha256"] == ttcc.file_sha256(src)
    h264 = tmp_path / "h264clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x112:r=10:d=8", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264))
    r2 = FakeRunner()
    run_ttcc([VideoInput("h264clip", h264)], tmp_path / "b", ENABLED, r2, **QUIET)
    assert Path(r2.requests[0]["jobs"][0]["video_path"]) == h264.resolve()                                       # readable codecs are sent as they are
    monkeypatch.setattr(ttcc, "normalised_copy", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    r3 = FakeRunner()
    rec3 = run_ttcc([VideoInput("vp9clip", src)], tmp_path / "c", ENABLED, r3, **QUIET)["records"][0]
    assert rec3["status"] == "failed" and "could not convert" in rec3["reason"] and r3.requests == []             # a failed conversion is reported, nothing is sent
