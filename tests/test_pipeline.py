"""End-to-end pipeline tests on small FFmpeg-generated clips with known event times."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from hookline.config import PipelineConfig
from hookline.pipeline import VideoInput, discover, process_video, run
from hookline.preprocess import preprocess

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg not installed")

SIZE = "180x320"
CFG = PipelineConfig()


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    vcodec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"]
    acodec = ["-c:a", "aac", "-b:a", "64k"]
    c: dict[str, Path] = {}

    # Black for 2 s then white for 2 s (hard cut at 2.0); audio silent until 1.0 then a tone.
    c["timed"] = d / "timed.mp4"
    ffmpeg(
        "-f", "lavfi", "-i", f"color=black:s={SIZE}:r=30:d=2", "-f", "lavfi", "-i", f"color=white:s={SIZE}:r=30:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a]volume=enable='lt(t,1)':volume=0[a]",
        "-map", "[v]", "-map", "[a]", *vcodec, *acodec, str(c["timed"]),
    )
    c["silent_video"] = d / "silent_video.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=30:d=3", "-an", *vcodec, str(c["silent_video"]))
    c["audio_only"] = d / "audio_only.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=330:duration=3", "-vn", *acodec, str(c["audio_only"]))
    c["av"] = d / "av.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=30:d=2.3", "-f", "lavfi", "-i", "sine=duration=2.3",
           *vcodec, *acodec, "-shortest", str(c["av"]))
    # Static slideshow: same grey background, a box that jumps position at 1.5 s (no colour change).
    c["slides"] = d / "slides.mp4"
    ffmpeg(
        "-f", "lavfi", "-i", f"color=gray:s={SIZE}:r=30:d=3",
        "-vf", "drawbox=x=10:y=10:w=60:h=60:color=white:t=fill:enable='lt(t,1.5)',"
               "drawbox=x=100:y=230:w=60:h=60:color=white:t=fill:enable='gte(t,1.5)'",
        "-an", *vcodec, str(c["slides"]),
    )
    c["very_short"] = d / "very_short.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=30:d=0.3", "-f", "lavfi", "-i", "sine=duration=0.3",
           *vcodec, *acodec, "-shortest", str(c["very_short"]))
    c["odd_fps"] = d / "odd_fps.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=12.5:d=2", "-an", *vcodec, str(c["odd_fps"]))
    # Audio stream starts 1.0 s after the video stream.
    c["late_audio"] = d / "late_audio.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=30:d=3", "-itsoffset", "1.0", "-f", "lavfi", "-i", "sine=duration=2",
           "-map", "0:v", "-map", "1:a", *vcodec, *acodec, str(c["late_audio"]))
    c["zero_bytes"] = d / "zero_bytes.mp4"
    c["zero_bytes"].write_bytes(b"")
    data = c["av"].read_bytes()
    c["truncated_index"] = d / "truncated_index.mp4"  # moov at the end: index lost
    c["truncated_index"].write_bytes(data[: len(data) // 3])
    faststart = d / "faststart_full.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=30:d=4", "-f", "lavfi", "-i", "sine=duration=4",
           *vcodec, *acodec, "-shortest", "-movflags", "+faststart", str(faststart))
    fs = faststart.read_bytes()
    c["truncated_media"] = d / "truncated_media.mp4"  # index intact, media data cut short
    c["truncated_media"].write_bytes(fs[: len(fs) // 2])
    return c


def process(clip: Path, root: Path, cfg: PipelineConfig = CFG, vid: str | None = None) -> dict:
    return process_video(VideoInput(vid or clip.stem, clip), cfg, root)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_timestamps_match_known_events(clips, tmp_path) -> None:
    r = process(clips["timed"], tmp_path)
    assert r["processing_status"] == "ok"
    assert r["scene_cut_times"] == [pytest.approx(2.0, abs=0.1)]
    assert r["opening"]["audio"]["first_non_silent_s"] == pytest.approx(1.0, abs=0.06)
    segs = r["segments"]
    dark = [s for s in segs if s["end_s"] <= 1.95]
    bright = [s for s in segs if s["start_s"] >= 2.05]
    assert all(s["brightness_mean"] < 20 for s in dark)
    assert all(s["brightness_mean"] > 235 for s in bright)
    assert all(s["near_silence_ratio"] == 1.0 for s in segs if s["end_s"] <= 0.95)
    assert all(s["near_silence_ratio"] == 0.0 for s in segs if 1.05 <= s["start_s"] and s["end_s"] <= 3.9)
    cut_seg = next(s for s in segs if s["scene_cut_at_start"])
    assert cut_seg["start_s"] == pytest.approx(2.0, abs=0.1) and cut_seg["scene_index"] == 1
    tmap = r["preprocess"]["timestamp_map"]
    assert tmap["video"]["max_deviation_from_cfr_grid"] < 0.01
    assert tmap["video"]["first_frame_time"] == pytest.approx(0.0, abs=1e-3)


def test_silent_video_has_absent_audio_with_missing_values(clips, tmp_path) -> None:
    r = process(clips["silent_video"], tmp_path)
    assert r["processing_status"] == "ok"
    assert r["modalities"]["audio"]["stream_status"] == "absent"
    assert all(s["audio_status"] == "absent" for s in r["segments"])
    assert all(s["visual_status"] == "available" for s in r["segments"])
    rows = read_csv(tmp_path / "features" / "silent_video" / "segments.csv")
    for row in rows:
        assert row["audio_status"] == "absent"
        for col in ("rms", "rms_dbfs", "near_silence_ratio", "onset_count", "clipping_ratio"):
            assert row[col] == ""  # missing, never 0
    assert r["opening"]["coverage"]["audio"] == {"absent": 1.0}
    assert r["opening"]["valid_windows"]["audio"] == 0


def test_audio_only(clips, tmp_path) -> None:
    r = process(clips["audio_only"], tmp_path)
    assert r["processing_status"] == "ok"
    assert r["modalities"]["visual"]["stream_status"] == "absent"
    assert all(s["visual_status"] == "absent" and s["brightness_mean"] is None for s in r["segments"])
    assert all(s["audio_status"] == "available" for s in r["segments"])
    assert all(s["scene_index"] is None for s in r["segments"])
    assert not (tmp_path / "features" / "audio_only" / "frames.csv").exists()
    assert r["preprocess"]["proxy_path"] is None


def test_video_with_audio_and_partial_final_segment(clips, tmp_path) -> None:
    r = process(clips["av"], tmp_path)
    assert r["processing_status"] == "ok"
    segs = r["segments"]
    assert segs[-1]["end_s"] == pytest.approx(r["duration_s"])
    assert segs[-1]["is_partial"] is True and segs[-1]["duration_s"] > 0
    assert all(s["duration_s"] > 0 for s in segs)
    assert all(a["end_s"] == b["start_s"] for a, b in zip(segs, segs[1:]))
    assert all(s["visual_status"] == "available" and s["audio_status"] == "available" for s in segs)
    assert all(s["brightness_mean"] is not None and s["rms"] is not None for s in segs)
    report = json.loads((tmp_path / "features" / "av" / "report.json").read_text())
    for key in ("video_id", "source_path", "extractor", "modalities", "segments", "opening", "warnings"):
        assert key in report
    assert report["extractor"]["config"]["window_seconds"] == 0.5


def test_static_slideshow_detects_layout_cut_without_colour_change(clips, tmp_path) -> None:
    r = process(clips["slides"], tmp_path)
    assert r["scene_cut_times"] == [pytest.approx(1.5, abs=0.1)]
    non_cut = [s for s in r["segments"] if s["motion_mean"] is not None]
    assert all(s["motion_mean"] < 0.5 for s in non_cut)  # the cut frame is not counted as motion


def test_very_short_clip(clips, tmp_path) -> None:
    r = process(clips["very_short"], tmp_path)
    assert r["processing_status"] == "ok"
    assert len(r["segments"]) == 1
    seg = r["segments"][0]
    assert seg["start_s"] == 0.0 and seg["end_s"] == pytest.approx(r["duration_s"])
    o = r["opening"]
    assert o["shorter_than_opening_window"] is True
    assert o["analysed_seconds"] == pytest.approx(r["duration_s"])
    assert o["valid_windows"] == {"visual": 1, "audio": 1}
    assert any("very short" in w for w in r["warnings"])


def test_unusual_frame_rate_keeps_timeline(clips, tmp_path) -> None:
    r = process(clips["odd_fps"], tmp_path)
    assert r["processing_status"] == "ok"
    assert r["probe"]["fps"] == 12.5
    tmap = r["preprocess"]["timestamp_map"]["video"]
    assert tmap["proxy_frames"] == 20
    assert tmap["max_deviation_from_cfr_grid"] < 0.01
    assert any("low_frame_rate" in w for w in r["warnings"])


def test_late_starting_audio_is_offset_not_silence(clips, tmp_path) -> None:
    r = process(clips["late_audio"], tmp_path)
    amap = r["preprocess"]["timestamp_map"]["audio"]
    assert amap["offset_seconds"] == pytest.approx(1.0, abs=0.03)
    segs = r["segments"]
    assert all(s["audio_status"] == "absent" and s["rms"] is None for s in segs if s["end_s"] <= 0.95)
    assert all(s["audio_status"] == "available" for s in segs if 1.05 <= s["start_s"] and s["end_s"] <= 2.95)
    assert r["opening"]["audio"]["first_non_silent_s"] == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("name", ["zero_bytes", "truncated_index"])
def test_unreadable_files_fail_with_uncertain_streams(clips, tmp_path, name) -> None:
    before = sha(clips[name])
    r = process(clips[name], tmp_path)
    assert r["processing_status"] == "failed"
    assert r["modalities"]["visual"]["stream_status"] == "uncertain"  # unknown, not absent
    assert r["modalities"]["audio"]["stream_status"] == "uncertain"
    assert r["segments"] == []
    assert any("inspection failed" in e for e in r["errors"])
    assert sha(clips[name]) == before


def test_truncated_media_is_partial_with_failed_windows(clips, tmp_path) -> None:
    r = process(clips["truncated_media"], tmp_path)
    assert r["processing_status"] == "partial", r["errors"]
    statuses = {s["visual_status"] for s in r["segments"]}
    assert "available" in statuses and "failed" in statuses
    early = [s for s in r["segments"] if s["end_s"] <= 1.0]
    assert all(s["visual_status"] == "available" for s in early)
    failed = [s for s in r["segments"] if s["visual_status"] == "failed"]
    assert all(s["brightness_mean"] is None for s in failed)
    assert any("extraction failure" in w for w in r["warnings"])


def test_repeat_processing_uses_cache_and_is_stable(clips, tmp_path) -> None:
    clip = clips["av"]
    before = sha(clip)
    first = process(clip, tmp_path)
    assert first["_cache"] == {"preprocess": False, "features": False}
    seg_csv = tmp_path / "features" / "av" / "segments.csv"
    proxy = Path(first["preprocess"]["proxy_path"])
    seg_bytes, proxy_mtime = seg_csv.read_bytes(), proxy.stat().st_mtime_ns

    second = process(clip, tmp_path)
    assert second["_cache"] == {"preprocess": True, "features": True}
    assert seg_csv.read_bytes() == seg_bytes
    assert proxy.stat().st_mtime_ns == proxy_mtime

    third = process(clip, tmp_path, cfg=replace(CFG, window_seconds=1.0))
    assert third["_cache"] == {"preprocess": True, "features": False}  # feature-only config change
    assert len(third["segments"]) < len(first["segments"])

    fourth = process(clip, tmp_path, cfg=replace(CFG, proxy_fps=5))
    assert fourth["_cache"] == {"preprocess": False, "features": False}
    assert Path(fourth["preprocess"]["cache_dir"]) != Path(first["preprocess"]["cache_dir"])
    assert sha(clip) == before


def test_changed_source_invalidates_cache(clips, tmp_path) -> None:
    src = tmp_path / "in" / "v.mp4"
    src.parent.mkdir()
    shutil.copy(clips["av"], src)
    a = preprocess(src, "v", tmp_path / "cache", CFG)
    shutil.copy(clips["very_short"], src)
    b = preprocess(src, "v", tmp_path / "cache", CFG)
    assert not b.cached and a.key != b.key
    assert b.timeline_duration < 1.0


def test_run_writes_summaries_and_never_touches_inputs(clips, tmp_path) -> None:
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in ("av", "silent_video", "audio_only", "zero_bytes", "truncated_media"):
        shutil.copy(clips[name], inputs / f"{name}.mp4")
    (inputs / "manifest.csv").write_text("video_id,file_path\nmy_av_clip,av.mp4\n")
    before = {p.name: sha(p) for p in inputs.iterdir()}

    summary = run([inputs], CFG, root=tmp_path / "artifacts", log=lambda *_: None)
    assert summary["counts"] == {"videos": 5, "ok": 3, "partial": 1, "failed": 1}
    ids = {v["video_id"] for v in summary["videos"]}
    assert "my_av_clip" in ids  # ID from the manifest
    feats = tmp_path / "artifacts" / "features"
    opening = read_csv(feats / "first5s_summary.csv")
    assert {r["video_id"] for r in opening} == ids
    silent = next(r for r in opening if r["video_id"] == "silent_video")
    assert silent["audio_valid_windows"] == "0" and silent["audio_status_coverage"] == "absent:1.0"
    assert silent["rms_dbfs"] == ""
    all_rows = read_csv(feats / "segments_all.csv")
    assert {r["video_id"] for r in all_rows} == ids - {"zero_bytes"}
    assert {p.name: sha(p) for p in inputs.iterdir()} == before
    assert not any(p.suffix == ".wav" for p in inputs.iterdir())


def test_discover_disambiguates_duplicate_stems(tmp_path) -> None:
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "clip.mp4").write_bytes(b"x")
    ids = sorted(v.video_id for v in discover([tmp_path]))
    assert ids == ["clip", "clip__2"]
