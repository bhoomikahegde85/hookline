from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hookline import media
from hookline.media import STATUS_ERROR, STATUS_OK, STATUS_UNCHECKED, find_video_files, probe_video

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg not installed")


def make_clip(path: Path, seconds: float = 1.0, audio: bool = True) -> Path:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=64x96:rate=10:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={seconds}", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


def test_missing_file_does_not_raise(tmp_path: Path) -> None:
    result = probe_video(tmp_path / "missing.mp4")
    assert result.status == STATUS_ERROR
    assert result.error == "file not found"
    assert result.has_audio is None  # unknown, not "no audio"


def test_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.mp4").write_bytes(b"")
    result = probe_video(tmp_path / "empty.mp4")
    assert result.status == STATUS_ERROR
    assert "empty" in result.error


def test_directory_is_not_a_file(tmp_path: Path) -> None:
    assert probe_video(tmp_path).status == STATUS_ERROR


def test_no_tools_available_degrades_to_unchecked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "x.mp4"
    clip.write_bytes(b"not really a video")
    monkeypatch.setattr(media, "find_tool", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)  # make `import cv2` fail
    result = probe_video(clip)
    assert result.status == STATUS_UNCHECKED
    assert "install FFmpeg" in result.error
    assert result.has_audio is None and result.has_video is None


def test_find_video_files_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.MOV").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.mp4").write_bytes(b"x")
    names = [p.name for p in find_video_files([tmp_path, tmp_path / "does-not-exist"])]
    assert names == ["a.mp4", "b.MOV"]


@needs_ffmpeg
def test_probe_valid_clip(tmp_path: Path) -> None:
    result = probe_video(make_clip(tmp_path / "ok.mp4", seconds=2))
    assert result.status == STATUS_OK, result.error
    assert result.has_audio is True and result.has_video is True
    assert result.width == 64 and result.height == 96
    assert result.orientation == "vertical"
    assert abs(result.duration_seconds - 2) < 0.3
    assert result.fps == 10


@needs_ffmpeg
def test_probe_silent_clip_reports_no_audio(tmp_path: Path) -> None:
    result = probe_video(make_clip(tmp_path / "silent.mp4", audio=False))
    assert result.status == STATUS_OK
    assert result.has_audio is False
    assert any("not negative evidence" in w for w in result.warnings)


@needs_ffmpeg
def test_probe_very_short_clip_warns(tmp_path: Path) -> None:
    result = probe_video(make_clip(tmp_path / "short.mp4", seconds=0.3))
    assert result.status == STATUS_OK
    assert any("very short" in w for w in result.warnings)


@needs_ffmpeg
def test_probe_truncated_clip_is_error(tmp_path: Path) -> None:
    good = make_clip(tmp_path / "good.mp4", seconds=2)
    bad = tmp_path / "bad.mp4"
    data = good.read_bytes()
    bad.write_bytes(data[: len(data) // 3])
    result = probe_video(bad)
    assert result.status == STATUS_ERROR
    assert "corrupt" in result.error
    assert good.read_bytes() == data  # inputs never modified
