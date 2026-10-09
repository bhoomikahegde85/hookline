"""Read-only probing of media files.

Unknown is represented as ``None``, never as ``False``: a file we could not
inspect has *unknown* audio, not *no* audio.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"})
VERY_SHORT_SECONDS = 1.0
LONG_FORM_SECONDS = 180.0
PROBE_TIMEOUT_SECONDS = 30
DECODE_TIMEOUT_SECONDS = 120

STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_UNCHECKED = "unchecked"


@dataclass
class ProbeResult:
    path: str
    status: str = STATUS_UNCHECKED
    backend: str = "none"
    error: str | None = None
    size_bytes: int | None = None
    container: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_video: bool | None = None
    has_audio: bool | None = None
    decode_errors: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Stream timing detail (seconds). Start times are container timestamps.
    format_start_time: float | None = None
    video_start_time: float | None = None
    video_duration: float | None = None
    video_frame_count: int | None = None
    variable_frame_rate: bool | None = None
    audio_start_time: float | None = None
    audio_duration: float | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None

    @property
    def orientation(self) -> str | None:
        if not self.width or not self.height:
            return None
        if self.height > self.width:
            return "vertical"
        return "horizontal" if self.width > self.height else "square"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["orientation"] = self.orientation
        return d


def find_tool(name: str) -> str | None:
    return shutil.which(name)


def find_video_files(roots: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
            found.add(root.resolve())
        elif root.is_dir():
            found.update(
                p.resolve()
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith(".")
            )
    return sorted(found)


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            return round(float(num) / float(den), 3) if float(den) else None
        return round(float(rate), 3)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_content_warnings(result: ProbeResult) -> None:
    if result.duration_seconds is not None:
        if result.duration_seconds < VERY_SHORT_SECONDS:
            result.warnings.append(f"very short (<{VERY_SHORT_SECONDS:g}s)")
        elif result.duration_seconds > LONG_FORM_SECONDS:
            result.warnings.append(f"longer than {LONG_FORM_SECONDS:g}s; not typical short-form")
    if result.has_video is False:
        result.warnings.append("no video stream")
    if result.has_audio is False:
        result.warnings.append("no audio stream (not negative evidence; analyse visuals only)")


def _probe_ffprobe(result: ProbeResult, path: Path, ffprobe: str) -> ProbeResult:
    result.backend = "ffprobe"
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        result.status, result.error = STATUS_ERROR, f"ffprobe timed out after {PROBE_TIMEOUT_SECONDS}s"
        return result
    if proc.returncode != 0:
        lines = [ln.replace(f"{path}: ", "") for ln in proc.stderr.strip().splitlines() if ln.strip()]
        result.status = STATUS_ERROR
        result.error = "unreadable or corrupt: " + (lines[-1] if lines else f"ffprobe exit code {proc.returncode}")
        return result
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        result.status, result.error = STATUS_ERROR, "ffprobe returned invalid JSON"
        return result

    streams = info.get("streams", [])
    fmt = info.get("format", {})
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    result.container = fmt.get("format_name")
    result.has_video = video is not None
    result.has_audio = audio is not None
    result.format_start_time = _to_float(fmt.get("start_time"))
    result.duration_seconds = _to_float(fmt.get("duration")) or _to_float((video or audio or {}).get("duration"))
    if result.duration_seconds is not None:
        result.duration_seconds = round(result.duration_seconds, 3)
    if video:
        result.video_codec = video.get("codec_name")
        result.width, result.height = video.get("width"), video.get("height")
        rotation = _rotation(video)
        if rotation in (90, 270) and result.width and result.height:
            result.width, result.height = result.height, result.width
        avg = _parse_rate(video.get("avg_frame_rate"))
        nominal = _parse_rate(video.get("r_frame_rate"))
        result.fps = avg or nominal
        if avg and nominal:
            result.variable_frame_rate = abs(avg - nominal) / nominal > 0.01
        result.video_start_time = _to_float(video.get("start_time"))
        result.video_duration = _to_float(video.get("duration"))
        try:
            result.video_frame_count = int(video["nb_frames"]) if video.get("nb_frames") else None
        except ValueError:
            result.video_frame_count = None
    if audio:
        result.audio_codec = audio.get("codec_name")
        result.audio_start_time = _to_float(audio.get("start_time"))
        result.audio_duration = _to_float(audio.get("duration"))
        try:
            result.audio_sample_rate = int(audio["sample_rate"]) if audio.get("sample_rate") else None
        except ValueError:
            result.audio_sample_rate = None
        result.audio_channels = audio.get("channels")
    if not streams:
        result.status, result.error = STATUS_ERROR, "no media streams found"
        return result
    result.status = STATUS_OK
    return result


def _rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags", {}) or {}
    if "rotate" in tags:
        try:
            return abs(int(tags["rotate"])) % 360
        except ValueError:
            return 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                return abs(int(side["rotation"])) % 360
            except (TypeError, ValueError):
                return 0
    return 0


def _probe_opencv(result: ProbeResult, path: Path) -> ProbeResult:
    try:
        import cv2  # type: ignore
    except ImportError:
        result.error = "neither ffprobe nor OpenCV available; install FFmpeg to inspect media"
        return result
    result.backend = "opencv"
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            result.status, result.error = STATUS_ERROR, "OpenCV could not open file (unreadable or corrupt)"
            return result
        ok, _ = cap.read()
        if not ok:
            result.status, result.error = STATUS_ERROR, "OpenCV could not decode a frame"
            return result
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or None
        result.fps = round(fps, 3) if fps else None
        result.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        result.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        result.duration_seconds = round(frames / fps, 3) if fps and frames else None
        result.has_video = True
        result.has_audio = None  # OpenCV cannot see audio streams
        result.status = STATUS_OK
        return result
    finally:
        cap.release()


def decode_check(path: Path, ffmpeg: str | None = None) -> str | None:
    """Fully decode the file and return decoder errors, or None if clean."""
    ffmpeg = ffmpeg or find_tool("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DECODE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return f"decode timed out after {DECODE_TIMEOUT_SECONDS}s"
    errors = proc.stderr.strip()
    if proc.returncode != 0 and not errors:
        errors = f"ffmpeg exit code {proc.returncode}"
    return errors[:500] or None


def probe_video(path: str | Path, ffprobe: str | None = None, deep: bool = False) -> ProbeResult:
    """Inspect a media file without modifying it. Never raises for bad files."""
    path = Path(path)
    result = ProbeResult(path=str(path))
    if not path.exists():
        result.status, result.error = STATUS_ERROR, "file not found"
        return result
    if not path.is_file():
        result.status, result.error = STATUS_ERROR, "not a regular file"
        return result
    result.size_bytes = path.stat().st_size
    if result.size_bytes == 0:
        result.status, result.error = STATUS_ERROR, "empty file (0 bytes)"
        return result

    ffprobe = ffprobe or find_tool("ffprobe")
    if ffprobe:
        _probe_ffprobe(result, path, ffprobe)
    else:
        _probe_opencv(result, path)

    if deep and result.status == STATUS_OK:
        result.decode_errors = decode_check(path)
        if result.decode_errors:
            result.warnings.append("decoder reported errors; file may be partially corrupt")
    if result.status == STATUS_OK:
        _add_content_warnings(result)
    return result
