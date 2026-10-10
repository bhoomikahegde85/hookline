"""Cached preprocessing: analysis proxy video, mono WAV and a timestamp map.

Timeline convention: all times are seconds from the start of the original's
playback, i.e. ``original_pts - format_start_time``. The proxy keeps that
timeline (FFmpeg rebases output to the input start), so proxy frame times are
timeline times. WAV sample ``n`` sits at ``audio_offset + n / sample_rate``.

Artifacts live in ``<cache_root>/<video_id>/<key16>/`` where the key hashes the
source bytes and the preprocessing config. Originals are only ever read.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig, file_sha256, stable_hash
from .media import STATUS_OK, ProbeResult, find_tool, probe_video
from .provenance import ffmpeg_version
from .status import AVAILABLE, UNCERTAIN, stream_status

FFMPEG_TIMEOUT = 600
STATUS_FILE = "preprocess.json"


class ToolMissingError(RuntimeError):
    pass


@dataclass
class PreprocessResult:
    video_id: str
    source_path: str
    source_sha256: str | None
    key: str
    cache_dir: str
    status: str  # "ok" | "partial" | "failed"
    probe: dict[str, Any]
    timeline_duration: float | None = None
    visual_stream: str = UNCERTAIN
    audio_stream: str = UNCERTAIN
    proxy_path: str | None = None
    proxy_width: int | None = None
    proxy_height: int | None = None
    proxy_frame_times: list[float] = field(default_factory=list)
    audio_path: str | None = None
    audio_samples: int | None = None
    timestamp_map: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ffmpeg_version: str | None = None  # FFmpeg that made the proxy/WAV; None in caches that predate provenance
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cached")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreprocessResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _run(cmd: list[str], timeout: int = FFMPEG_TIMEOUT) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, "", f"timed out after {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr.strip()


def _encoder_args(ffmpeg: str) -> list[str]:
    code, out, _ = _run([ffmpeg, "-hide_banner", "-encoders"], timeout=20)
    if code == 0 and " libx264 " in out:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    return ["-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p"]


def proxy_dimensions(width: int | None, height: int | None, short_side: int) -> tuple[int, int] | None:
    """Scale so the shorter side is ``short_side`` (never upscaled), keeping aspect ratio, even dims."""
    if not width or not height:
        return None
    scale = min(1.0, short_side / min(width, height))
    even = lambda v: max(2, int(round(v * scale / 2)) * 2)  # noqa: E731
    return even(width), even(height)


def proxy_frame_times(ffprobe: str, proxy: Path) -> list[float]:
    code, out, _ = _run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(proxy)]
    )
    if code != 0:
        return []
    times = []
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        try:
            times.append(round(float(line), 6))
        except ValueError:
            continue
    return times


def wav_sample_count(path: Path) -> int | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes()
    except (wave.Error, EOFError, OSError):
        return None


def _offset(stream_start: float | None, format_start: float | None) -> float:
    if stream_start is None:
        return 0.0
    return max(0.0, round(stream_start - (format_start or 0.0), 6))


def _expected_span(offset: float, stream_duration: float | None, timeline: float) -> list[float]:
    end = offset + stream_duration if stream_duration else timeline
    return [offset, round(min(end, timeline) if timeline else end, 6)]


def cache_key(source_sha: str | None, source_path: Path, cfg: PipelineConfig) -> str:
    return stable_hash(source_sha or f"unhashable:{source_path}", cfg.preprocess_dict())


def load_cached(cache_dir: Path, key: str) -> PreprocessResult | None:
    status_file = cache_dir / STATUS_FILE
    if not status_file.exists():
        return None
    try:
        data = json.loads(status_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("key") != key:
        return None
    made_with = data.get("ffmpeg_version")
    if made_with is not None and made_with != ffmpeg_version():
        return None  # proxy/WAV encoded by a different FFmpeg: rebuild rather than mix
    for p in (data.get("proxy_path"), data.get("audio_path")):
        if p and not Path(p).is_file():
            return None
    result = PreprocessResult.from_dict(data)
    result.cached = True
    return result


def _atomic_ffmpeg(cmd_prefix: list[str], dest: Path) -> tuple[bool, str]:
    tmp = dest.with_name(dest.stem + ".tmp" + dest.suffix)
    code, _, err = _run(cmd_prefix + [str(tmp)])
    if code != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False, err or f"ffmpeg exit code {code}"
    tmp.replace(dest)
    return True, err


def preprocess(source: str | Path, video_id: str, cache_root: str | Path, cfg: PipelineConfig, force: bool = False) -> PreprocessResult:
    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ToolMissingError("ffmpeg and ffprobe are required (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg)")

    source = Path(source).resolve()
    sha = file_sha256(source) if source.is_file() else None
    key = cache_key(sha, source, cfg)
    cache_dir = (Path(cache_root) / video_id / key[:16]).resolve()
    if source == cache_dir or cache_dir in source.parents:
        raise ValueError("cache directory must not contain the source video")

    if not force:
        cached = load_cached(cache_dir, key)
        if cached is not None:
            return cached

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    probe = probe_video(source, ffprobe=ffprobe)
    result = PreprocessResult(
        video_id=video_id,
        source_path=str(source),
        source_sha256=sha,
        key=key,
        cache_dir=str(cache_dir),
        status="failed",
        probe=probe.to_dict(),
        ffmpeg_version=ffmpeg_version(ffmpeg),
    )
    if probe.status != STATUS_OK:
        # Inspection failure: presence of streams is unknown, not absent.
        result.errors.append(f"inspection failed: {probe.error}")
        result.visual_stream = result.audio_stream = UNCERTAIN
        return _finish(result, cache_dir)

    timeline = probe.duration_seconds
    if not timeline:
        ends = [
            _offset(probe.video_start_time, probe.format_start_time) + (probe.video_duration or 0),
            _offset(probe.audio_start_time, probe.format_start_time) + (probe.audio_duration or 0),
        ]
        timeline = max(ends) or None
    if not timeline or timeline <= 0:
        result.errors.append("zero or unknown duration")
        result.visual_stream = stream_status(probe.has_video, False if probe.has_video else None)
        result.audio_stream = stream_status(probe.has_audio, False if probe.has_audio else None)
        return _finish(result, cache_dir)
    result.timeline_duration = round(timeline, 6)
    result.warnings.extend(_file_warnings(probe, cfg))

    video_ok = _make_proxy(result, probe, ffmpeg, ffprobe, cache_dir, cfg) if probe.has_video else None
    audio_ok = _make_wav(result, probe, ffmpeg, cache_dir, cfg) if probe.has_audio else None
    result.visual_stream = stream_status(probe.has_video, video_ok)
    result.audio_stream = stream_status(probe.has_audio, audio_ok)

    v_off = _offset(probe.video_start_time, probe.format_start_time)
    a_off = _offset(probe.audio_start_time, probe.format_start_time)
    times = result.proxy_frame_times
    result.timestamp_map = {
        "timeline": "seconds from start of original playback = original_pts - format_start_time",
        "format_start_time": probe.format_start_time,
        "duration": result.timeline_duration,
        "video": None
        if not probe.has_video
        else {
            "expected_span": _expected_span(v_off, probe.video_duration, timeline),
            "source_fps": probe.fps,
            "source_variable_frame_rate": probe.variable_frame_rate,
            "proxy_fps": cfg.proxy_fps,
            "proxy_frames": len(times),
            "first_frame_time": times[0] if times else None,
            "last_frame_time": times[-1] if times else None,
            "max_deviation_from_cfr_grid": _grid_deviation(times, cfg.proxy_fps),
            "resolution_seconds": round(1 / cfg.proxy_fps, 6),
            "proxy_to_timeline": "identity (proxy frame time == timeline time)",
        },
        "audio": None
        if not probe.has_audio
        else {
            "expected_span": _expected_span(a_off, probe.audio_duration, timeline),
            "offset_seconds": a_off,
            "sample_rate": cfg.audio_sample_rate,
            "samples": result.audio_samples,
            "covered_span": [a_off, round(a_off + (result.audio_samples or 0) / cfg.audio_sample_rate, 6)]
            if result.audio_samples
            else None,
            "wav_to_timeline": "t = offset_seconds + sample_index / sample_rate",
        },
    }

    present = [s for s, has in ((result.visual_stream, probe.has_video), (result.audio_stream, probe.has_audio)) if has]
    if not any(s == AVAILABLE for s in present):
        result.status = "failed"
    elif all(s == AVAILABLE for s in present):
        result.status = "ok"
    else:
        result.status = "partial"
    return _finish(result, cache_dir)


def _grid_deviation(times: list[float], fps: int) -> float | None:
    if not times:
        return None
    t0 = times[0]
    return round(max(abs(t - (t0 + i / fps)) for i, t in enumerate(times)), 6)


def _file_warnings(probe: ProbeResult, cfg: PipelineConfig) -> list[str]:
    w = list(probe.warnings)
    if probe.has_video:
        if probe.width and probe.height and min(probe.width, probe.height) < cfg.low_resolution_short_side:
            w.append(f"low_resolution ({probe.width}x{probe.height}, short side < {cfg.low_resolution_short_side}px)")
        if probe.fps and probe.fps < cfg.low_fps:
            w.append(f"low_frame_rate ({probe.fps} fps)")
        if probe.fps and probe.fps > 60:
            w.append(f"high_frame_rate ({probe.fps} fps; proxy resampled to {cfg.proxy_fps} fps)")
        if probe.variable_frame_rate:
            w.append("possible_variable_frame_rate (proxy resampled to constant rate)")
    if probe.has_audio and probe.audio_sample_rate and probe.audio_sample_rate < cfg.audio_sample_rate:
        w.append(f"audio_sample_rate_below_target ({probe.audio_sample_rate} Hz upsampled to {cfg.audio_sample_rate} Hz)")
    return w


def _make_proxy(result: PreprocessResult, probe: ProbeResult, ffmpeg: str, ffprobe: str, cache_dir: Path, cfg: PipelineConfig) -> bool:
    dims = proxy_dimensions(probe.width, probe.height, cfg.proxy_short_side)
    scale = f"scale={dims[0]}:{dims[1]}:flags=area" if dims else f"scale=-2:{cfg.proxy_short_side}:flags=area"
    if not dims:
        result.warnings.append("source resolution unknown; proxy scaled to fixed height")
    proxy = cache_dir / "proxy.mp4"
    ok, err = _atomic_ffmpeg(
        [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", result.source_path, "-map", "0:v:0", "-an", "-sn", "-dn",
         "-vf", f"fps={cfg.proxy_fps},{scale},setsar=1", *_encoder_args(ffmpeg), "-g", str(cfg.proxy_fps * 2)],
        proxy,
    )
    if not ok:
        result.errors.append(f"proxy creation failed: {err[-300:]}")
        return False
    if err:
        result.warnings.append(f"decoder reported errors while creating proxy: {err[-300:]}")
    times = proxy_frame_times(ffprobe, proxy)
    if not times:
        result.errors.append("proxy has no decodable frames")
        proxy.unlink(missing_ok=True)
        return False
    pprobe = probe_video(proxy, ffprobe=ffprobe)
    result.proxy_path = str(proxy)
    result.proxy_width, result.proxy_height = pprobe.width, pprobe.height
    result.proxy_frame_times = times
    return True


def _make_wav(result: PreprocessResult, probe: ProbeResult, ffmpeg: str, cache_dir: Path, cfg: PipelineConfig) -> bool:
    wav = cache_dir / "audio.wav"
    ok, err = _atomic_ffmpeg(
        [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", result.source_path, "-map", "0:a:0", "-vn", "-sn", "-dn",
         "-ac", "1", "-ar", str(cfg.audio_sample_rate), "-c:a", "pcm_s16le", "-f", "wav"],
        wav,
    )
    if not ok:
        result.errors.append(f"audio extraction failed: {err[-300:]}")
        return False
    if err:
        result.warnings.append(f"decoder reported errors while extracting audio: {err[-300:]}")
    n = wav_sample_count(wav)
    if not n:
        result.errors.append("audio stream decoded to zero samples")
        wav.unlink(missing_ok=True)
        return False
    result.audio_path = str(wav)
    result.audio_samples = n
    return True


def _finish(result: PreprocessResult, cache_dir: Path) -> PreprocessResult:
    tmp = cache_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=2))
    tmp.replace(cache_dir / STATUS_FILE)
    return result
