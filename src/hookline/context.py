"""Shared inputs for the speech and on-screen-text stages.

``MediaContext`` is everything those stages need from an existing features
report (stream status, timeline, cached WAV, segment rows), so they never
re-run preprocessing or visual/audio extraction.
"""

from __future__ import annotations

import math
import sys
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable

from .status import UNCERTAIN


class BackendUnavailable(RuntimeError):
    """A model backend could not be imported, loaded or found locally."""


class LazyBackend:
    """Create a model backend on first use, at most once (also remembers a failed load)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._obj: Any = None
        self._error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._obj is not None

    @property
    def error(self) -> str | None:
        return self._error

    def get(self) -> Any:
        if self._obj is None and self._error is None:
            try:
                self._obj = self._factory()
            except Exception as exc:
                self._error = str(exc) if isinstance(exc, BackendUnavailable) else f"{type(exc).__name__}: {exc}"
        if self._error is not None:
            raise BackendUnavailable(self._error)
        return self._obj


@dataclass
class MediaContext:
    video_id: str
    source_path: str
    source_sha256: str | None
    prep_key: str | None
    duration: float | None
    visual_stream: str = UNCERTAIN
    audio_stream: str = UNCERTAIN
    has_video: bool | None = None
    has_audio: bool | None = None
    audio_path: str | None = None
    audio_offset: float = 0.0
    video_span: list[float] | None = None
    audio_span: list[float] | None = None
    source_fps: float | None = None
    source_vfr: bool | None = None
    width: int | None = None  # display orientation (rotation applied)
    height: int | None = None
    processing_status: str | None = None
    segments: list[dict] = field(default_factory=list)
    opening: dict | None = None
    proxy_path: str | None = None  # cached analysis proxy (constant frame rate, original timeline)
    proxy_size: list[int] | None = None  # [width, height]
    proxy_fps: float | None = None
    scene_cut_times: list[float] = field(default_factory=list)

    @classmethod
    def from_report(cls, report: dict) -> "MediaContext":
        mods = report.get("modalities") or {}
        probe = report.get("probe") or {}
        prep = report.get("preprocess") or {}
        tmap = prep.get("timestamp_map") or {}
        video, audio = tmap.get("video") or {}, tmap.get("audio") or {}
        return cls(
            video_id=report["video_id"],
            source_path=report.get("source_path", ""),
            source_sha256=report.get("source_sha256"),
            prep_key=prep.get("key"),
            duration=report.get("duration_s"),
            visual_stream=(mods.get("visual") or {}).get("stream_status") or UNCERTAIN,
            audio_stream=(mods.get("audio") or {}).get("stream_status") or UNCERTAIN,
            has_video=probe.get("has_video"),
            has_audio=probe.get("has_audio"),
            audio_path=prep.get("audio_path"),
            audio_offset=audio.get("offset_seconds") or 0.0,
            video_span=video.get("expected_span"),
            audio_span=audio.get("expected_span"),
            source_fps=probe.get("fps"),
            source_vfr=probe.get("variable_frame_rate"),
            width=probe.get("width"),
            height=probe.get("height"),
            processing_status=report.get("processing_status"),
            segments=report.get("segments") or [],
            opening=report.get("opening"),
            proxy_path=prep.get("proxy_path"),
            proxy_size=prep.get("proxy_size"),
            proxy_fps=video.get("proxy_fps"),
            scene_cut_times=report.get("scene_cut_times") or [],
        )


def clean(v: Any, ndigits: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, ndigits) if math.isfinite(v) else None
    return v


def segment_locator(segments: list[dict]) -> Callable[[float], int | None]:
    """Map a timeline time to the ``segment_index`` of the existing segment containing it.
    Times at or after the last segment's end map to the last segment."""
    starts = [s["start_s"] for s in segments]

    def locate(t: float) -> int | None:
        if not segments:
            return None
        return segments[max(0, bisect_right(starts, t) - 1)]["segment_index"]

    return locate


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far (MB)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)


def covered_length(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    """Length of the union of ``intervals`` inside ``[start, end)``."""
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total
