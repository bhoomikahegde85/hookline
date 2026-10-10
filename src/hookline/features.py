"""Per-video timestamped feature extraction and output writing.

Outputs per video, in ``<features_root>/<video_id>/``:
- ``segments.csv``: one row per segment (schema: docs/feature_schema.md)
- ``frames.csv``: per-proxy-frame visual measurements
- ``report.json``: provenance, config, modality status, warnings, opening summary, segments
Missing values are written as empty cells / JSON null, never as 0.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .audio import AudioSeries, aggregate_audio, analyze_audio, load_wav, opening_audio
from .config import FEATURE_VERSION, PREPROCESS_VERSION, SCHEMA_VERSION, PipelineConfig, stable_hash
from .preprocess import PreprocessResult
from .provenance import feature_dependencies, runtime_info
from .segments import build_segments, overlap
from .status import ALL_STATUSES, AVAILABLE, FAILED, MEASURED_STATUSES
from .visual import FrameSeries, aggregate_visual, analyze_proxy, detect_cuts, opening_visual

SEGMENT_COLUMNS = [
    "video_id", "segment_index", "start_s", "end_s", "duration_s", "is_partial",
    "scene_index", "scene_cut_at_start", "scene_cut_count",
    "visual_status", "visual_coverage", "visual_frames", "brightness_mean", "contrast_std",
    "sharpness_lapvar", "noise_sigma", "motion_mean", "motion_max", "scene_change_max", "visual_warnings",
    "audio_status", "audio_coverage", "rms", "rms_dbfs", "peak_dbfs", "near_silence_ratio",
    "energy_variation_db", "onset_strength_mean", "onset_count", "clipping_ratio", "audio_warnings",
    "notes",
]
FRAME_COLUMNS = ["time_s", "brightness", "contrast", "sharpness", "noise", "motion", "hist_change", "struct_change", "is_cut"]
OPENING_NOTE = "Descriptive measurements of the opening only; not an engagement or retention prediction."


def feature_key(prep: PreprocessResult, cfg: PipelineConfig) -> str:
    """Preprocessing key (source bytes + proxy/WAV settings) + full config + the OpenCV, NumPy and
    FFmpeg versions that compute the features: a library upgrade recomputes features (from the
    cached proxy/WAV) instead of mixing old and new values."""
    return stable_hash(prep.key, cfg.feature_dict(), feature_dependencies())


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 6)
    return v


def _segment_rows(prep: PreprocessResult, visual: FrameSeries | None, audio: AudioSeries | None, cuts: list[float], cfg: PipelineConfig) -> list[dict]:
    tmap = prep.timestamp_map or {}
    v_span = (tmap.get("video") or {}).get("expected_span")
    a_span = (tmap.get("audio") or {}).get("expected_span")
    rows = []
    for seg in build_segments(prep.timeline_duration, cfg.window_seconds, cfg.min_segment_seconds, cuts):
        v = aggregate_visual(visual, seg.start, seg.end, prep.visual_stream, v_span, cfg, cuts)
        a = aggregate_audio(audio, seg.start, seg.end, prep.audio_stream, a_span, cfg)
        feature_status = {**v.pop("feature_status"), **a.pop("feature_status")}
        has_video_frames = visual is not None and v["visual_status"] in MEASURED_STATUSES
        if seg.cut_at_start is None and has_video_frames:
            feature_status["scene_cut_at_start"] = "not_applicable: first segment"
        row = {
            "video_id": prep.video_id,
            "segment_index": seg.index,
            "start_s": round(seg.start, 6),
            "end_s": round(seg.end, 6),
            "duration_s": round(seg.duration, 6),
            "is_partial": seg.is_partial,
            "scene_index": seg.scene_index if visual is not None else None,
            "scene_cut_at_start": seg.cut_at_start if has_video_frames else None,
            "scene_cut_count": len(seg.cut_times) if has_video_frames else None,
            **v,
            **a,
        }
        row = {k: _clean(val) for k, val in row.items()}
        row["notes"] = "; ".join(f"{k}={s}" for k, s in sorted(feature_status.items()))
        row["feature_status"] = feature_status
        rows.append(row)
    return rows


def _opening_summary(rows: list[dict], prep: PreprocessResult, visual: FrameSeries | None, audio: AudioSeries | None, cuts: list[float], cfg: PipelineConfig) -> dict:
    duration = prep.timeline_duration or 0.0
    end = min(cfg.opening_seconds, duration)
    summary: dict[str, Any] = {
        "window_s": [0.0, round(end, 6)],
        "analysed_seconds": round(end, 6),
        "shorter_than_opening_window": duration < cfg.opening_seconds,
        "segments": 0,
        "valid_windows": {"visual": 0, "audio": 0},
        "coverage": {},
        "visual": None,
        "audio": None,
        "note": OPENING_NOTE,
    }
    if end <= 0:
        return summary
    in_window = [r for r in rows if overlap(r["start_s"], r["end_s"], 0.0, end) > 0]
    summary["segments"] = len(in_window)
    for modality in ("visual", "audio"):
        seconds = {s: 0.0 for s in ALL_STATUSES}
        for r in in_window:
            seconds[r[f"{modality}_status"]] += overlap(r["start_s"], r["end_s"], 0.0, end)
        summary["coverage"][modality] = {s: round(sec / end, 4) for s, sec in seconds.items() if sec > 0}
        summary["valid_windows"][modality] = sum(1 for r in in_window if r[f"{modality}_status"] == AVAILABLE)
    if prep.visual_stream in MEASURED_STATUSES:
        summary["visual"] = _clean_dict(opening_visual(visual, cuts, end))
    if prep.audio_stream in MEASURED_STATUSES:
        summary["audio"] = _clean_dict(opening_audio(audio, end, cfg))
    return summary


def _clean_dict(d: dict | None) -> dict | None:
    if d is None:
        return None
    return {k: ([_clean(x) for x in v] if isinstance(v, list) else _clean(v)) for k, v in d.items()}


def extract(prep: PreprocessResult, cfg: PipelineConfig) -> dict:
    """Run feature extraction for a preprocessed video and return the report dict."""
    warnings = list(prep.warnings)
    errors = list(prep.errors)
    visual_stream, audio_stream = prep.visual_stream, prep.audio_stream

    visual = None
    if prep.proxy_path and visual_stream == AVAILABLE:
        try:
            visual = analyze_proxy(prep.proxy_path, prep.proxy_frame_times, prep.proxy_width, prep.proxy_height, cfg)
            errors.extend(visual.errors)
            if len(visual) == 0:
                visual, visual_stream = None, FAILED
                errors.append("no frames could be read from the proxy")
        except Exception as exc:  # keep going with other modalities
            visual, visual_stream = None, FAILED
            errors.append(f"visual extraction failed: {type(exc).__name__}: {exc}")

    audio = None
    if prep.audio_path and audio_stream == AVAILABLE:
        try:
            samples, sr = load_wav(prep.audio_path)
            offset = (prep.timestamp_map.get("audio") or {}).get("offset_seconds", 0.0)
            audio = analyze_audio(samples, sr, offset, cfg)
        except Exception as exc:
            audio, audio_stream = None, FAILED
            errors.append(f"audio extraction failed: {type(exc).__name__}: {exc}")

    prep_view = PreprocessResult.from_dict({**prep.to_dict(), "visual_stream": visual_stream, "audio_stream": audio_stream})
    cuts = detect_cuts(visual, cfg)
    rows = _segment_rows(prep_view, visual, audio, cuts, cfg) if prep.status != "failed" else []

    for modality in ("visual", "audio"):
        failed = sum(1 for r in rows if r[f"{modality}_status"] == FAILED)
        if failed:
            warnings.append(f"{failed} segment(s) with {modality} extraction failure inside the declared stream span")

    present = [s for s, has in ((visual_stream, prep.probe.get("has_video")), (audio_stream, prep.probe.get("has_audio"))) if has]
    if not rows:
        status = "failed"
    elif all(s == AVAILABLE for s in present) and not any(
        r["visual_status"] == FAILED or r["audio_status"] == FAILED for r in rows
    ):
        status = "ok"
    else:
        status = "partial"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": prep.video_id,
        "source_path": prep.source_path,
        "source_sha256": prep.source_sha256,
        "processing_status": status,
        "duration_s": prep.timeline_duration,
        "modalities": {
            "visual": {"stream_status": visual_stream, "has_stream": prep.probe.get("has_video")},
            "audio": {"stream_status": audio_stream, "has_stream": prep.probe.get("has_audio")},
            "not_extracted_in_this_phase": ["speech", "on_screen_text", "semantic_visual"],
        },
        "probe": prep.probe,
        "preprocess": {
            "key": prep.key,
            "cache_dir": prep.cache_dir,
            "proxy_path": prep.proxy_path,
            "proxy_size": [prep.proxy_width, prep.proxy_height] if prep.proxy_path else None,
            "audio_path": prep.audio_path,
            "timestamp_map": prep.timestamp_map,
        },
        "extractor": {
            "hookline_version": __version__,
            "preprocess_version": PREPROCESS_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_key": feature_key(prep, cfg),
            "config": cfg.to_dict(),
            "dependencies": feature_dependencies(),
            "preprocess_ffmpeg": prep.ffmpeg_version,  # None: proxy/WAV cache predates provenance recording
            "runtime": runtime_info(),
        },
        "scene_cut_times": cuts,
        "warnings": warnings,
        "errors": errors,
        "opening": _opening_summary(rows, prep_view, visual, audio, cuts, cfg),
        "segments": rows,
        "_frames": visual,
    }


def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    return v


def write_outputs(report: dict, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: FrameSeries | None = report.pop("_frames", None)
    paths = {"report": str(out_dir / "report.json"), "segments": str(out_dir / "segments.csv")}

    seg_tmp = out_dir / "segments.csv.tmp"
    with seg_tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEGMENT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in report["segments"]:
            w.writerow({k: _csv_value(row.get(k)) for k in SEGMENT_COLUMNS})
    seg_tmp.replace(out_dir / "segments.csv")

    frames_path = out_dir / "frames.csv"
    if frames is not None and len(frames):
        cuts = {round(t, 6) for t in report.get("scene_cut_times") or []}
        tmp = out_dir / "frames.csv.tmp"
        with tmp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FRAME_COLUMNS)
            for i in range(len(frames)):
                t = float(frames.times[i])
                vals = [
                    t, frames.brightness[i], frames.contrast[i], frames.sharpness[i], frames.noise[i],
                    frames.motion[i], frames.hist_change[i], frames.struct_change[i],
                ]
                w.writerow(["" if not math.isfinite(x) else round(float(x), 4) for x in vals] + [_csv_value(round(t, 6) in cuts)])
        tmp.replace(frames_path)
        paths["frames"] = str(frames_path)
    else:
        frames_path.unlink(missing_ok=True)

    report["outputs"] = paths
    tmp = out_dir / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out_dir / "report.json")
    return paths
