"""Batch pipeline: discover inputs, preprocess (cached), extract features, write outputs.

Run-level outputs in ``<artifacts>/features/``:
- ``<video_id>/segments.csv``, ``<video_id>/frames.csv``, ``<video_id>/report.json``
- ``segments_all.csv``: all segment rows of this run
- ``first5s_summary.csv``: one row per video describing the opening window
- ``run_summary.json``: per-video processing status, counts, config
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import PipelineConfig
from .features import SEGMENT_COLUMNS, _csv_value, extract, feature_key, write_outputs
from .manifest import read_manifest, resolve_file_path
from .media import find_video_files
from .paths import artifacts_dir, raw_dir, sample_dir
from .preprocess import preprocess

OPENING_COLUMNS = [
    "video_id", "processing_status", "duration_s", "analysed_seconds", "shorter_than_opening_window", "segments",
    "visual_valid_windows", "visual_available_coverage", "visual_status_coverage",
    "audio_valid_windows", "audio_available_coverage", "audio_status_coverage",
    "scene_cuts", "first_scene_cut_s", "brightness_mean", "motion_mean",
    "rms_dbfs", "near_silence_ratio", "first_non_silent_s", "onset_count",
]


@dataclass
class VideoInput:
    video_id: str
    path: Path
    manifest_row: dict[str, str] | None = None


def default_inputs() -> list[Path]:
    return [raw_dir(), sample_dir()]


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "video"


def discover(inputs: Iterable[str | Path]) -> list[VideoInput]:
    """Find videos in directories (recursively) or explicit files.

    IDs come from a ``manifest.csv`` in the input directory when it lists the
    file, otherwise from the file stem. Duplicate IDs get a numeric suffix.
    """
    found: dict[Path, VideoInput] = {}
    for item in inputs:
        item = Path(item).expanduser()
        manifest_ids: dict[Path, tuple[str, dict[str, str]]] = {}
        if item.is_dir():
            for manifest in sorted(item.rglob("manifest.csv")):
                try:
                    _, rows = read_manifest(manifest)
                except Exception:
                    continue
                for row in rows:
                    if row.get("video_id") and row.get("file_path"):
                        manifest_ids[resolve_file_path(manifest, row["file_path"]).resolve()] = (row["video_id"], row)
            paths = find_video_files([item])
        elif item.is_file():
            paths = [item.resolve()]
        else:
            continue
        for p in paths:
            if p in found:
                continue
            vid, row = manifest_ids.get(p, (p.stem, None))
            found[p] = VideoInput(_safe_id(vid), p, row)

    used: dict[str, int] = {}
    out = []
    for v in sorted(found.values(), key=lambda v: (v.video_id, str(v.path))):
        n = used.get(v.video_id, 0) + 1
        used[v.video_id] = n
        if n > 1:
            v.video_id = f"{v.video_id}__{n}"
        out.append(v)
    return out


def _load_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def process_video(item: VideoInput, cfg: PipelineConfig, root: Path, force: bool = False) -> dict:
    cache_root, features_root = root / "cache", root / "features"
    prep = preprocess(item.path, item.video_id, cache_root, cfg, force=force)
    out_dir = features_root / item.video_id
    report_path = out_dir / "report.json"

    existing = None if force else _load_report(report_path)
    if (
        prep.cached
        and existing
        and existing.get("extractor", {}).get("feature_key") == feature_key(prep, cfg)
        and existing.get("source_sha256") == prep.source_sha256
    ):
        existing["_cache"] = {"preprocess": True, "features": True}
        return existing

    report = extract(prep, cfg)
    write_outputs(report, out_dir)
    report["_cache"] = {"preprocess": prep.cached, "features": False}
    return report


def _opening_row(report: dict) -> dict[str, Any]:
    o = report.get("opening") or {}
    v, a = o.get("visual") or {}, o.get("audio") or {}
    cov = o.get("coverage") or {}
    status_cov = lambda m: "|".join(f"{k}:{val}" for k, val in sorted((cov.get(m) or {}).items()))  # noqa: E731
    return {
        "video_id": report["video_id"],
        "processing_status": report["processing_status"],
        "duration_s": report.get("duration_s"),
        "analysed_seconds": o.get("analysed_seconds"),
        "shorter_than_opening_window": o.get("shorter_than_opening_window"),
        "segments": o.get("segments"),
        "visual_valid_windows": (o.get("valid_windows") or {}).get("visual"),
        "visual_available_coverage": (cov.get("visual") or {}).get("available", 0.0) if cov else None,
        "visual_status_coverage": status_cov("visual"),
        "audio_valid_windows": (o.get("valid_windows") or {}).get("audio"),
        "audio_available_coverage": (cov.get("audio") or {}).get("available", 0.0) if cov else None,
        "audio_status_coverage": status_cov("audio"),
        "scene_cuts": v.get("scene_cuts"),
        "first_scene_cut_s": v.get("first_scene_cut_s"),
        "brightness_mean": v.get("brightness_mean"),
        "motion_mean": v.get("motion_mean"),
        "rms_dbfs": a.get("rms_dbfs"),
        "near_silence_ratio": a.get("near_silence_ratio"),
        "first_non_silent_s": a.get("first_non_silent_s"),
        "onset_count": a.get("onset_count"),
    }


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _csv_value(row.get(k)) for k in columns})
    tmp.replace(path)


def run(
    inputs: Iterable[str | Path] | None = None,
    cfg: PipelineConfig | None = None,
    root: str | Path | None = None,
    force: bool = False,
    limit: int | None = None,
    log=print,
) -> dict:
    cfg = cfg or PipelineConfig()
    root = Path(root) if root else artifacts_dir()
    inputs = [Path(p) for p in (inputs or default_inputs())]
    videos = discover(inputs)[: limit or None]
    features_root = root / "features"
    features_root.mkdir(parents=True, exist_ok=True)

    records, reports = [], []
    for i, item in enumerate(videos, 1):
        try:
            report = process_video(item, cfg, root, force=force)
        except Exception as exc:  # isolate per-video failures
            report = {
                "video_id": item.video_id,
                "source_path": str(item.path),
                "processing_status": "failed",
                "errors": [f"pipeline error: {type(exc).__name__}: {exc}"],
                "warnings": [],
                "segments": [],
                "modalities": {},
                "_cache": {"preprocess": False, "features": False},
            }
        cache = report.pop("_cache", {})
        mods = report.get("modalities") or {}
        report_path = features_root / item.video_id / "report.json"
        rec = {
            "video_id": item.video_id,
            "source_path": str(item.path),
            "processing_status": report["processing_status"],
            "duration_s": report.get("duration_s"),
            "segments": len(report.get("segments") or []),
            "visual_stream": (mods.get("visual") or {}).get("stream_status"),
            "audio_stream": (mods.get("audio") or {}).get("stream_status"),
            "warnings": len(report.get("warnings") or []),
            "errors": report.get("errors") or [],
            "cached_preprocess": cache.get("preprocess"),
            "cached_features": cache.get("features"),
            "report": str(report_path) if report.get("schema_version") and report_path.exists() else None,
        }
        records.append(rec)
        reports.append(report)
        cached = "cached" if cache.get("features") else ("prep-cached" if cache.get("preprocess") else "new")
        log(
            f"[{i}/{len(videos)}] {item.video_id}: {rec['processing_status']} "
            f"(visual={rec['visual_stream']}, audio={rec['audio_stream']}, segments={rec['segments']}, {cached})"
            + (f" error: {rec['errors'][0][:120]}" if rec["errors"] and rec["processing_status"] != "ok" else "")
        )

    counts = {s: sum(1 for r in records if r["processing_status"] == s) for s in ("ok", "partial", "failed")}
    _write_csv(features_root / "segments_all.csv", SEGMENT_COLUMNS, (row for r in reports for row in r.get("segments") or []))
    _write_csv(features_root / "first5s_summary.csv", OPENING_COLUMNS, (_opening_row(r) for r in reports if "opening" in r))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [str(p) for p in inputs],
        "artifacts_root": str(root),
        "config": cfg.to_dict(),
        "counts": {"videos": len(records), **counts},
        "videos": records,
    }
    tmp = features_root / "run_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(features_root / "run_summary.json")
    return summary
