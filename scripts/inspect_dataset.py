#!/usr/bin/env python3
"""Inspect local videos, manifests and SnapUGC metadata. Read-only.

    python scripts/inspect_dataset.py                    # defaults: data/raw, data/sample, data/snapugc
    python scripts/inspect_dataset.py --decode-check     # also fully decode each video (slower)
    python scripts/inspect_dataset.py --videos path/to/dir_or_file --strict

Writes reports/dataset_report.md and reports/dataset_report.json.
Exit code is 0 unless --strict is given and errors were found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline import paths  # noqa: E402
from hookline.manifest import ManifestError, parse_bool, resolve_file_path, validate_manifest  # noqa: E402
from hookline.media import (  # noqa: E402
    STATUS_ERROR,
    STATUS_OK,
    VIDEO_EXTENSIONS,
    ProbeResult,
    find_tool,
    find_video_files,
    probe_video,
)
from hookline.tabular import METADATA_EXTENSIONS, inspect_table  # noqa: E402

DURATION_TOLERANCE_SECONDS = 0.5
IGNORED_NAMES = {"README.md", ".gitkeep", ".DS_Store"}


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def find_files(roots: list[Path], suffixes: frozenset[str], exclude_names: set[str] = frozenset()) -> list[Path]:
    out: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() in suffixes:
            out.add(root.resolve())
        elif root.is_dir():
            out.update(
                p.resolve() for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes and p.name not in exclude_names
            )
    return sorted(out)


def other_files(roots: list[Path]) -> list[Path]:
    known = VIDEO_EXTENSIONS | METADATA_EXTENSIONS
    out = []
    for root in roots:
        if root.is_dir():
            out += [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() not in known and p.name not in IGNORED_NAMES]
    return sorted(out)


def cross_check(manifest_path: Path, rows: list[dict[str, str]], probes: dict[str, ProbeResult]) -> list[str]:
    """Compare manifest claims with what the files actually contain."""
    notes: list[str] = []
    for row in rows:
        fp = row.get("file_path")
        if not fp:
            continue
        probe = probes.get(str(resolve_file_path(manifest_path, fp)))
        if not probe or probe.status != STATUS_OK:
            continue
        vid = row.get("video_id", fp)
        for col, actual in (("has_audio", probe.has_audio), ("has_video", probe.has_video)):
            try:
                claimed = parse_bool(row.get(col))
            except ValueError:
                continue
            if claimed is not None and actual is not None and claimed != actual:
                notes.append(f"{vid}: manifest {col}={claimed} but file has {col}={actual}")
        try:
            claimed_d = float(row["duration_seconds"]) if row.get("duration_seconds") else None
        except ValueError:
            claimed_d = None
        if claimed_d is not None and probe.duration_seconds is not None:
            if abs(claimed_d - probe.duration_seconds) > DURATION_TOLERANCE_SECONDS:
                notes.append(f"{vid}: manifest duration {claimed_d}s but file is {probe.duration_seconds}s")
    return notes


def build_report(video_roots: list[Path], manifest_paths: list[Path], metadata_roots: list[Path], deep: bool, max_videos: int | None) -> dict[str, Any]:
    tools = {"ffprobe": find_tool("ffprobe"), "ffmpeg": find_tool("ffmpeg")}
    try:
        import cv2  # type: ignore  # noqa: F401
        tools["opencv"] = "available"
    except ImportError:
        tools["opencv"] = None

    existing_roots = [r for r in video_roots if r.exists()]
    missing_roots = [rel(r) for r in video_roots if not r.exists()]
    videos = find_video_files(existing_roots)
    skipped = 0
    if max_videos is not None and len(videos) > max_videos:
        skipped = len(videos) - max_videos
        videos = videos[:max_videos]

    probes: dict[str, ProbeResult] = {}
    for i, v in enumerate(videos, 1):
        print(f"  probing [{i}/{len(videos)}] {rel(v)}", file=sys.stderr)
        probes[str(v)] = probe_video(v, ffprobe=tools["ffprobe"], deep=deep and bool(tools["ffmpeg"]))

    manifests: list[dict[str, Any]] = []
    listed: set[str] = set()
    for mp in manifest_paths:
        entry: dict[str, Any] = {"path": rel(mp)}
        try:
            res = validate_manifest(mp)
        except ManifestError as exc:
            entry["error"] = str(exc)
            manifests.append(entry)
            continue
        for row in res.rows:
            if row.get("file_path"):
                listed.add(str(resolve_file_path(res.path, row["file_path"])))
        splits = Counter(r.get("dataset_split") or "(blank)" for r in res.rows)
        labelled = Counter(r.get("label_name") for r in res.rows if r.get("label_value"))
        entry.update(
            rows=len(res.rows),
            columns=res.columns,
            splits=dict(splits),
            labelled=dict(labelled),
            errors=[str(i) for i in res.errors],
            warnings=[str(i) for i in res.warnings],
            cross_check=cross_check(res.path, res.rows, probes),
        )
        manifests.append(entry)

    manifest_names = {Path(m).name for m in manifest_paths}
    tables = [
        asdict(inspect_table(p))
        for p in find_files([r for r in metadata_roots if r.exists()], METADATA_EXTENSIONS, exclude_names=manifest_names)
    ]

    probe_list = [p.to_dict() for p in probes.values()]
    ok = [p for p in probes.values() if p.status == STATUS_OK]
    durations = [p.duration_seconds for p in ok if p.duration_seconds is not None]
    summary = {
        "videos_found": len(videos) + skipped,
        "videos_probed": len(videos),
        "videos_skipped_by_limit": skipped,
        "readable": len(ok),
        "errors": sum(1 for p in probes.values() if p.status == STATUS_ERROR),
        "unchecked": sum(1 for p in probes.values() if p.status not in (STATUS_OK, STATUS_ERROR)),
        "formats": dict(Counter(Path(p.path).suffix.lower() for p in probes.values())),
        "with_audio": sum(1 for p in ok if p.has_audio),
        "without_audio": sum(1 for p in ok if p.has_audio is False),
        "audio_unknown": sum(1 for p in ok if p.has_audio is None),
        "without_video": sum(1 for p in ok if p.has_video is False),
        "orientation": dict(Counter(p.orientation or "unknown" for p in ok)),
        "duration_min": min(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
        "duration_total": round(sum(durations), 1) if durations else None,
        "not_in_any_manifest": [rel(v) for v in probes if v not in listed],
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tools": tools,
        "video_roots": [rel(r) for r in video_roots],
        "missing_roots": missing_roots,
        "summary": summary,
        "videos": probe_list,
        "manifests": manifests,
        "metadata_tables": tables,
        "other_files": [rel(p) for p in other_files(existing_roots)][:50],
    }


def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(r: dict[str, Any]) -> str:
    s = r["summary"]
    L = ["# Hookline dataset report", "", f"Generated: {r['generated_at']}", "", "## Tools", ""]
    for k, v in r["tools"].items():
        L.append(f"- {k}: {v or 'NOT AVAILABLE'}")
    if not r["tools"]["ffprobe"]:
        L.append("- ffprobe missing: audio presence is reported as unknown. Install FFmpeg for full results.")
    L += ["", "## Summary", "", f"- Video roots: {', '.join(r['video_roots'])}"]
    if r["missing_roots"]:
        L.append(f"- Missing roots (skipped): {', '.join(r['missing_roots'])}")
    L += [
        f"- Videos found: {s['videos_found']} (probed {s['videos_probed']}, skipped by --max-videos {s['videos_skipped_by_limit']})",
        f"- Readable: {s['readable']}, errors: {s['errors']}, unchecked: {s['unchecked']}",
        f"- Formats: {s['formats'] or '—'}",
        f"- Audio stream: {s['with_audio']} with, {s['without_audio']} without, {s['audio_unknown']} unknown; video stream missing: {s['without_video']}",
        f"- Orientation: {s['orientation'] or '—'}",
        f"- Duration range: {fmt(s['duration_min'])}s to {fmt(s['duration_max'])}s (total {fmt(s['duration_total'], 1)}s)",
    ]
    if s["not_in_any_manifest"]:
        L.append(f"- Videos not listed in any manifest: {len(s['not_in_any_manifest'])}")

    L += ["", "## Videos", ""]
    if r["videos"]:
        L += ["| File | Status | Duration (s) | Resolution | FPS | Audio | Video | Notes |", "|---|---|---|---|---|---|---|---|"]
        for v in r["videos"]:
            res = f"{v['width']}x{v['height']}" if v["width"] else "—"
            notes = v["error"] or "; ".join(v["warnings"]) or ""
            L.append(
                f"| {rel(v['path'])} | {v['status']} | {fmt(v['duration_seconds'])} | {res} | {fmt(v['fps'])} "
                f"| {fmt(v['has_audio'])} | {fmt(v['has_video'])} | {notes} |"
            )
    else:
        L.append("No video files found. Add videos to `data/raw/` and list them in `data/raw/manifest.csv` (see data/raw/README.md).")

    L += ["", "## Manifests", ""]
    if not r["manifests"]:
        L.append("No manifests found.")
    for m in r["manifests"]:
        L.append(f"### {m['path']}")
        L.append("")
        if "error" in m:
            L += [f"- ERROR: {m['error']}", ""]
            continue
        L.append(f"- Rows: {m['rows']}; splits: {m['splits'] or '—'}; labelled rows: {m['labelled'] or 'none'}")
        L.append(f"- Validation: {len(m['errors'])} errors, {len(m['warnings'])} warnings")
        L += [f"  - {x}" for x in (m["errors"] + m["warnings"])[:30]]
        if m["cross_check"]:
            L.append("- Manifest vs file mismatches:")
            L += [f"  - {x}" for x in m["cross_check"][:30]]
        L.append("")

    L += ["## Metadata tables", ""]
    if not r["metadata_tables"]:
        L.append("No CSV/JSON metadata found (SnapUGC metadata goes in `data/snapugc/metadata/`).")
    for t in r["metadata_tables"]:
        L.append(f"### {rel(t['path'])}")
        L.append("")
        if t["error"]:
            L += [f"- ERROR: {t['error']}", ""]
            continue
        L.append(f"- Rows: {t['n_rows']}; columns: {', '.join(t['columns'])}")
        missing = {k: v for k, v in t["missing"].items() if v}
        L.append(f"- Missing/empty values: {missing or 'none'}")
        if t["id_column"]:
            L.append(f"- ID column `{t['id_column']}`: {t['duplicate_ids']} duplicates {t['duplicate_examples'] or ''}")
        for col, lab in t["labels"].items():
            if lab["type"] == "numeric":
                L.append(
                    f"- Target `{col}` (numeric, n={lab['count']}): min {lab['min']:.4f}, p25 {lab['p25']:.4f}, "
                    f"median {lab['median']:.4f}, p75 {lab['p75']:.4f}, max {lab['max']:.4f}, mean {lab['mean']:.4f}"
                )
            else:
                L.append(f"- Target `{col}` (categorical, n={lab['count']}): {lab['counts']}")
        L.append("")

    if r["other_files"]:
        L += ["## Other files (not inspected)", ""] + [f"- {x}" for x in r["other_files"]]
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="*", type=Path, help="video files/dirs (default: data/raw data/sample data/snapugc)")
    parser.add_argument("--manifests", nargs="*", type=Path, help="manifest CSVs (default: every manifest.csv under data/)")
    parser.add_argument("--metadata", nargs="*", type=Path, help="metadata dirs/files (default: data/snapugc)")
    parser.add_argument("--decode-check", action="store_true", help="fully decode each video to catch partial corruption")
    parser.add_argument("--max-videos", type=int, default=None, help="probe at most N videos")
    parser.add_argument("--report", type=Path, default=paths.reports_dir() / "dataset_report.md")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any errors were found")
    args = parser.parse_args()

    video_roots = args.videos if args.videos else [paths.raw_dir(), paths.sample_dir(), paths.snapugc_dir()]
    manifests = args.manifests if args.manifests is not None else sorted(paths.data_dir().rglob("manifest.csv"))
    metadata_roots = args.metadata if args.metadata is not None else [paths.snapugc_dir()]

    report = build_report(video_roots, manifests, metadata_roots, args.decode_check, args.max_videos)
    md = to_markdown(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(md, encoding="utf-8")
    args.report.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(md)
    print(f"Saved: {rel(args.report)} and {rel(args.report.with_suffix('.json'))}")

    n_errors = report["summary"]["errors"] + sum(len(m.get("errors", [])) + ("error" in m) for m in report["manifests"])
    n_errors += sum(1 for t in report["metadata_tables"] if t["error"])
    return 1 if args.strict and n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
