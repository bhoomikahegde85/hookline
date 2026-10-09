"""Video manifest schema, reading, writing and validation.

``file_path`` values are resolved relative to the manifest's own directory,
so a manifest and its videos can be moved together.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

MANIFEST_FIELDS: tuple[str, ...] = (
    "video_id",
    "file_path",
    "duration_seconds",
    "has_audio",
    "has_video",
    "source",
    "license_or_permission",
    "dataset_split",
    "notes",
)
# Only fill these when the label's definition and provenance are documented.
LABEL_FIELDS: tuple[str, ...] = ("label_name", "label_value", "label_source")

VALID_SPLITS = frozenset({"train", "val", "test", "demo", "fixture", "unassigned"})
TRUE_VALUES = frozenset({"true", "yes", "1"})
FALSE_VALUES = frozenset({"false", "no", "0"})
UNKNOWN_VALUES = frozenset({"", "unknown"})


class ManifestError(Exception):
    """The manifest cannot be read at all."""


@dataclass
class Issue:
    level: str  # "error" | "warning"
    message: str
    row: int | None = None  # 1-based data row (header excluded)
    column: str | None = None

    def __str__(self) -> str:
        where = f"row {self.row}" if self.row is not None else "manifest"
        if self.column:
            where += f", {self.column}"
        return f"[{self.level}] {where}: {self.message}"


@dataclass
class ValidationResult:
    path: Path
    columns: list[str]
    rows: list[dict[str, str]]
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_bool(value: str | None) -> bool | None:
    """Parse a manifest boolean. Returns None for unknown; raises ValueError if invalid."""
    v = (value or "").strip().lower()
    if v in TRUE_VALUES:
        return True
    if v in FALSE_VALUES:
        return False
    if v in UNKNOWN_VALUES:
        return None
    raise ValueError(f"expected true/false/unknown, got {value!r}")


def format_bool(value: bool | None) -> str:
    return "unknown" if value is None else str(value).lower()


def read_manifest(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = [c.strip() for c in (reader.fieldnames or [])]
            rows = [
                {(k or "").strip(): (v or "").strip() if isinstance(v, str) else "" for k, v in row.items()}
                for row in reader
            ]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ManifestError(f"could not parse {path} as UTF-8 CSV: {exc}") from exc
    if not columns:
        raise ManifestError(f"manifest has no header row: {path}")
    return columns, rows


def resolve_file_path(manifest_path: Path, file_path: str) -> Path:
    p = Path(file_path).expanduser()
    return p if p.is_absolute() else (manifest_path.parent / p).resolve()


def validate_manifest(path: str | Path, check_files: bool = True) -> ValidationResult:
    path = Path(path).resolve()
    columns, rows = read_manifest(path)
    result = ValidationResult(path=path, columns=columns, rows=rows)
    issues = result.issues

    missing = [c for c in MANIFEST_FIELDS if c not in columns]
    if missing:
        issues.append(Issue("error", f"missing required columns: {', '.join(missing)}"))
    unknown = [c for c in columns if c not in MANIFEST_FIELDS and c not in LABEL_FIELDS]
    if unknown:
        issues.append(Issue("warning", f"unrecognised columns (ignored): {', '.join(unknown)}"))
    if not rows:
        issues.append(Issue("warning", "manifest has no video rows yet"))

    id_counts = Counter(r.get("video_id", "") for r in rows if r.get("video_id"))
    path_counts = Counter(r.get("file_path", "") for r in rows if r.get("file_path"))

    for i, row in enumerate(rows, start=1):
        vid = row.get("video_id", "")
        if not vid:
            issues.append(Issue("error", "video_id is empty", i, "video_id"))
        elif id_counts[vid] > 1:
            issues.append(Issue("error", f"duplicate video_id {vid!r}", i, "video_id"))

        fp = row.get("file_path", "")
        if not fp:
            issues.append(Issue("error", "file_path is empty", i, "file_path"))
        else:
            if path_counts[fp] > 1:
                issues.append(Issue("warning", f"file_path {fp!r} listed more than once", i, "file_path"))
            if check_files and not resolve_file_path(path, fp).is_file():
                issues.append(Issue("error", f"file not found: {fp}", i, "file_path"))

        dur = row.get("duration_seconds", "")
        if dur:
            try:
                if float(dur) < 0:
                    issues.append(Issue("error", "duration_seconds is negative", i, "duration_seconds"))
            except ValueError:
                issues.append(Issue("error", f"duration_seconds is not a number: {dur!r}", i, "duration_seconds"))

        for col in ("has_audio", "has_video"):
            if col in row:
                try:
                    parse_bool(row[col])
                except ValueError as exc:
                    issues.append(Issue("error", str(exc), i, col))

        if "license_or_permission" in columns and not row.get("license_or_permission"):
            issues.append(
                Issue("error", "license_or_permission is empty; only use videos you may use", i, "license_or_permission")
            )

        split = row.get("dataset_split", "")
        if split and split not in VALID_SPLITS:
            issues.append(
                Issue("error", f"dataset_split {split!r} not in {sorted(VALID_SPLITS)}", i, "dataset_split")
            )

        if row.get("label_value") and not (row.get("label_name") and row.get("label_source")):
            issues.append(
                Issue("error", "label_value requires label_name and label_source", i, "label_value")
            )
    return result


def write_manifest(path: str | Path, rows: Iterable[Mapping[str, object]], include_labels: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MANIFEST_FIELDS) + (list(LABEL_FIELDS) if include_labels else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})
    return path
