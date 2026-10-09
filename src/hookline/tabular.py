"""Read-only summaries of CSV / JSON / JSONL metadata files (stdlib only)."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METADATA_EXTENSIONS = frozenset({".csv", ".json", ".jsonl"})
ID_COLUMN_CANDIDATES = ("video_id", "Id", "id", "video")
# Known engagement-style targets. Presence here does not imply a definition is known.
LABEL_COLUMN_CANDIDATES = ("ECR", "NAWP", "label", "label_value", "engagement", "popularity")
MAX_CATEGORIES = 15


@dataclass
class TableSummary:
    path: str
    format: str
    error: str | None = None
    n_rows: int = 0
    columns: list[str] = field(default_factory=list)
    missing: dict[str, int] = field(default_factory=dict)
    id_column: str | None = None
    duplicate_ids: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Accept {"data": [...]} or a mapping of id -> record.
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            data = lists[0]
        else:
            data = [{"_key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in data.items()]
    if not isinstance(data, list):
        raise ValueError("JSON root is not a list of records")
    return [r if isinstance(r, dict) else {"value": r} for r in data]


def summarize_values(values: list[Any]) -> dict[str, Any]:
    present = [v for v in values if not _is_missing(v)]
    numeric: list[float] = []
    for v in present:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            numeric = []
            break
    counts = Counter(str(v) for v in present)
    if numeric and len(counts) > MAX_CATEGORIES:
        q = statistics.quantiles(numeric, n=20) if len(numeric) >= 2 else [numeric[0]] * 19
        return {
            "type": "numeric",
            "count": len(numeric),
            "min": min(numeric),
            "p05": q[0],
            "p25": q[4],
            "median": q[9],
            "p75": q[14],
            "p95": q[18],
            "max": max(numeric),
            "mean": statistics.fmean(numeric),
        }
    return {"type": "categorical", "count": len(present), "counts": dict(counts.most_common(MAX_CATEGORIES))}


def inspect_table(path: str | Path) -> TableSummary:
    path = Path(path)
    summary = TableSummary(path=str(path), format=path.suffix.lower().lstrip("."))
    try:
        rows = _load_rows(path)
    except Exception as exc:  # report, never crash the inspection run
        summary.error = f"{type(exc).__name__}: {exc}"
        return summary

    summary.n_rows = len(rows)
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    summary.columns = columns
    summary.missing = {c: sum(1 for r in rows if _is_missing(r.get(c))) for c in columns}

    summary.id_column = next((c for c in ID_COLUMN_CANDIDATES if c in columns), None)
    if summary.id_column:
        ids = Counter(str(r.get(summary.id_column)) for r in rows if not _is_missing(r.get(summary.id_column)))
        dups = [k for k, n in ids.items() if n > 1]
        summary.duplicate_ids = sum(ids[k] - 1 for k in dups)
        summary.duplicate_examples = dups[:5]

    wanted = {c.lower() for c in LABEL_COLUMN_CANDIDATES}
    for col in columns:
        if col.lower() in wanted:
            summary.labels[col] = summarize_values([r.get(col) for r in rows])
    return summary
