"""Portable project paths. Override the data location with HOOKLINE_DATA_DIR."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    override = os.environ.get("HOOKLINE_DATA_DIR")
    return Path(override).expanduser().resolve() if override else PROJECT_ROOT / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def sample_dir() -> Path:
    return data_dir() / "sample"


def snapugc_dir() -> Path:
    return data_dir() / "snapugc"


def reports_dir() -> Path:
    return PROJECT_ROOT / "reports"


def artifacts_dir() -> Path:
    return PROJECT_ROOT / "artifacts"
