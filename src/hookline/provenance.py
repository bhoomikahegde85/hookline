"""Which software produced the Task 2 features, and which environment may extend a root.

Task 2 features (``run_pipeline.py``) depend on OpenCV (frame measurements), NumPy (all
numerics) and FFmpeg (proxy decoding, WAV extraction). Those versions are recorded in every
``report.json`` and are part of the feature cache key, so features computed with different
library versions are never silently reused.

Each artifacts root also records the environment that built its features in
``features/feature_env.json``. ``run_pipeline.py`` refuses to add features from an environment
with different OpenCV/NumPy/FFmpeg versions to that root (which would mix incomparable
features across videos), and refuses to run in this project's ``.venv-models`` (whose
PaddleOCR pins an older OpenCV/NumPy). Model stages (``run_speech_ocr.py``) never compute
Task 2 features; they only read them.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .media import find_tool
from .paths import PROJECT_ROOT

FEATURE_ENV_FILE = "feature_env.json"
MODELS_VENV = ".venv-models"
FEATURE_VENV = ".venv"


class FeatureEnvironmentError(RuntimeError):
    """Task 2 features would be computed in the wrong environment."""


@lru_cache(maxsize=None)
def ffmpeg_version(ffmpeg: str | None = None) -> str | None:
    ffmpeg = ffmpeg or find_tool("ffmpeg")
    if not ffmpeg:
        return None
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"ffmpeg version (\S+)", out)
    return m.group(1) if m else None


def opencv_version() -> str | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2.__version__


def feature_dependencies() -> dict[str, str | None]:
    """Versions that change Task 2 feature values; part of the feature cache key."""
    return {"opencv": opencv_version(), "numpy": np.__version__, "ffmpeg": ffmpeg_version()}


def runtime_info() -> dict[str, Any]:
    """Recorded alongside the dependencies; not part of any cache key."""
    return {"python": platform.python_version(), "python_executable": sys.executable, "sys_prefix": sys.prefix,
            "platform": platform.platform(terse=True)}


def current_prefix() -> str:
    """The running interpreter's environment (``sys.prefix``): a venv's own directory."""
    return sys.prefix


def _is_project_venv(name: str, prefix: str | None = None) -> bool:
    try:
        return Path(prefix or current_prefix()).resolve() == (PROJECT_ROOT / name).resolve()
    except OSError:
        return False


def feature_command(root: Path | str | None = None, inputs: str = "<video or dir>") -> str:
    python = PROJECT_ROOT / FEATURE_VENV / "bin" / "python"
    shown = python.relative_to(PROJECT_ROOT) if python.exists() else "python  # the environment from requirements.txt"
    return f"{shown} scripts/run_pipeline.py --input {inputs}" + (f" --artifacts {root}" if root else "")


def check_feature_environment(features_root: Path, allow_new_env: bool = False, prefix: str | None = None) -> dict:
    """Raise ``FeatureEnvironmentError`` unless this interpreter may compute features for the root.

    Returns the environment record (written on first use, or replaced when ``allow_new_env``)."""
    if _is_project_venv(MODELS_VENV, prefix):
        raise FeatureEnvironmentError(
            f"Task 2 features must not be computed with {MODELS_VENV} (python: {sys.executable}). Its OpenCV/NumPy "
            f"differ from the feature environment. Run: {feature_command(features_root.parent)}")
    current = {"dependencies": feature_dependencies(), "runtime": runtime_info()}
    path = features_root / FEATURE_ENV_FILE
    try:
        recorded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        recorded = None
    if recorded and recorded.get("dependencies") != current["dependencies"] and not allow_new_env:
        then = recorded.get("dependencies") or {}
        now = current["dependencies"]
        exe = (recorded.get("runtime") or {}).get("python_executable")
        raise FeatureEnvironmentError(
            f"features in {features_root} were built with OpenCV {then.get('opencv')}, NumPy {then.get('numpy')}, "
            f"FFmpeg {then.get('ffmpeg')} ({exe}); this interpreter ({sys.executable}) has OpenCV {now['opencv']}, "
            f"NumPy {now['numpy']}, FFmpeg {now['ffmpeg']}. Mixing them would make features incomparable across videos. "
            f"Use {exe} scripts/run_pipeline.py, or pass --new-feature-env to rebuild this root's features here.")
    if not recorded or recorded.get("dependencies") != current["dependencies"]:
        features_root.mkdir(parents=True, exist_ok=True)
        record = current | {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "previous": recorded.get("dependencies") if recorded else None}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(path)
        return record
    return recorded
