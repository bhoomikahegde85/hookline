"""Real TTCC inference on one explicitly chosen video (GPU, ~9.4 GB checkpoint). Skipped unless
``HOOKLINE_TTCC_VIDEO`` names the video; the TTCC environment is ``TTCC_PYTHON`` (default
``.venv-ttcc/bin/python``) and the checkpoint must already be downloaded (or set
``HOOKLINE_TTCC_DOWNLOAD=1``). See docs/ttcc.md.

    HOOKLINE_TTCC_VIDEO=/path/to/reel.mp4 TTCC_PYTHON=$(which python) python -m pytest -q tests/test_ttcc_integration.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hookline.config import TTCC_CHECKPOINT_REVISION, TtccConfig, env_flag
from hookline.pipeline import VideoInput
from hookline.ttcc import RAW_FILE, RETENTION_FILE, SUCCESS, run_ttcc

VIDEO = os.environ.get("HOOKLINE_TTCC_VIDEO")
pytestmark = pytest.mark.skipif(not VIDEO, reason="real TTCC inference: set HOOKLINE_TTCC_VIDEO (GPU and checkpoint needed)")


def test_real_checkpoint_predicts_a_valid_curve(tmp_path):
    video = Path(VIDEO).expanduser()
    cfg = TtccConfig.from_env(enabled=True, allow_download=env_flag("HOOKLINE_TTCC_DOWNLOAD") or None,
                              attn_impl=os.environ.get("HOOKLINE_TTCC_ATTN") or None)
    res = run_ttcc([VideoInput(video.stem, video)], tmp_path, cfg)
    rec = res["records"][0]
    assert rec["status"] == SUCCESS, rec.get("reason")
    d = tmp_path / "features" / video.stem
    art, raw = json.loads((d / RETENTION_FILE).read_text()), json.loads((d / RAW_FILE).read_text())
    T = art["horizon_s"]
    assert art["timestamps_s"] == list(range(T + 1)) and art["retention"][0] == 1.0
    assert all(b <= a + 1e-6 for a, b in zip(art["retention"], art["retention"][1:]))
    assert len(raw["r_pred"]) == 60 and raw["provenance"]["retention_head"]["verified"] is True
    assert raw["provenance"]["checkpoint"]["revision"] == TTCC_CHECKPOINT_REVISION
    assert raw["anchor"] == "</cot>"
