"""Pre-batch reliability: run-level outputs survive partial runs, feature provenance and cache
invalidation, the feature-environment guard, and read-only feature reuse by the model stages."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from hookline import features, preprocess as prep_mod, provenance
from hookline.config import PipelineConfig
from hookline.features import SEGMENT_COLUMNS
from hookline.multimodal import Backends, run_multimodal
from hookline.pipeline import FeaturesUnavailable, VideoInput, load_features, process_video, run
from hookline.provenance import FeatureEnvironmentError, check_feature_environment

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg not installed")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG = PipelineConfig()
QUIET = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def two_clips(tmp_path_factory) -> dict[str, Path]:
    from test_pipeline import ffmpeg  # shared FFmpeg helper
    d = tmp_path_factory.mktemp("rel_clips")
    v, a = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"], ["-c:a", "aac", "-b:a", "64k"]
    clips = {"alpha": d / "alpha.mp4", "beta": d / "beta.mp4"}
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x284:r=30:d=2", "-f", "lavfi", "-i", "sine=duration=2", *v, *a, "-shortest",
           str(clips["alpha"]))
    ffmpeg("-f", "lavfi", "-i", "color=gray:s=160x284:r=30:d=3", "-an", *v, str(clips["beta"]))
    return clips


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_by_video(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in csv.DictReader(path.open()):
        out.setdefault(r["video_id"], []).append(r)
    return out


# ------------------------------------------------------------- 1. run-level outputs

def test_single_video_rerun_keeps_other_videos_in_feature_aggregates(two_clips, tmp_path) -> None:
    root = tmp_path / "art"
    run(list(two_clips.values()), CFG, root=root, **QUIET)
    feats = root / "features"
    before_rows = rows_by_video(feats / "segments_all.csv")
    before_open = {r["video_id"]: r for r in csv.DictReader((feats / "first5s_summary.csv").open())}
    assert set(before_rows) == {"alpha", "beta"}

    summary = run([two_clips["alpha"]], CFG, root=root, force=True, **QUIET)  # rerun one video only
    after_rows = rows_by_video(feats / "segments_all.csv")
    assert after_rows["beta"] == before_rows["beta"] and len(after_rows["alpha"]) == len(before_rows["alpha"])  # no duplicates
    after_open = {r["video_id"]: r for r in csv.DictReader((feats / "first5s_summary.csv").open())}
    assert after_open["beta"] == before_open["beta"] and set(after_open) == {"alpha", "beta"}
    assert [v["video_id"] for v in summary["videos"]] == ["alpha", "beta"] and summary["counts"]["videos"] == 2
    assert summary["last_run"]["video_ids"] == ["alpha"] and summary["last_run"]["counts"]["videos"] == 1
    disk = json.loads((feats / "run_summary.json").read_text())
    assert disk["videos"][1]["video_id"] == "beta" and disk["videos"][1]["cached_features"] is False  # beta's record kept


def test_stale_schema_videos_are_reported_not_padded(two_clips, tmp_path) -> None:
    root = tmp_path / "art"
    run(list(two_clips.values()), CFG, root=root, **QUIET)
    stale = root / "features" / "beta" / "segments.csv"
    stale.write_text("video_id,segment_index,old_column\nbeta,0,1\n")
    summary = run([two_clips["alpha"]], CFG, root=root, **QUIET)
    assert summary["excluded_stale_schema"] == ["beta"]
    with (root / "features" / "segments_all.csv").open() as f:
        header = next(csv.reader(f))
    assert header == SEGMENT_COLUMNS and set(rows_by_video(root / "features" / "segments_all.csv")) == {"alpha"}


class Nothing:
    def info(self):
        return {"backend": "none"}

    def transcribe(self, samples, sr):
        return {"language": "en", "language_probability": 0.99, "segments": []}

    def read(self, img):
        return []


def test_single_video_model_rerun_keeps_other_videos_in_merged_aggregates(two_clips, tmp_path) -> None:
    from hookline.context import LazyBackend
    root = tmp_path / "art"
    run(list(two_clips.values()), CFG, root=root, **QUIET)
    backends = Backends(LazyBackend(Nothing), LazyBackend(Nothing))
    run_multimodal(list(two_clips.values()), root, CFG, backends=backends, run_librosa=False, **QUIET)
    feats = root / "features"
    before = rows_by_video(feats / "segments_merged_all.csv")
    before_open = {r["video_id"]: r for r in csv.DictReader((feats / "first5s_multimodal.csv").open())}
    summary = run_multimodal([two_clips["alpha"]], root, CFG, backends=backends, run_librosa=False, force=True, **QUIET)
    after = rows_by_video(feats / "segments_merged_all.csv")
    assert after["beta"] == before["beta"] and len(after["alpha"]) == len(before["alpha"])
    assert {r["video_id"]: r for r in csv.DictReader((feats / "first5s_multimodal.csv").open())}["beta"] == before_open["beta"]
    assert [v["video_id"] for v in summary["videos"]] == ["alpha", "beta"] and summary["last_run"]["video_ids"] == ["alpha"]


# ------------------------------------------------------------- 2. provenance and cache invalidation

def test_report_records_dependency_versions(two_clips, tmp_path) -> None:
    rep = process_video(VideoInput("alpha", two_clips["alpha"]), CFG, tmp_path)
    ext = rep["extractor"]
    assert set(ext["dependencies"]) == {"opencv", "numpy", "ffmpeg"} and all(ext["dependencies"].values())
    assert ext["preprocess_ffmpeg"] == ext["dependencies"]["ffmpeg"]
    assert {"python", "python_executable"} <= set(ext["runtime"]) and ext["hookline_version"]
    cached = json.loads(next((tmp_path / "cache" / "alpha").glob("*/preprocess.json")).read_text())
    assert cached["ffmpeg_version"] == ext["dependencies"]["ffmpeg"]


def test_library_version_change_recomputes_features_but_reuses_proxy(two_clips, tmp_path, monkeypatch) -> None:
    item = VideoInput("alpha", two_clips["alpha"])
    first = process_video(item, CFG, tmp_path)
    assert first["_cache"] == {"preprocess": False, "features": False}
    assert process_video(item, CFG, tmp_path)["_cache"] == {"preprocess": True, "features": True}
    deps = provenance.feature_dependencies() | {"opencv": "0.0-other"}
    monkeypatch.setattr(features, "feature_dependencies", lambda: deps)
    again = process_video(item, CFG, tmp_path)
    assert again["_cache"] == {"preprocess": True, "features": False}  # recomputed from the cached proxy/WAV
    assert again["extractor"]["dependencies"]["opencv"] == "0.0-other"
    assert again["extractor"]["feature_key"] != first["extractor"]["feature_key"]


def test_ffmpeg_change_rebuilds_proxy_and_legacy_cache_is_accepted(two_clips, tmp_path, monkeypatch) -> None:
    item = VideoInput("alpha", two_clips["alpha"])
    process_video(item, CFG, tmp_path)
    status = next((tmp_path / "cache" / "alpha").glob("*/preprocess.json"))
    data = json.loads(status.read_text())
    data.pop("ffmpeg_version")  # a cache written before provenance existed
    status.write_text(json.dumps(data))
    legacy = prep_mod.preprocess(item.path, "alpha", tmp_path / "cache", CFG)
    assert legacy.cached and legacy.ffmpeg_version is None
    status.write_text(json.dumps(data | {"ffmpeg_version": "0.0-old"}))
    rebuilt = prep_mod.preprocess(item.path, "alpha", tmp_path / "cache", CFG)
    assert not rebuilt.cached and rebuilt.ffmpeg_version == provenance.ffmpeg_version()


# ------------------------------------------------------------- 3. environment guard

def test_feature_environment_is_recorded_and_mixing_is_refused(tmp_path, monkeypatch) -> None:
    feats = tmp_path / "features"
    rec = check_feature_environment(feats)
    assert rec["dependencies"] == provenance.feature_dependencies()
    assert json.loads((feats / "feature_env.json").read_text())["dependencies"] == rec["dependencies"]
    assert check_feature_environment(feats)["dependencies"] == rec["dependencies"]  # same env: fine
    other = rec["dependencies"] | {"opencv": "0.0-test", "numpy": "0.0-test"}
    monkeypatch.setattr(provenance, "feature_dependencies", lambda: other)
    with pytest.raises(FeatureEnvironmentError, match="OpenCV 0.0-test") as err:
        check_feature_environment(feats)
    assert rec["runtime"]["python_executable"] in str(err.value) and "--new-feature-env" in str(err.value)
    moved = check_feature_environment(feats, allow_new_env=True)
    assert moved["dependencies"] == other and moved["previous"] == rec["dependencies"]


def test_models_venv_is_refused_with_the_right_command(tmp_path) -> None:
    with pytest.raises(FeatureEnvironmentError, match=r"\.venv/bin/python scripts/run_pipeline.py"):
        check_feature_environment(tmp_path / "features", prefix=str(PROJECT_ROOT / ".venv-models"))
    assert not (tmp_path / "features" / "feature_env.json").exists()
    check_feature_environment(tmp_path / "features", prefix=str(PROJECT_ROOT / ".venv"))  # the feature env is fine


def test_cli_exits_3_on_environment_mismatch(two_clips, tmp_path, monkeypatch, capsys) -> None:
    spec = importlib.util.spec_from_file_location("run_pipeline_cli", PROJECT_ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = ["--input", str(two_clips["alpha"]), "--artifacts", str(tmp_path / "art")]
    assert mod.main(args) == 0
    monkeypatch.setattr(provenance, "feature_dependencies", lambda: {"opencv": "x", "numpy": "y", "ffmpeg": "z"})
    assert mod.main(args) == 3
    assert "wrong environment for Task 2 features" in capsys.readouterr().err
    seg_before = (tmp_path / "art" / "features" / "alpha" / "segments.csv").read_bytes()
    assert mod.main(args) == 3 and (tmp_path / "art" / "features" / "alpha" / "segments.csv").read_bytes() == seg_before


# ------------------------------------------------------------- model stages only read features

def test_model_stage_never_computes_features(two_clips, tmp_path) -> None:
    item = VideoInput("alpha", two_clips["alpha"])
    with pytest.raises(FeaturesUnavailable, match="run_pipeline.py --input"):
        load_features(item, CFG, tmp_path)
    assert not (tmp_path / "features" / "alpha").exists() and not (tmp_path / "cache").exists()
    process_video(item, CFG, tmp_path)
    assert load_features(item, CFG, tmp_path)["video_id"] == "alpha"
    with pytest.raises(FeaturesUnavailable, match="other settings"):
        load_features(item, replace(CFG, window_seconds=1.0), tmp_path)
    report_path = tmp_path / "features" / "alpha" / "report.json"
    rep = json.loads(report_path.read_text())
    rep["extractor"].pop("dependencies")
    report_path.write_text(json.dumps(rep))
    with pytest.raises(FeaturesUnavailable, match="predate dependency provenance"):
        load_features(item, CFG, tmp_path)
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(two_clips["beta"].read_bytes())
    with pytest.raises(FeaturesUnavailable, match="source changed"):
        load_features(VideoInput("alpha", copy), CFG, tmp_path)


def test_batch_marks_videos_without_features_failed_and_keeps_going(two_clips, tmp_path) -> None:
    from hookline.context import LazyBackend
    before = {k: sha(p) for k, p in two_clips.items()}
    process_video(VideoInput("alpha", two_clips["alpha"]), CFG, tmp_path)  # beta has no features
    backends = Backends(LazyBackend(Nothing), LazyBackend(Nothing))
    summary = run_multimodal(list(two_clips.values()), tmp_path, CFG, backends=backends, run_librosa=False, **QUIET)
    by_id = {v["video_id"]: v for v in summary["videos"]}
    assert by_id["alpha"]["status"] == "ok" and by_id["beta"]["status"] == "failed"
    assert "no Task 2 features for beta" in by_id["beta"]["error"] and "run_pipeline.py" in by_id["beta"]["error"]
    assert not (tmp_path / "features" / "beta").exists()
    assert {k: sha(p) for k, p in two_clips.items()} == before
