from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hookline.tabular import inspect_table

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_summary_reports_missing_duplicates_and_numeric_label(tmp_path: Path) -> None:
    rows = ["Id,Title,ECR"] + [f"id{i},t{i},{i / 100}" for i in range(30)] + ["id0,,0.5"]
    path = tmp_path / "train.csv"
    path.write_text("\n".join(rows))
    s = inspect_table(path)
    assert s.error is None
    assert s.n_rows == 31
    assert s.missing["Title"] == 1
    assert s.id_column == "Id" and s.duplicate_ids == 1
    assert s.labels["ECR"]["type"] == "numeric"
    assert s.labels["ECR"]["max"] == 0.5


def test_json_summary_categorical_label(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"video": "a.mp4", "label": "0"}, {"video": "b.mp4", "label": "2"}, {"video": "c.mp4", "label": "2"}]))
    s = inspect_table(path)
    assert s.labels["label"] == {"type": "categorical", "count": 3, "counts": {"2": 2, "0": 1}}


def test_malformed_json_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    s = inspect_table(path)
    assert s.error and "JSONDecodeError" in s.error


def test_dataset_report_handles_missing_everything(tmp_path: Path) -> None:
    mod = load_script("inspect_dataset")
    report = mod.build_report(
        video_roots=[tmp_path / "no-videos-here"],
        manifest_paths=[tmp_path / "missing_manifest.csv"],
        metadata_roots=[tmp_path / "no-metadata"],
        deep=False,
        max_videos=None,
    )
    assert report["summary"]["videos_found"] == 0
    assert report["missing_roots"]
    assert "manifest not found" in report["manifests"][0]["error"]
    md = mod.to_markdown(report)
    assert "No video files found" in md


def test_dataset_report_flags_manifest_and_file_problems(tmp_path: Path) -> None:
    (tmp_path / "corrupt.mp4").write_bytes(b"\x00" * 64)
    (tmp_path / "manifest.csv").write_text(
        "video_id,file_path,duration_seconds,has_audio,has_video,source,license_or_permission,dataset_split,notes\n"
        "c,corrupt.mp4,,unknown,unknown,test,own,fixture,\n"
        "m,missing.mp4,,unknown,unknown,test,own,fixture,\n"
    )
    mod = load_script("inspect_dataset")
    report = mod.build_report([tmp_path], [tmp_path / "manifest.csv"], [], deep=False, max_videos=None)
    assert report["summary"]["videos_probed"] == 1
    assert report["summary"]["readable"] == 0
    errors = report["manifests"][0]["errors"]
    assert any("missing.mp4" in e for e in errors)
    assert (tmp_path / "corrupt.mp4").read_bytes() == b"\x00" * 64
