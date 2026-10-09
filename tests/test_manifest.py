from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hookline.manifest import (
    MANIFEST_FIELDS,
    ManifestError,
    parse_bool,
    validate_manifest,
    write_manifest,
)


def make_row(**overrides: str) -> dict[str, str]:
    row = {
        "video_id": "v1",
        "file_path": "v1.mp4",
        "duration_seconds": "12.5",
        "has_audio": "true",
        "has_video": "true",
        "source": "my phone",
        "license_or_permission": "own content",
        "dataset_split": "demo",
        "notes": "",
    }
    row.update(overrides)
    return row


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> Path:
    fields = fields or list(MANIFEST_FIELDS)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def messages(result) -> str:
    return "\n".join(str(i) for i in result.issues)


def test_valid_manifest_passes(tmp_path: Path) -> None:
    (tmp_path / "v1.mp4").write_bytes(b"x")
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", [make_row()]))
    assert result.ok, messages(result)
    assert not result.warnings


def test_header_only_manifest_is_valid_with_warning(tmp_path: Path) -> None:
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", []))
    assert result.ok
    assert any("no video rows" in w.message for w in result.warnings)


def test_missing_manifest_raises_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="manifest not found"):
        validate_manifest(tmp_path / "nope.csv")


def test_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", [make_row(file_path="gone.mp4")]))
    assert not result.ok
    assert any(i.column == "file_path" and "file not found" in i.message for i in result.errors)


def test_check_files_false_skips_existence(tmp_path: Path) -> None:
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", [make_row(file_path="gone.mp4")]), check_files=False)
    assert result.ok, messages(result)


def test_file_paths_resolve_relative_to_manifest(tmp_path: Path) -> None:
    sub = tmp_path / "data" / "raw"
    sub.mkdir(parents=True)
    (sub / "clips").mkdir()
    (sub / "clips" / "v1.mp4").write_bytes(b"x")
    result = validate_manifest(write_csv(sub / "manifest.csv", [make_row(file_path="clips/v1.mp4")]))
    assert result.ok, messages(result)


def test_missing_required_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "manifest.csv", [{"video_id": "a", "file_path": "a.mp4"}], ["video_id", "file_path"])
    result = validate_manifest(path, check_files=False)
    assert any("missing required columns" in i.message for i in result.errors)


@pytest.mark.parametrize(
    "overrides, column",
    [
        ({"video_id": ""}, "video_id"),
        ({"duration_seconds": "abc"}, "duration_seconds"),
        ({"duration_seconds": "-1"}, "duration_seconds"),
        ({"has_audio": "maybe"}, "has_audio"),
        ({"dataset_split": "holdout"}, "dataset_split"),
        ({"license_or_permission": ""}, "license_or_permission"),
    ],
)
def test_invalid_values(tmp_path: Path, overrides: dict[str, str], column: str) -> None:
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", [make_row(**overrides)]), check_files=False)
    assert any(i.column == column for i in result.errors), messages(result)


def test_duplicate_ids(tmp_path: Path) -> None:
    rows = [make_row(), make_row(file_path="other.mp4")]
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", rows), check_files=False)
    assert sum(1 for i in result.errors if "duplicate video_id" in i.message) == 2


def test_unknown_audio_is_allowed(tmp_path: Path) -> None:
    rows = [make_row(has_audio="unknown"), make_row(video_id="v2", has_audio="")]
    result = validate_manifest(write_csv(tmp_path / "manifest.csv", rows), check_files=False)
    assert result.ok, messages(result)


def test_label_requires_definition_and_source(tmp_path: Path) -> None:
    fields = list(MANIFEST_FIELDS) + ["label_name", "label_value", "label_source"]
    path = write_csv(tmp_path / "manifest.csv", [make_row(label_value="0.4")], fields)
    result = validate_manifest(path, check_files=False)
    assert any(i.column == "label_value" for i in result.errors)


def test_parse_bool() -> None:
    assert parse_bool("TRUE") is True
    assert parse_bool("no") is False
    assert parse_bool("") is None
    assert parse_bool("unknown") is None
    with pytest.raises(ValueError):
        parse_bool("perhaps")


def test_write_then_validate_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "v1.mp4").write_bytes(b"x")
    path = write_manifest(tmp_path / "manifest.csv", [make_row(has_audio=None)])  # type: ignore[arg-type]
    result = validate_manifest(path)
    assert result.ok, messages(result)
    assert result.rows[0]["has_audio"] == ""
