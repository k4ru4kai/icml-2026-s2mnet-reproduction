"""Synthetic-only tests for the offline DRIVE structural integrity checker.

Fixtures contain minimal generic TIFF/GIF headers and no retinal image pixels,
medical content, or dataset-derived bytes.
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import pytest

from repro.data.drive_integrity import (
    MANIFEST_FIELDS,
    DriveIntegrityError,
    main,
    validate_drive,
    write_integrity_report,
    write_manifest,
)


def _write_tiff_header(path: Path, width: int = 11, height: int = 7) -> None:
    """Write a minimal metadata-only TIFF header with no image pixel payload."""

    header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    entries = [
        struct.pack("<HHII", 256, 4, 1, width),
        struct.pack("<HHII", 257, 4, 1, height),
        struct.pack("<HHI", 277, 3, 1) + struct.pack("<H", 3) + b"\x00\x00",
    ]
    path.write_bytes(header + struct.pack("<H", len(entries)) + b"".join(entries))


def _write_gif_header(path: Path, width: int = 11, height: int = 7) -> None:
    """Write a minimal metadata-only GIF header with no image pixel payload."""

    path.write_bytes(b"GIF89a" + struct.pack("<HH", width, height))


def _build_synthetic_tree(root: Path) -> None:
    """Create only tiny generic headers under the expected directory schema."""

    for relative_directory in (
        "training/images",
        "training/1st_manual",
        "training/mask",
        "test/images",
        "test/1st_manual",
        "test/mask",
    ):
        (root / relative_directory).mkdir(parents=True, exist_ok=True)

    for image_id in range(1, 21):
        identifier = f"{image_id:02d}"
        _write_tiff_header(root / f"test/images/{identifier}_test.tif")
        _write_gif_header(root / f"test/1st_manual/{identifier}_manual1.gif")
        _write_gif_header(root / f"test/mask/{identifier}_test_mask.gif")

    for image_id in range(21, 41):
        identifier = f"{image_id:02d}"
        _write_tiff_header(root / f"training/images/{identifier}_training.tif")
        _write_gif_header(root / f"training/1st_manual/{identifier}_manual1.gif")
        _write_gif_header(root / f"training/mask/{identifier}_training_mask.gif")


def test_valid_synthetic_tree_generates_deterministic_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synthetic_drive_structure"
    output_root = tmp_path / "generated"
    _build_synthetic_tree(dataset_root)
    raw_snapshot = {
        path.relative_to(dataset_root).as_posix(): path.read_bytes()
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
    }

    result = validate_drive(dataset_root, provenance_source="synthetic-test-fixture")

    assert result.ok
    assert len(result.records) == 40
    assert [record.image_id for record in result.records] == [
        f"{value:02d}" for value in range(1, 41)
    ]
    assert sum(record.assigned_split == "test" for record in result.records) == 20
    assert sum(record.assigned_split == "train" for record in result.records) == 16
    assert sum(record.assigned_split == "validation" for record in result.records) == 4
    assert {record.annotation_observer for record in result.records} == {"first_manual"}
    assert {record.original_dimensions for record in result.records} == {"7x11x3"}
    assert {record.file_format for record in result.records} == {
        "image:tiff|vessel_mask:gif|fov_mask:gif"
    }
    assert all(len(record.sha256_image) == 64 for record in result.records)
    assert all(len(record.sha256_vessel_mask) == 64 for record in result.records)
    assert all(len(record.sha256_fov_mask) == 64 for record in result.records)

    first_manifest = write_manifest(result, output_root / "manifest-a.csv")
    second_manifest = write_manifest(result, output_root / "manifest-b.csv")
    first_report = write_integrity_report(result, output_root / "report-a.json")
    second_report = write_integrity_report(result, output_root / "report-b.json")

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
    with first_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == MANIFEST_FIELDS
    assert len(rows) == 40
    assert rows[0]["image_id"] == "01"
    assert rows[-1]["image_id"] == "40"
    report = json.loads(first_report.read_text(encoding="utf-8"))
    assert report["protocol_status"] == "PROVISIONAL"
    assert report["structural_validation_status"] == "pass"
    assert report["record_count"] == 40

    assert raw_snapshot == {
        path.relative_to(dataset_root).as_posix(): path.read_bytes()
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
    }


def test_absent_root_fails_safely_without_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    absent_root = tmp_path / "absent"
    manifest_path = tmp_path / "generated/manifest.csv"
    report_path = tmp_path / "generated/report.json"

    with pytest.raises(DriveIntegrityError, match="Dataset root does not exist"):
        validate_drive(absent_root)

    exit_code = main(
        [
            "--dataset-root",
            str(absent_root),
            "--manifest-out",
            str(manifest_path),
            "--report-out",
            str(report_path),
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 2
    assert "PROTOCOL_STATUS=PROVISIONAL" in output
    assert "DRIVE_INTEGRITY_STATUS=FAIL" in output
    assert "Dataset root does not exist" in output
    assert not manifest_path.exists()
    assert not report_path.exists()


def test_missing_duplicate_and_unexpected_ids_are_reported(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synthetic_drive_structure"
    _build_synthetic_tree(dataset_root)
    (dataset_root / "test/images/01_test.tif").unlink()
    _write_tiff_header(dataset_root / "test/images/02_test.tiff")
    _write_tiff_header(dataset_root / "test/images/99_test.tif")

    result = validate_drive(dataset_root)
    errors = "\n".join(result.errors)

    assert not result.ok
    assert "Missing image for test ID 01" in errors
    assert "Duplicate image files for test ID 02" in errors
    assert "Unexpected test ID 99 for image" in errors
    assert "Incomplete image/vessel/FOV correspondence for test ID 01" in errors
    assert "Expected 40 complete records" in errors


def test_dimension_mismatch_and_raw_output_write_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synthetic_drive_structure"
    _build_synthetic_tree(dataset_root)
    mismatched_fov = dataset_root / "training/mask/21_training_mask.gif"
    _write_gif_header(mismatched_fov, width=9, height=7)

    failed_result = validate_drive(dataset_root)
    assert not failed_result.ok
    assert any(
        "FOV mask dimensions do not match image for ID 21" in error
        for error in failed_result.errors
    )

    _write_gif_header(mismatched_fov)
    valid_result = validate_drive(dataset_root)
    assert valid_result.ok
    with pytest.raises(DriveIntegrityError, match="outside the immutable dataset root"):
        write_manifest(valid_result, dataset_root / "drive_manifest.csv")
    with pytest.raises(DriveIntegrityError, match="outside the immutable dataset root"):
        write_integrity_report(valid_result, dataset_root / "integrity_report.json")


def test_malformed_file_cannot_produce_a_valid_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synthetic_drive_structure"
    manifest_path = tmp_path / "generated/drive_manifest.csv"
    _build_synthetic_tree(dataset_root)
    (dataset_root / "test/images/01_test.tif").write_bytes(b"malformed-synthetic-header")

    result = validate_drive(dataset_root)

    assert not result.ok
    assert any("Invalid TIFF byte-order marker" in error for error in result.errors)
    with pytest.raises(
        DriveIntegrityError,
        match="Cannot write a manifest for a failed integrity result",
    ):
        write_manifest(result, manifest_path)
    assert not manifest_path.exists()
