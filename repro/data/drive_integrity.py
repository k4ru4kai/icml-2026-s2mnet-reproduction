#!/usr/bin/env python3
"""Validate a local DRIVE tree and generate a deterministic manifest offline.

PROTOCOL_STATUS=PROVISIONAL. This module performs structural validation only.
It has no network, acquisition, preprocessing, training, or evaluation behavior.
Successful validation does not establish authenticity, provenance, licensing,
annotation semantics, or compliance with terms of use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROTOCOL_STATUS = "PROVISIONAL"
AUTHORITATIVE_PROFILE = "authoritative_hidden_test"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/drive/drive_manifest.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data/reports/drive/drive_integrity_report.json"
DEFAULT_PROVENANCE = "UNVERIFIED_LOCAL_SOURCE"
TRAINING_ANNOTATION_AVAILABILITY = "available_authoritative_distribution"
TEST_ANNOTATION_AVAILABILITY = "not_distributed"
UNAVAILABLE_ANNOTATION_OBSERVER = "not_distributed"

PROTOCOL_WARNINGS = (
    "The DRIVE reproduction protocol remains provisional.",
    "Successful structural validation does not establish dataset authenticity.",
    "Authoritative provenance and terms of use must be checked separately.",
    "Raw filename patterns and encodings require authenticated confirmation.",
    "Official test vessel annotations are hidden and must not be present locally.",
    "No training or evaluation is performed.",
)

MANIFEST_FIELDS = (
    "image_id",
    "assigned_split",
    "relative_image_path",
    "relative_vessel_mask_path",
    "relative_fov_mask_path",
    "annotation_observer",
    "annotation_availability",
    "original_dimensions",
    "file_format",
    "sha256_image",
    "sha256_vessel_mask",
    "sha256_fov_mask",
    "provenance_source",
    "integrity_check_status",
)

TEST_IDS = tuple(f"{value:02d}" for value in range(1, 21))
OFFICIAL_TRAINING_IDS = tuple(f"{value:02d}" for value in range(21, 41))
TRAIN_IDS = frozenset(f"{value:02d}" for value in range(21, 37))
VALIDATION_IDS = frozenset(f"{value:02d}" for value in range(37, 41))


class DriveIntegrityError(RuntimeError):
    """Raised when validation cannot start or generated output would be unsafe."""


@dataclass(frozen=True)
class ImageMetadata:
    """Header-level metadata read without decoding or modifying an image."""

    width: int
    height: int
    channels: int
    file_format: str


@dataclass(frozen=True)
class ManifestRecord:
    """One deterministic manifest row."""

    image_id: str
    assigned_split: str
    relative_image_path: str
    relative_vessel_mask_path: str
    relative_fov_mask_path: str
    annotation_observer: str
    annotation_availability: str
    original_dimensions: str
    file_format: str
    sha256_image: str
    sha256_vessel_mask: str
    sha256_fov_mask: str
    provenance_source: str
    integrity_check_status: str


@dataclass(frozen=True)
class IntegrityResult:
    """Complete structural-validation result."""

    dataset_root: Path
    records: tuple[ManifestRecord, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = PROTOCOL_WARNINGS

    @property
    def ok(self) -> bool:
        """Return whether all 40 records satisfy their split-specific rules."""

        if self.errors or len(self.records) != 40:
            return False
        expected_ids = TEST_IDS + OFFICIAL_TRAINING_IDS
        if tuple(record.image_id for record in self.records) != expected_ids:
            return False
        return all(_record_satisfies_split_requirements(record) for record in self.records)

    def report_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible report."""

        return {
            "dataset_root": str(self.dataset_root),
            "distribution_profile": AUTHORITATIVE_PROFILE,
            "errors": list(self.errors),
            "manifest_fields": list(MANIFEST_FIELDS),
            "protocol_status": PROTOCOL_STATUS,
            "record_count": len(self.records),
            "records": [asdict(record) for record in self.records],
            "structural_validation_status": "pass" if self.ok else "fail",
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class FileRule:
    """Expected directory and filename rule for one DRIVE file role."""

    split: str
    role: str
    relative_directory: Path
    filename_pattern: re.Pattern[str]
    expected_ids: tuple[str, ...]


FILE_RULES = (
    FileRule(
        split="training",
        role="image",
        relative_directory=Path("training/images"),
        filename_pattern=re.compile(r"(?P<id>\d{2})_training\.(?:tif|tiff)"),
        expected_ids=OFFICIAL_TRAINING_IDS,
    ),
    FileRule(
        split="training",
        role="vessel annotation",
        relative_directory=Path("training/1st_manual"),
        filename_pattern=re.compile(r"(?P<id>\d{2})_manual1\.gif"),
        expected_ids=OFFICIAL_TRAINING_IDS,
    ),
    FileRule(
        split="training",
        role="FOV mask",
        relative_directory=Path("training/mask"),
        filename_pattern=re.compile(r"(?P<id>\d{2})_training_mask\.gif"),
        expected_ids=OFFICIAL_TRAINING_IDS,
    ),
    FileRule(
        split="test",
        role="image",
        relative_directory=Path("test/images"),
        filename_pattern=re.compile(r"(?P<id>\d{2})_test\.(?:tif|tiff)"),
        expected_ids=TEST_IDS,
    ),
    FileRule(
        split="test",
        role="FOV mask",
        relative_directory=Path("test/mask"),
        filename_pattern=re.compile(r"(?P<id>\d{2})_test_mask\.gif"),
        expected_ids=TEST_IDS,
    ),
)

PROHIBITED_TEST_ANNOTATION_DIRECTORY = Path("test/1st_manual")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_slice(data: bytes, start: int, size: int, path: Path) -> bytes:
    end = start + size
    if start < 0 or end > len(data):
        raise DriveIntegrityError(f"Truncated image header: {path}")
    return data[start:end]


def _read_gif_metadata(path: Path, data: bytes) -> ImageMetadata:
    if len(data) < 10 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        raise DriveIntegrityError(f"Invalid GIF header: {path}")
    width, height = struct.unpack("<HH", data[6:10])
    if width <= 0 or height <= 0:
        raise DriveIntegrityError(f"Invalid GIF dimensions: {path}")
    return ImageMetadata(width=width, height=height, channels=1, file_format="gif")


def _tiff_scalar(
    data: bytes,
    entry_offset: int,
    byte_order: str,
    path: Path,
) -> tuple[int, int]:
    entry = _require_slice(data, entry_offset, 12, path)
    tag, value_type, count = struct.unpack(f"{byte_order}HHI", entry[:8])
    type_details = {3: ("H", 2), 4: ("I", 4)}
    if value_type not in type_details or count < 1:
        return tag, -1

    format_code, value_size = type_details[value_type]
    total_size = count * value_size
    if total_size <= 4:
        value_offset = entry_offset + 8
    else:
        value_offset = struct.unpack(f"{byte_order}I", entry[8:12])[0]
    raw_value = _require_slice(data, value_offset, value_size, path)
    return tag, struct.unpack(f"{byte_order}{format_code}", raw_value)[0]


def _read_tiff_metadata(path: Path, data: bytes) -> ImageMetadata:
    if len(data) < 8 or data[:2] not in {b"II", b"MM"}:
        raise DriveIntegrityError(f"Invalid TIFF byte-order marker: {path}")
    byte_order = "<" if data[:2] == b"II" else ">"
    if struct.unpack(f"{byte_order}H", data[2:4])[0] != 42:
        raise DriveIntegrityError(f"Invalid TIFF magic number: {path}")

    ifd_offset = struct.unpack(f"{byte_order}I", data[4:8])[0]
    entry_count = struct.unpack(
        f"{byte_order}H", _require_slice(data, ifd_offset, 2, path)
    )[0]
    values: dict[int, int] = {}
    for index in range(entry_count):
        tag, value = _tiff_scalar(data, ifd_offset + 2 + 12 * index, byte_order, path)
        if value >= 0:
            values[tag] = value

    width = values.get(256, 0)
    height = values.get(257, 0)
    photometric = values.get(262, 0)
    channels = values.get(277, 3 if photometric == 2 else 1)
    if width <= 0 or height <= 0 or channels <= 0:
        raise DriveIntegrityError(f"Missing or invalid TIFF dimensions: {path}")
    return ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        file_format="tiff",
    )


def _read_image_metadata(path: Path) -> ImageMetadata:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DriveIntegrityError(f"Cannot read file {path}: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".gif":
        return _read_gif_metadata(path, data)
    if suffix in {".tif", ".tiff"}:
        return _read_tiff_metadata(path, data)
    raise DriveIntegrityError(f"Unsupported DRIVE file format: {path}")


def _scan_rule(
    dataset_root: Path,
    rule: FileRule,
    errors: list[str],
) -> dict[str, Path]:
    directory = dataset_root / rule.relative_directory
    if not directory.is_dir():
        errors.append(f"Missing required directory: {rule.relative_directory.as_posix()}")
        return {}

    matches: dict[str, list[Path]] = {}
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        errors.append(f"Cannot list {rule.relative_directory.as_posix()}: {exc}")
        return {}

    for entry in entries:
        relative_entry = entry.relative_to(dataset_root).as_posix()
        if entry.is_symlink():
            errors.append(f"Symlinks are not accepted in the raw tree: {relative_entry}")
            continue
        if not entry.is_file():
            errors.append(f"Unexpected non-file entry: {relative_entry}")
            continue
        match = rule.filename_pattern.fullmatch(entry.name)
        if match is None:
            errors.append(f"Unexpected filename for {rule.role}: {relative_entry}")
            continue
        image_id = match.group("id")
        matches.setdefault(image_id, []).append(entry)

    expected = set(rule.expected_ids)
    observed = set(matches)
    for image_id in sorted(expected - observed):
        errors.append(
            f"Missing {rule.role} for {rule.split} ID {image_id} "
            f"in {rule.relative_directory.as_posix()}"
        )
    for image_id in sorted(observed - expected):
        errors.append(
            f"Unexpected {rule.split} ID {image_id} for {rule.role} "
            f"in {rule.relative_directory.as_posix()}"
        )
    for image_id in sorted(observed):
        paths = matches[image_id]
        if len(paths) > 1:
            names = ", ".join(path.name for path in paths)
            errors.append(
                f"Duplicate {rule.role} files for {rule.split} ID {image_id}: {names}"
            )

    return {
        image_id: paths[0]
        for image_id, paths in matches.items()
        if image_id in expected and len(paths) == 1
    }


def _assigned_split(image_id: str) -> str:
    if image_id in TEST_IDS:
        return "test"
    if image_id in TRAIN_IDS:
        return "train"
    if image_id in VALIDATION_IDS:
        return "validation"
    raise DriveIntegrityError(f"No provisional split assignment for ID {image_id}")


def _record_satisfies_split_requirements(record: ManifestRecord) -> bool:
    """Check annotation fields required by the record's deterministic split."""

    if record.image_id in TEST_IDS:
        return (
            record.assigned_split == "test"
            and not record.relative_vessel_mask_path
            and not record.sha256_vessel_mask
            and record.annotation_observer == UNAVAILABLE_ANNOTATION_OBSERVER
            and record.annotation_availability == TEST_ANNOTATION_AVAILABILITY
        )
    return (
        record.assigned_split in {"train", "validation"}
        and bool(record.relative_vessel_mask_path)
        and bool(record.sha256_vessel_mask)
        and record.annotation_observer == "first_manual"
        and record.annotation_availability == TRAINING_ANNOTATION_AVAILABILITY
    )


def _reject_local_test_annotations(dataset_root: Path, errors: list[str]) -> None:
    """Reject annotations excluded by the authoritative hidden-test profile."""

    directory = dataset_root / PROHIBITED_TEST_ANNOTATION_DIRECTORY
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        errors.append(
            "Unexpected local test-annotation path under the authoritative "
            f"hidden-test profile: {PROHIBITED_TEST_ANNOTATION_DIRECTORY.as_posix()}"
        )
        return
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        errors.append(
            f"Cannot list {PROHIBITED_TEST_ANNOTATION_DIRECTORY.as_posix()}: {exc}"
        )
        return
    for entry in entries:
        errors.append(
            "Local test vessel annotations are not accepted under the authoritative "
            "hidden-test profile: "
            f"{entry.relative_to(dataset_root).as_posix()}"
        )


def _build_record(
    dataset_root: Path,
    image_id: str,
    image_path: Path,
    vessel_path: Path | None,
    fov_path: Path,
    provenance_source: str,
) -> ManifestRecord:
    image_metadata = _read_image_metadata(image_path)
    fov_metadata = _read_image_metadata(fov_path)
    expected_size = (image_metadata.width, image_metadata.height)
    if (fov_metadata.width, fov_metadata.height) != expected_size:
        raise DriveIntegrityError(f"FOV mask dimensions do not match image for ID {image_id}")

    if vessel_path is None:
        relative_vessel_mask_path = ""
        annotation_observer = UNAVAILABLE_ANNOTATION_OBSERVER
        annotation_availability = TEST_ANNOTATION_AVAILABILITY
        vessel_file_format = "not_distributed"
        sha256_vessel_mask = ""
    else:
        vessel_metadata = _read_image_metadata(vessel_path)
        if (vessel_metadata.width, vessel_metadata.height) != expected_size:
            raise DriveIntegrityError(
                f"Vessel annotation dimensions do not match image for ID {image_id}"
            )
        relative_vessel_mask_path = vessel_path.relative_to(dataset_root).as_posix()
        annotation_observer = "first_manual"
        annotation_availability = TRAINING_ANNOTATION_AVAILABILITY
        vessel_file_format = vessel_metadata.file_format
        sha256_vessel_mask = _sha256(vessel_path)

    file_format = (
        f"image:{image_metadata.file_format}|vessel_mask:{vessel_file_format}|"
        f"fov_mask:{fov_metadata.file_format}"
    )
    return ManifestRecord(
        image_id=image_id,
        assigned_split=_assigned_split(image_id),
        relative_image_path=image_path.relative_to(dataset_root).as_posix(),
        relative_vessel_mask_path=relative_vessel_mask_path,
        relative_fov_mask_path=fov_path.relative_to(dataset_root).as_posix(),
        annotation_observer=annotation_observer,
        annotation_availability=annotation_availability,
        original_dimensions=(
            f"{image_metadata.height}x{image_metadata.width}x{image_metadata.channels}"
        ),
        file_format=file_format,
        sha256_image=_sha256(image_path),
        sha256_vessel_mask=sha256_vessel_mask,
        sha256_fov_mask=_sha256(fov_path),
        provenance_source=provenance_source or DEFAULT_PROVENANCE,
        integrity_check_status="pass",
    )


def validate_drive(
    dataset_root: str | Path,
    provenance_source: str = DEFAULT_PROVENANCE,
) -> IntegrityResult:
    """Validate DRIVE structure and return deterministic records without writes."""

    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise DriveIntegrityError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise DriveIntegrityError(f"Dataset root is not a directory: {root}")
    root = root.resolve()

    errors: list[str] = []
    scanned: dict[tuple[str, str], dict[str, Path]] = {}
    for rule in FILE_RULES:
        scanned[(rule.split, rule.role)] = _scan_rule(root, rule, errors)
    _reject_local_test_annotations(root, errors)

    records: list[ManifestRecord] = []
    test_images = scanned[("test", "image")]
    test_fov_masks = scanned[("test", "FOV mask")]
    for image_id in TEST_IDS:
        if not all(image_id in mapping for mapping in (test_images, test_fov_masks)):
            errors.append(f"Incomplete image/FOV correspondence for test ID {image_id}")
            continue
        try:
            records.append(
                _build_record(
                    dataset_root=root,
                    image_id=image_id,
                    image_path=test_images[image_id],
                    vessel_path=None,
                    fov_path=test_fov_masks[image_id],
                    provenance_source=provenance_source,
                )
            )
        except DriveIntegrityError as exc:
            errors.append(str(exc))

    training_images = scanned[("training", "image")]
    training_vessels = scanned[("training", "vessel annotation")]
    training_fov_masks = scanned[("training", "FOV mask")]
    for image_id in OFFICIAL_TRAINING_IDS:
        if not all(
            image_id in mapping
            for mapping in (training_images, training_vessels, training_fov_masks)
        ):
            errors.append(
                f"Incomplete image/vessel/FOV correspondence for training ID {image_id}"
            )
            continue
        try:
            records.append(
                _build_record(
                    dataset_root=root,
                    image_id=image_id,
                    image_path=training_images[image_id],
                    vessel_path=training_vessels[image_id],
                    fov_path=training_fov_masks[image_id],
                    provenance_source=provenance_source,
                )
            )
        except DriveIntegrityError as exc:
            errors.append(str(exc))

    if len(records) != 40:
        errors.append(f"Expected 40 split-complete records, found {len(records)}")

    return IntegrityResult(
        dataset_root=root,
        records=tuple(records),
        errors=tuple(errors),
    )


def _ensure_output_outside_raw(dataset_root: Path, output_path: Path) -> Path:
    target = output_path.expanduser().resolve(strict=False)
    raw_root = dataset_root.resolve()
    if target == raw_root or raw_root in target.parents:
        raise DriveIntegrityError(
            f"Generated output must be outside the immutable dataset root: {target}"
        )
    return target


def write_manifest(result: IntegrityResult, output_path: str | Path) -> Path:
    """Write a deterministic CSV manifest after complete structural success."""

    if not result.ok:
        raise DriveIntegrityError("Cannot write a manifest for a failed integrity result")
    target = _ensure_output_outside_raw(result.dataset_root, Path(output_path))
    target.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in sorted(result.records, key=lambda item: int(item.image_id)):
        writer.writerow(asdict(record))
    target.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return target


def write_integrity_report(result: IntegrityResult, output_path: str | Path) -> Path:
    """Write a deterministic JSON integrity report outside the raw tree."""

    target = _ensure_output_outside_raw(result.dataset_root, Path(output_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.report_payload(), indent=2, sort_keys=True) + "\n"
    target.write_text(payload, encoding="utf-8", newline="")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline structural validation and deterministic manifest generation "
            "for a local DRIVE tree. PROTOCOL_STATUS=PROVISIONAL."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Explicit path to the immutable DRIVE root containing training/ and test/.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Generated CSV path (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Generated JSON report path (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--provenance-source",
        default=DEFAULT_PROVENANCE,
        help="Human-reviewed source identifier; does not prove authenticity.",
    )
    return parser


def _print_warnings(warnings: Iterable[str]) -> None:
    print(f"PROTOCOL_STATUS={PROTOCOL_STATUS}")
    for warning in warnings:
        print(f"WARNING={warning}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline CLI and return a process exit code."""

    args = _build_parser().parse_args(argv)
    _print_warnings(PROTOCOL_WARNINGS)
    try:
        result = validate_drive(args.dataset_root, args.provenance_source)
    except DriveIntegrityError as exc:
        print("DRIVE_INTEGRITY_STATUS=FAIL")
        print(f"ERROR={exc}")
        return 2

    try:
        report_path = write_integrity_report(result, args.report_out)
        if not result.ok:
            print("DRIVE_INTEGRITY_STATUS=FAIL")
            for error in result.errors:
                print(f"ERROR={error}")
            print(f"INTEGRITY_REPORT={report_path}")
            return 2
        manifest_path = write_manifest(result, args.manifest_out)
    except (DriveIntegrityError, OSError) as exc:
        print("DRIVE_INTEGRITY_STATUS=FAIL")
        print(f"ERROR={exc}")
        return 2

    print("DRIVE_INTEGRITY_STATUS=PASS")
    print(f"RECORD_COUNT={len(result.records)}")
    print(f"MANIFEST={manifest_path}")
    print(f"INTEGRITY_REPORT={report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
