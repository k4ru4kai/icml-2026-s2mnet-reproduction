"""Framework-neutral EndoVis17 inventory, target, and transform pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

from .contract import (
    CLASS_NAMES,
    CONTRACT,
    ELIGIBLE_FRAME_IDS,
    IGNORE_INDEX,
    PART_VALUES,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    VALIDATION_SEQUENCES,
    ClassId,
    split_for_sequence,
)


@dataclass(frozen=True)
class MaskRecord:
    path: Path
    class_id: int


@dataclass(frozen=True)
class FrameRecord:
    sequence: int
    frame_id: int
    split: str
    image_path: Path
    masks: tuple[MaskRecord, ...]

    @property
    def key(self) -> str:
        return f"instrument_dataset_{self.sequence}/frame{self.frame_id:03d}"


def class_from_directory(name: str) -> int:
    token = name.lower()
    if "bipolar_forceps" in token:
        return int(ClassId.BIPOLAR_FORCEPS)
    if "prograsp_forceps" in token:
        return int(ClassId.PROGRASP_FORCEPS)
    if "large_needle_driver" in token:
        return int(ClassId.LARGE_NEEDLE_DRIVER)
    if "vessel_sealer" in token:
        return int(ClassId.VESSEL_SEALER)
    if "grasping_retractor" in token:
        return int(ClassId.GRASPING_RETRACTOR)
    if "monopolar_curved_scissors" in token:
        return int(ClassId.MONOPOLAR_CURVED_SCISSORS)
    if token == "other_labels":
        return int(ClassId.OTHER)
    raise ValueError(f"Unknown instrument annotation directory: {name}")


def _load_uint8_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar part values and invalid pixels under the frozen decoder."""
    with Image.open(path) as image:
        array = np.asarray(image)
        mode = image.mode
    if array.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 mask, got {array.dtype}")
    if mode == "L" and array.ndim == 2:
        values = array
        invalid_encoding = np.zeros(array.shape, dtype=bool)
    elif mode in {"RGB", "RGBA"} and array.ndim == 3:
        rgb = array[..., :3]
        values = rgb[..., 0]
        invalid_encoding = (rgb[..., 0] != rgb[..., 1]) | (
            rgb[..., 1] != rgb[..., 2]
        )
        if mode == "RGBA":
            invalid_encoding |= array[..., 3] != 255
    else:
        raise ValueError(f"{path}: unsupported mask mode/shape {mode}/{array.shape}")
    valid_value = np.isin(values, (0, *sorted(PART_VALUES)))
    return values, invalid_encoding | ~valid_value


def decode_instrument_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values, invalid = _load_uint8_mask(path)
    foreground = np.isin(values, tuple(PART_VALUES)) & ~invalid
    return foreground, invalid


def combine_type_masks(
    masks: Sequence[MaskRecord],
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    if not masks:
        raise ValueError("At least one per-instrument mask is required")
    target: np.ndarray | None = None
    invalid_union: np.ndarray | None = None
    conflict_count = 0
    for mask_record in masks:
        foreground, invalid = decode_instrument_mask(mask_record.path)
        if target is None:
            target = np.zeros(foreground.shape, dtype=np.uint8)
            invalid_union = np.zeros(foreground.shape, dtype=bool)
        if foreground.shape != target.shape:
            raise ValueError(
                f"{mask_record.path}: shape {foreground.shape} differs from {target.shape}"
            )
        assert invalid_union is not None
        invalid_union |= invalid
        unassigned = foreground & (target == int(ClassId.BACKGROUND))
        same = foreground & (target == mask_record.class_id)
        conflict = foreground & ~(unassigned | same) & (target != IGNORE_INDEX)
        conflict_count += int(conflict.sum())
        target[unassigned] = mask_record.class_id
        target[conflict] = IGNORE_INDEX
    assert target is not None and invalid_union is not None
    if expected_shape is not None and target.shape != expected_shape:
        raise ValueError(f"Combined target shape {target.shape} != {expected_shape}")
    target[invalid_union] = IGNORE_INDEX
    return target, {
        "conflict_pixels": conflict_count,
        "invalid_pixels": int(invalid_union.sum()),
        "ignored_pixels": int((target == IGNORE_INDEX).sum()),
    }


def build_inventory(dataset_root: Path) -> tuple[FrameRecord, ...]:
    training_root = dataset_root / "training"
    if not training_root.is_dir():
        raise FileNotFoundError(f"Missing training directory: {training_root}")
    records: list[FrameRecord] = []
    for sequence in range(1, 9):
        sequence_root = training_root / f"instrument_dataset_{sequence}"
        image_root = sequence_root / "left_frames"
        ground_truth = sequence_root / "ground_truth"
        if not image_root.is_dir() or not ground_truth.is_dir():
            raise FileNotFoundError(f"Incomplete sequence layout: {sequence_root}")
        annotation_dirs = sorted(path for path in ground_truth.iterdir() if path.is_dir())
        if not annotation_dirs:
            raise ValueError(f"No annotation directories in {ground_truth}")
        class_dirs = tuple(
            (annotation_dir, class_from_directory(annotation_dir.name))
            for annotation_dir in annotation_dirs
        )
        for frame_id in ELIGIBLE_FRAME_IDS:
            stem = f"frame{frame_id:03d}.png"
            image_path = image_root / stem
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")
            masks = tuple(
                MaskRecord(annotation_dir / stem, class_id)
                for annotation_dir, class_id in class_dirs
            )
            missing = [str(mask.path) for mask in masks if not mask.path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{sequence_root}: missing {len(missing)} masks; first={missing[0]}"
                )
            records.append(
                FrameRecord(
                    sequence=sequence,
                    frame_id=frame_id,
                    split=split_for_sequence(sequence),
                    image_path=image_path,
                    masks=masks,
                )
            )
    if len(records) != 1_800:
        raise AssertionError(f"Expected 1800 records, found {len(records)}")
    return tuple(records)


def records_for_split(records: Sequence[FrameRecord], split: str) -> tuple[FrameRecord, ...]:
    return tuple(record for record in records if record.split == split)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_payload(
    dataset_root: Path,
    records: Sequence[FrameRecord],
    include_hashes: bool = True,
) -> dict:
    rows = []
    for record in records:
        row = {
            "key": record.key,
            "sequence": record.sequence,
            "frame_id": record.frame_id,
            "split": record.split,
            "image": str(record.image_path.relative_to(dataset_root)),
            "masks": [
                {
                    "path": str(mask.path.relative_to(dataset_root)),
                    "class_id": mask.class_id,
                    "class_name": CLASS_NAMES[mask.class_id],
                }
                for mask in record.masks
            ],
        }
        if include_hashes:
            target, target_stats = combine_type_masks(record.masks)
            target_stats["class_pixel_counts"] = {
                str(class_id): int(np.count_nonzero(target == class_id))
                for class_id in range(len(CLASS_NAMES))
            }
            row["image_sha256"] = _sha256_file(record.image_path)
            row["mask_sha256"] = {
                str(mask.path.relative_to(dataset_root)): _sha256_file(mask.path)
                for mask in record.masks
            }
            row["combined_target_sha256"] = hashlib.sha256(target.tobytes()).hexdigest()
            row["target_stats"] = target_stats
        rows.append(row)
    return {
        "schema_version": 1,
        "dataset_revision": CONTRACT.dataset_revision,
        "eligible_frames": "000-224",
        "record_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "records": rows,
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def compute_training_normalization(
    records: Sequence[FrameRecord],
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    count = 0
    for record in records:
        if record.split != "train":
            continue
        image = load_rgb(record.image_path).astype(np.float64)
        pixels = image.reshape(-1, 3)
        total += pixels.sum(axis=0)
        squared += np.square(pixels).sum(axis=0)
        count += pixels.shape[0]
    if count == 0:
        raise ValueError("No training pixels available for normalization")
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0):
        raise ValueError(f"Non-positive training standard deviation: {std}")
    return mean, std


def counter_rng(seed: int, training_pass: int, shuffled_position: int) -> np.random.Generator:
    key = f"{seed}:{training_pass}:{shuffled_position}".encode()
    counter_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    return np.random.default_rng(counter_seed)


def resize_image_target(
    image: np.ndarray, target: np.ndarray, size: int = 384
) -> tuple[np.ndarray, np.ndarray]:
    resized_image = cv2.resize(
        image, (size, size), interpolation=cv2.INTER_LINEAR_EXACT
    )
    resized_target = cv2.resize(
        target, (size, size), interpolation=cv2.INTER_NEAREST_EXACT
    )
    return resized_image, resized_target


def augment_training(
    image: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.shape
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        target = np.ascontiguousarray(target[:, ::-1])
    if rng.random() < 0.7:
        angle = float(rng.uniform(-15.0, 15.0))
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        target = cv2.warpAffine(
            target,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=IGNORE_INDEX,
        )
    if rng.random() < 0.8:
        contrast = float(rng.uniform(0.8, 1.2))
        brightness = float(rng.uniform(-0.2, 0.2))
        image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    if rng.random() < 0.2:
        sigma = float(rng.uniform(0.1, 1.5))
        image = cv2.GaussianBlur(image, (5, 5), sigmaX=sigma, sigmaY=sigma)
    if rng.random() < 0.3:
        sigma = float(rng.uniform(0.0, 0.05))
        image = np.clip(
            image + rng.normal(0.0, sigma, size=image.shape).astype(np.float32),
            0.0,
            1.0,
        )
    return image.astype(np.float32), target.astype(np.uint8)


def prepare_sample(
    record: FrameRecord,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    training_pass: int | None = None,
    shuffled_position: int | None = None,
    seed: int | None = None,
    size: int = 384,
) -> tuple[np.ndarray, np.ndarray]:
    image = load_rgb(record.image_path)
    target, _ = combine_type_masks(record.masks, expected_shape=image.shape[:2])
    image, target = resize_image_target(image, target, size=size)
    if training_pass is not None:
        if seed is None or shuffled_position is None:
            raise ValueError("Training augmentation requires seed and shuffled_position")
        image, target = augment_training(
            image, target, counter_rng(seed, training_pass, shuffled_position)
        )
    image = ((image.astype(np.float32) - mean) / std).astype(np.float32)
    return image, target


def shuffled_training_pass(
    records: Sequence[FrameRecord], seed: int, training_pass: int
) -> tuple[FrameRecord, ...]:
    train = list(records_for_split(records, "train"))
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.blake2b(f"order:{seed}:{training_pass}".encode(), digest_size=8).digest(),
            "little",
        )
    )
    order = rng.permutation(len(train))
    return tuple(train[index] for index in order)


def effective_batches(
    records: Sequence[FrameRecord], seed: int
) -> Iterator[tuple[int, int, tuple[FrameRecord, ...]]]:
    optimizer_step = 0
    for training_pass in range(45):
        shuffled = shuffled_training_pass(records, seed, training_pass)
        for start in range(0, len(shuffled), 4):
            batch = shuffled[start : start + 4]
            if len(batch) != 4:
                raise AssertionError("Frozen 900-sample pass must divide into batches of 4")
            optimizer_step += 1
            yield optimizer_step, training_pass, batch
    if optimizer_step != 10_125:
        raise AssertionError(f"Expected 10125 steps, generated {optimizer_step}")
