"""Frozen scientific constants from the published Claim 3 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final


class ClassId(IntEnum):
    BACKGROUND = 0
    BIPOLAR_FORCEPS = 1
    PROGRASP_FORCEPS = 2
    LARGE_NEEDLE_DRIVER = 3
    VESSEL_SEALER = 4
    GRASPING_RETRACTOR = 5
    MONOPOLAR_CURVED_SCISSORS = 6
    OTHER = 7


class Split:
    TRAIN: Final = "train"
    VALIDATION: Final = "validation"
    TEST: Final = "test"


CLASS_NAMES: Final = (
    "Background",
    "Bipolar Forceps",
    "Prograsp Forceps",
    "Large Needle Driver",
    "Vessel Sealer",
    "Grasping Retractor",
    "Monopolar Curved Scissors",
    "Other",
)
IGNORE_INDEX: Final = 255
PART_VALUES: Final = frozenset((10, 20, 30, 40))
SEEDS: Final = (42, 7, 123)
TRAIN_SEQUENCES: Final = (2, 3, 7, 8)
VALIDATION_SEQUENCES: Final = (4,)
TEST_SEQUENCES: Final = (1, 5, 6)
ELIGIBLE_FRAME_IDS: Final = tuple(range(225))


@dataclass(frozen=True)
class Claim3Contract:
    dataset_revision: str = "518d8a542b83b6af8cf2c37e4aa210b218655248"
    umamba_revision: str = "28459e33ca03769800dd35e23c6e62491d1925b5"
    s2mnet_revision: str = "3ec59668ab9b438ab9b170306d29b01e9270fd5a"
    image_size: int = 384
    num_classes: int = 8
    ignore_index: int = IGNORE_INDEX
    effective_batch_size: int = 4
    initial_micro_batch_size: int = 2
    gradient_accumulation: int = 2
    maximum_optimizer_steps: int = 10_125
    validation_frequency: int = 225
    training_passes: int = 45
    processed_samples: int = 40_500
    peak_learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    output_root: Path = Path("outputs/claim3")


CONTRACT: Final = Claim3Contract()


def split_for_sequence(sequence: int) -> str:
    if sequence in TRAIN_SEQUENCES:
        return Split.TRAIN
    if sequence in VALIDATION_SEQUENCES:
        return Split.VALIDATION
    if sequence in TEST_SEQUENCES:
        return Split.TEST
    raise ValueError(f"Sequence {sequence} is outside the frozen split")


def run_matrix() -> tuple[tuple[str, int], ...]:
    return tuple((model, seed) for seed in SEEDS for model in ("s2mnet", "umamba"))


def learning_rate(step: int) -> float:
    """Learning rate for one-indexed optimizer step."""
    if not 1 <= step <= CONTRACT.maximum_optimizer_steps:
        raise ValueError(f"step must be in [1, {CONTRACT.maximum_optimizer_steps}]")
    if step <= CONTRACT.warmup_steps:
        return CONTRACT.peak_learning_rate * step / CONTRACT.warmup_steps
    import math

    q = (step - CONTRACT.warmup_steps) / (
        CONTRACT.maximum_optimizer_steps - CONTRACT.warmup_steps
    )
    return CONTRACT.minimum_learning_rate + 0.5 * (
        CONTRACT.peak_learning_rate - CONTRACT.minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * q))


def validate_contract() -> None:
    all_sequences = (
        set(TRAIN_SEQUENCES) | set(VALIDATION_SEQUENCES) | set(TEST_SEQUENCES)
    )
    assert all_sequences == set(range(1, 9))
    assert not (
        set(TRAIN_SEQUENCES) & set(VALIDATION_SEQUENCES)
        or set(TRAIN_SEQUENCES) & set(TEST_SEQUENCES)
        or set(VALIDATION_SEQUENCES) & set(TEST_SEQUENCES)
    )
    assert len(ELIGIBLE_FRAME_IDS) == 225
    assert len(TRAIN_SEQUENCES) * 225 == 900
    assert len(VALIDATION_SEQUENCES) * 225 == 225
    assert len(TEST_SEQUENCES) * 225 == 675
    assert len(all_sequences) * 225 == 1_800
    assert CONTRACT.training_passes * 900 == CONTRACT.processed_samples
    assert (
        CONTRACT.processed_samples // CONTRACT.effective_batch_size
        == CONTRACT.maximum_optimizer_steps
    )
    assert (
        CONTRACT.initial_micro_batch_size * CONTRACT.gradient_accumulation
        == CONTRACT.effective_batch_size
    )
    assert CONTRACT.maximum_optimizer_steps % CONTRACT.validation_frequency == 0
    assert CONTRACT.image_size % 64 == 0
    assert len(CLASS_NAMES) == CONTRACT.num_classes
    assert len(run_matrix()) == 6


validate_contract()
