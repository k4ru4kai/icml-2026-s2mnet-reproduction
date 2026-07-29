from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from repro.claim3.contract import (
    CONTRACT,
    IGNORE_INDEX,
    SEEDS,
    run_matrix,
    validate_contract,
)
from repro.claim3.data import (
    MaskRecord,
    augment_training,
    build_inventory,
    class_from_directory,
    combine_type_masks,
    counter_rng,
    decode_instrument_mask,
    records_for_split,
)
from repro.claim3.metrics import (
    PooledForegroundMetrics,
    common_loss_numpy,
    native_prediction,
)
from repro.claim3.runtime import run_directory, write_json
from repro.experiments.claim3_train import summarize_campaign


DATASET_ROOT = Path("/home/sarah/Datasets/EndoVis17_HF")


def _mask(path: Path, values: np.ndarray, mode: str = "L") -> Path:
    Image.fromarray(values, mode=mode).save(path)
    return path


def test_exact_real_inventory_and_sequence_split():
    if not DATASET_ROOT.is_dir():
        pytest.skip("Local EndoVis17 dataset is unavailable")
    records = build_inventory(DATASET_ROOT)
    assert len(records) == 1_800
    assert len(records_for_split(records, "train")) == 900
    assert len(records_for_split(records, "validation")) == 225
    assert len(records_for_split(records, "test")) == 675
    assert {record.sequence for record in records_for_split(records, "train")} == {
        2,
        3,
        7,
        8,
    }
    assert {record.sequence for record in records_for_split(records, "validation")} == {4}
    assert {record.sequence for record in records_for_split(records, "test")} == {
        1,
        5,
        6,
    }
    assert min(record.frame_id for record in records) == 0
    assert max(record.frame_id for record in records) == 224


def test_class_mapping_and_ignore_index():
    assert class_from_directory("Maryland_Bipolar_Forceps_labels") == 1
    assert class_from_directory("Left_Prograsp_Forceps_labels") == 2
    assert class_from_directory("Large_Needle_Driver_Right_labels") == 3
    assert class_from_directory("Right_Vessel_Sealer") == 4
    assert class_from_directory("Left_Grasping_Retractor_labels") == 5
    assert class_from_directory("Monopolar_Curved_Scissors_labels") == 6
    assert class_from_directory("Other_labels") == 7
    assert IGNORE_INDEX == 255
    with pytest.raises(ValueError, match="Unknown instrument"):
        class_from_directory("Ultrasound_Probe")


def test_sequence7_rgb_decoding_and_noncanonical_ignore(tmp_path):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 10, 10)
    rgb[0, 1] = (20, 20, 20)
    rgb[0, 2] = (10, 11, 10)
    rgb[1, 0] = (31, 31, 31)
    path = _mask(tmp_path / "sequence7.png", rgb, "RGB")
    foreground, invalid = decode_instrument_mask(path)
    assert foreground[0, 0]
    assert foreground[0, 1]
    assert invalid[0, 2]
    assert invalid[1, 0]
    assert not foreground[0, 2]


def test_overlapping_types_are_ignored_but_same_type_is_unioned(tmp_path):
    first = np.array([[10, 10], [0, 0]], dtype=np.uint8)
    second = np.array([[0, 20], [20, 0]], dtype=np.uint8)
    third = np.array([[0, 30], [0, 30]], dtype=np.uint8)
    records = [
        MaskRecord(_mask(tmp_path / "a.png", first), 1),
        MaskRecord(_mask(tmp_path / "b.png", second), 1),
        MaskRecord(_mask(tmp_path / "c.png", third), 2),
    ]
    target, stats = combine_type_masks(records)
    assert target.tolist() == [[1, IGNORE_INDEX], [1, 2]]
    assert stats["conflict_pixels"] == 1
    assert stats["ignored_pixels"] == 1


def test_missing_and_malformed_masks_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_instrument_mask(tmp_path / "missing.png")
    malformed = np.zeros((2, 2), dtype=np.uint16)
    Image.fromarray(malformed).save(tmp_path / "malformed.png")
    with pytest.raises(ValueError, match="expected uint8"):
        decode_instrument_mask(tmp_path / "malformed.png")


def test_augmentation_is_deterministic_and_preserves_discrete_target():
    image = np.linspace(0, 1, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    target = np.zeros((16, 16), dtype=np.uint8)
    target[4:12, 4:12] = 3
    first = augment_training(image.copy(), target.copy(), counter_rng(42, 2, 17))
    second = augment_training(image.copy(), target.copy(), counter_rng(42, 2, 17))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(np.unique(first[1])).issubset({0, 3, IGNORE_INDEX})


def test_loss_masking_removes_ignored_pixels():
    target = np.array([[[1, IGNORE_INDEX], [0, 2]]], dtype=np.uint8)
    logits = np.zeros((1, 2, 2, 8), dtype=np.float64)
    logits[0, 0, 0, 1] = 4
    logits[0, 1, 0, 0] = 4
    logits[0, 1, 1, 2] = 4
    baseline = common_loss_numpy(logits, target)
    logits[0, 0, 1] = np.array([100, -100, 90, 80, 70, 60, 50, 40])
    changed = common_loss_numpy(logits, target)
    assert baseline == pytest.approx(changed)


def test_pooled_dice_absent_rule_and_background_exclusion():
    target = np.array([[0, 1, 1, IGNORE_INDEX], [0, 2, 2, 0]], dtype=np.uint8)
    prediction = np.array([[7, 1, 0, 3], [7, 2, 2, 0]], dtype=np.uint8)
    metrics = PooledForegroundMetrics()
    metrics.update(target, prediction)
    result = metrics.result()
    assert result["classes"]["1"]["dice"] == pytest.approx(2 / 3)
    assert result["classes"]["2"]["dice"] == pytest.approx(1.0)
    assert result["classes"]["7"]["dice"] == pytest.approx(0.0)
    assert result["classes"]["3"]["dice"] is None
    assert "0" not in result["classes"]


def test_identical_evaluation_inputs_produce_identical_predictions():
    probabilities = np.zeros((4, 4, 8), dtype=np.float32)
    probabilities[..., 3] = 1.0
    first = native_prediction(probabilities.copy(), (9, 13))
    second = native_prediction(probabilities.copy(), (9, 13))
    assert np.array_equal(first, second)
    assert first.shape == (9, 13)
    assert np.all(first == 3)


def test_run_matrix_and_processed_sample_budget():
    validate_contract()
    assert run_matrix() == (
        ("s2mnet", 42),
        ("umamba", 42),
        ("s2mnet", 7),
        ("umamba", 7),
        ("s2mnet", 123),
        ("umamba", 123),
    )
    assert len(SEEDS) == 3
    assert CONTRACT.maximum_optimizer_steps * CONTRACT.effective_batch_size == 40_500
    assert CONTRACT.gradient_accumulation * CONTRACT.initial_micro_batch_size == 4


def test_paired_summary_uses_all_three_predeclared_seeds(tmp_path):
    for seed in SEEDS:
        for model, offset in (("s2mnet", 0.1), ("umamba", 0.0)):
            run_root = run_directory(tmp_path, model, seed)
            write_json(run_root / "status.json", {"status": "completed"})
            write_json(
                run_root / "metrics" / "test.json",
                {"foreground_macro_dice": seed / 1_000 + offset},
            )
    summary = summarize_campaign(tmp_path)
    assert len(summary["runs"]) == 6
    assert [row["seed"] for row in summary["paired_differences"]] == list(SEEDS)
    assert summary["paired_difference_mean"] == pytest.approx(0.1)
    assert summary["paired_difference_sample_std"] == pytest.approx(0.0)
