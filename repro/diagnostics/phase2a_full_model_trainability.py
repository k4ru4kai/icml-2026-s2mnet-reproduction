#!/usr/bin/env python3
"""Run one synthetic optimization step through the official DRIVE Full Model."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

# Avoid writing import bytecode into the official repository.
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = PROJECT_ROOT / "official_repo"
CONFIG_PATH = OFFICIAL_REPO / "configs" / "retinal.yaml"

SEED = 42
LEARNING_RATE = 1.0e-4
EXPECTED_INPUT_SHAPE = (1, 256, 256, 3)
EXPECTED_MASK_SHAPE = (1, 256, 256, 1)


def git_status(repo: Path) -> str:
    """Return a complete porcelain status without changing the repository."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def parameter_count(weights: list[tf.Variable]) -> int:
    return sum(int(np.prod(weight.shape)) for weight in weights)


def finite_counts(tensors: list[object]) -> tuple[int, int, int]:
    """Return value, NaN, and Inf counts for dense or indexed tensors."""
    value_count = 0
    nan_count = 0
    inf_count = 0
    for tensor in tensors:
        values = tensor.values if isinstance(tensor, tf.IndexedSlices) else tensor
        values = tf.convert_to_tensor(values)
        value_count += int(tf.size(values).numpy())
        nan_count += int(tf.math.count_nonzero(tf.math.is_nan(values)).numpy())
        inf_count += int(tf.math.count_nonzero(tf.math.is_inf(values)).numpy())
    return value_count, nan_count, inf_count


def configure_single_gpu() -> tuple[str, str, int]:
    """Require and select one physical GPU before TensorFlow initializes it."""
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise RuntimeError("No TensorFlow GPU is visible; CPU fallback is disabled.")

    selected_gpu = physical_gpus[0]
    tf.config.set_visible_devices([selected_gpu], "GPU")
    tf.config.experimental.set_memory_growth(selected_gpu, True)

    logical_gpus = tf.config.list_logical_devices("GPU")
    if len(logical_gpus) != 1:
        raise RuntimeError(f"Expected one logical GPU, found {len(logical_gpus)}.")
    return selected_gpu.name, logical_gpus[0].name, len(physical_gpus)


def main() -> None:
    official_status_before = git_status(OFFICIAL_REPO)
    physical_gpu, logical_gpu, physical_gpu_count = configure_single_gpu()

    # Use exactly the same official entry point and DRIVE configuration as the
    # preceding Phase 2A forward-pass diagnostic.
    sys.path.insert(0, str(OFFICIAL_REPO))
    official_train = importlib.import_module("train")
    official_train.set_seed(SEED)

    cfg = official_train.load_config(str(CONFIG_PATH), [])
    model_cfg = cfg["model"]
    full_model_flags = {
        "bfp_routing": model_cfg.get("bfp_routing", "soft"),
        "sstm_stages": model_cfg.get("sstm_stages", [True] * 5),
        "use_bfp": model_cfg.get("use_bfp", True),
        "use_mrfse": model_cfg.get("use_mrfse", True),
        "use_sstm": model_cfg.get("use_sstm", True),
    }
    all_modules_configured = bool(
        full_model_flags["use_mrfse"]
        and full_model_flags["use_sstm"]
        and all(full_model_flags["sstm_stages"])
        and full_model_flags["use_bfp"]
    )

    input_size = int(model_cfg["input_size"])
    if input_size != 256:
        raise RuntimeError(f"Expected DRIVE input_size=256, found {input_size}.")

    with tf.device(logical_gpu):
        model = official_train.build_model(cfg)

        # Stateless generation makes the only synthetic image and mask exactly
        # reproducible without loading or downloading DRIVE.
        synthetic_input = tf.random.stateless_uniform(
            EXPECTED_INPUT_SHAPE,
            seed=(SEED, 0),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )
        synthetic_mask = tf.cast(
            tf.random.stateless_uniform(
                EXPECTED_MASK_SHAPE,
                seed=(SEED, 1),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
            >= 0.5,
            tf.float32,
        )

        # This clipped Binary Cross-Entropy is intentionally a simple numerical
        # diagnostic. It is NOT the paper's experimental Morphology-Aware Loss.
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name="diagnostic_binary_crossentropy",
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        trainable_variables = list(model.trainable_variables)
        weights_before_update = [tf.identity(weight) for weight in trainable_variables]
        weight_value_count_before, weight_nan_before, weight_inf_before = finite_counts(
            list(model.weights)
        )

        # Reset the global TensorFlow seed immediately before both training-mode
        # forwards so dropout follows the same deterministic random sequence.
        tf.random.set_seed(SEED)
        with tf.GradientTape() as tape:
            predictions_before = model(synthetic_input, training=True)
            loss_before = loss_fn(synthetic_mask, predictions_before)

        gradients = tape.gradient(loss_before, trainable_variables)
        none_gradient_count = sum(gradient is None for gradient in gradients)
        non_none_gradients = [gradient for gradient in gradients if gradient is not None]
        gradient_value_count, gradient_nan_count, gradient_inf_count = finite_counts(
            non_none_gradients
        )
        global_gradient_norm = tf.linalg.global_norm(non_none_gradients)

        optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, trainable_variables)
            if gradient is not None
        )

        changed_variable_count = 0
        changed_value_count = 0
        max_abs_weight_delta = 0.0
        for before, after in zip(weights_before_update, trainable_variables):
            delta = tf.abs(after - before)
            tensor_changed_values = int(tf.math.count_nonzero(delta).numpy())
            changed_value_count += tensor_changed_values
            changed_variable_count += int(tensor_changed_values > 0)
            max_abs_weight_delta = max(
                max_abs_weight_delta,
                float(tf.reduce_max(delta).numpy()),
            )

        tf.random.set_seed(SEED)
        predictions_after = model(synthetic_input, training=True)
        loss_after = loss_fn(synthetic_mask, predictions_after)

        _, prediction_nan_before, prediction_inf_before = finite_counts(
            [predictions_before]
        )
        _, prediction_nan_after, prediction_inf_after = finite_counts(
            [predictions_after]
        )
        weight_value_count_after, weight_nan_after, weight_inf_after = finite_counts(
            list(model.weights)
        )

    mrfse_stages = sum(
        layer.name.startswith("mrfse_stage") for layer in model.layers
    )
    sstm_stages = sum(layer.name.startswith("sstm_stage") for layer in model.layers)
    bfp_stages = sum(layer.name.startswith("bfp_stage") for layer in model.layers)
    full_model_topology_ok = (
        all_modules_configured
        and mrfse_stages == 5
        and sstm_stages == 5
        and bfp_stages == 4
    )

    loss_before_value = float(loss_before.numpy())
    loss_after_value = float(loss_after.numpy())
    global_gradient_norm_value = float(global_gradient_norm.numpy())
    optimizer_iterations = int(optimizer.iterations.numpy())
    official_status_after = git_status(OFFICIAL_REPO)
    official_repo_unchanged = official_status_after == official_status_before
    official_repo_clean = official_status_after == ""

    checks = {
        "full_model_topology": full_model_topology_ok,
        "input_shape": tuple(synthetic_input.shape) == EXPECTED_INPUT_SHAPE,
        "mask_shape": tuple(synthetic_mask.shape) == EXPECTED_MASK_SHAPE,
        "mask_binary": set(np.unique(synthetic_mask.numpy()).tolist()) <= {0.0, 1.0},
        "trainable_variables_present": len(trainable_variables) > 0,
        "no_none_gradients": none_gradient_count == 0,
        "gradients_finite": gradient_nan_count == 0 and gradient_inf_count == 0,
        "gradient_norm_positive_finite": bool(
            np.isfinite(global_gradient_norm_value) and global_gradient_norm_value > 0.0
        ),
        "losses_finite": bool(
            np.isfinite(loss_before_value) and np.isfinite(loss_after_value)
        ),
        "predictions_finite": (
            prediction_nan_before == 0
            and prediction_inf_before == 0
            and prediction_nan_after == 0
            and prediction_inf_after == 0
        ),
        "weights_finite": (
            weight_nan_before == 0
            and weight_inf_before == 0
            and weight_nan_after == 0
            and weight_inf_after == 0
        ),
        "one_optimizer_update": optimizer_iterations == 1,
        "at_least_one_weight_changed": changed_variable_count > 0,
        "official_repo_unchanged": official_repo_unchanged,
        "official_repo_clean": official_repo_clean,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    print("=== PHASE2A_FULL_MODEL_TRAINABILITY_BEGIN ===")
    print("ENTRY_POINT=train.build_model")
    print(f"CONFIG={CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"SEED={SEED}")
    print(f"FULL_MODEL_FLAGS={json.dumps(full_model_flags, sort_keys=True)}")
    print(f"FULL_MODEL_ALL_MODULES_CONFIGURED={all_modules_configured}")
    print(f"MRFSE_STAGES={mrfse_stages}")
    print(f"SSTM_STAGES={sstm_stages}")
    print(f"BFP_STAGES={bfp_stages}")
    print(f"FULL_MODEL_TOPOLOGY_OK={full_model_topology_ok}")
    print(f"PHYSICAL_GPU_COUNT={physical_gpu_count}")
    print(f"SELECTED_PHYSICAL_GPU={physical_gpu}")
    print(f"SELECTED_LOGICAL_GPU={logical_gpu}")
    print(f"TRAINING_OUTPUT_DEVICE={predictions_before.device}")
    print(f"SYNTHETIC_INPUT_SHAPE={tuple(synthetic_input.shape)}")
    print(f"SYNTHETIC_INPUT_DTYPE={synthetic_input.dtype.name}")
    print(
        "SYNTHETIC_INPUT_RANGE="
        f"[{float(tf.reduce_min(synthetic_input).numpy()):.9g}, "
        f"{float(tf.reduce_max(synthetic_input).numpy()):.9g}]"
    )
    print(f"SYNTHETIC_MASK_SHAPE={tuple(synthetic_mask.shape)}")
    print(f"SYNTHETIC_MASK_DTYPE={synthetic_mask.dtype.name}")
    print(f"SYNTHETIC_MASK_VALUES={np.unique(synthetic_mask.numpy()).tolist()}")
    print(f"SYNTHETIC_MASK_POSITIVE_FRACTION={float(tf.reduce_mean(synthetic_mask).numpy()):.9g}")
    print("LOSS=BinaryCrossentropy(from_logits=False)")
    print("LOSS_IS_PAPER_EXPERIMENTAL_LOSS=False")
    print("MODEL_REGULARIZATION_TERMS_INCLUDED_IN_LOSS=False")
    print(f"LOSS_BEFORE_UPDATE={loss_before_value:.9g}")
    print(f"LOSS_AFTER_UPDATE={loss_after_value:.9g}")
    print(f"LOSS_DELTA={loss_after_value - loss_before_value:.9g}")
    print("LOSS_DECREASE_REQUIRED=False")
    print(f"TRAINABLE_VARIABLE_COUNT={len(trainable_variables)}")
    print(f"TRAINABLE_PARAMETER_COUNT={parameter_count(trainable_variables)}")
    print(f"GRADIENT_NONE_COUNT={none_gradient_count}")
    print(f"GRADIENT_VALUE_COUNT={gradient_value_count}")
    print(f"GRADIENT_NAN_COUNT={gradient_nan_count}")
    print(f"GRADIENT_INF_COUNT={gradient_inf_count}")
    print(f"GLOBAL_GRADIENT_NORM={global_gradient_norm_value:.9g}")
    print(f"PREDICTION_BEFORE_NAN_COUNT={prediction_nan_before}")
    print(f"PREDICTION_BEFORE_INF_COUNT={prediction_inf_before}")
    print(f"PREDICTION_AFTER_NAN_COUNT={prediction_nan_after}")
    print(f"PREDICTION_AFTER_INF_COUNT={prediction_inf_after}")
    print(f"WEIGHT_VALUE_COUNT_BEFORE={weight_value_count_before}")
    print(f"WEIGHT_NAN_COUNT_BEFORE={weight_nan_before}")
    print(f"WEIGHT_INF_COUNT_BEFORE={weight_inf_before}")
    print(f"WEIGHT_VALUE_COUNT_AFTER={weight_value_count_after}")
    print(f"WEIGHT_NAN_COUNT_AFTER={weight_nan_after}")
    print(f"WEIGHT_INF_COUNT_AFTER={weight_inf_after}")
    print("OPTIMIZER=Adam")
    print(f"LEARNING_RATE={LEARNING_RATE:.9g}")
    print(f"OPTIMIZER_ITERATIONS={optimizer_iterations}")
    print(f"CHANGED_TRAINABLE_VARIABLE_COUNT={changed_variable_count}")
    print(f"CHANGED_TRAINABLE_VALUE_COUNT={changed_value_count}")
    print(f"MAX_ABS_WEIGHT_DELTA={max_abs_weight_delta:.9g}")
    print(f"OFFICIAL_REPO_STATUS_BEFORE={json.dumps(official_status_before.splitlines())}")
    print(f"OFFICIAL_REPO_STATUS_AFTER={json.dumps(official_status_after.splitlines())}")
    print(f"OFFICIAL_REPO_UNCHANGED={official_repo_unchanged}")
    print(f"OFFICIAL_REPO_CLEAN={official_repo_clean}")
    print(f"CHECKS={json.dumps(checks, sort_keys=True)}")
    print(f"FAILED_CHECK_COUNT={len(failed_checks)}")
    print(f"FAILED_CHECKS={json.dumps(failed_checks)}")
    print(f"OVERALL_STATUS={'PASS' if not failed_checks else 'FAIL'}")
    print("=== PHASE2A_FULL_MODEL_TRAINABILITY_END ===")

    if failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
