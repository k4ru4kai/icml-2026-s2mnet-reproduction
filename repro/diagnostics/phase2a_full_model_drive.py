#!/usr/bin/env python3
"""Build and numerically inspect the official DRIVE/retinal Full Model once."""

from __future__ import annotations

import sys
from pathlib import Path

# Avoid writing import bytecode into the official repository.
sys.dont_write_bytecode = True

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = PROJECT_ROOT / "official_repo"
CONFIG_PATH = OFFICIAL_REPO / "configs" / "retinal.yaml"

sys.path.insert(0, str(OFFICIAL_REPO))

import tensorflow as tf  # noqa: E402
import train as official_train  # noqa: E402


def parameter_count(weights: list[tf.Variable]) -> int:
    return sum(int(np.prod(weight.shape)) for weight in weights)


def main() -> None:
    seed = 42
    official_train.set_seed(seed)

    cfg = official_train.load_config(str(CONFIG_PATH), [])
    model = official_train.build_model(cfg)

    total_params = int(model.count_params())
    trainable_params = parameter_count(model.trainable_weights)
    non_trainable_params = parameter_count(model.non_trainable_weights)

    input_size = int(cfg["model"]["input_size"])
    input_elements = input_size * input_size * 3
    synthetic_input = tf.reshape(
        tf.linspace(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(1.0, dtype=tf.float32),
            input_elements,
        ),
        (1, input_size, input_size, 3),
    )

    # The only concrete forward pass in this diagnostic.
    output = model(synthetic_input, training=False)
    output_np = output.numpy()

    weight_nan_count = 0
    weight_inf_count = 0
    weight_value_count = 0
    for weight in model.weights:
        values = weight.numpy()
        weight_value_count += int(values.size)
        weight_nan_count += int(np.isnan(values).sum())
        weight_inf_count += int(np.isinf(values).sum())

    output_nan_count = int(np.isnan(output_np).sum())
    output_inf_count = int(np.isinf(output_np).sum())

    m = cfg["model"]
    print("ENTRY_POINT=train.build_model")
    print(f"CONFIG={CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print("FULL_MODEL_FLAGS=" + str({
        "use_mrfse": m.get("use_mrfse", True),
        "use_sstm": m.get("use_sstm", True),
        "sstm_stages": m.get("sstm_stages", [True] * 5),
        "use_bfp": m.get("use_bfp", True),
        "bfp_routing": m.get("bfp_routing", "soft"),
    }))
    print(f"MODEL_NAME={model.name}")
    print(f"MODEL_INPUT_SHAPE={model.input_shape}")
    print(f"MODEL_INPUT_DTYPE={model.input.dtype.name}")
    print(f"MODEL_OUTPUT_SHAPE={model.output_shape}")
    print(f"MODEL_OUTPUT_DTYPE={model.output.dtype.name}")
    print(f"MODEL_TOP_LEVEL_LAYERS={len(model.layers)}")
    print(f"MRFSE_STAGES={sum(layer.name.startswith('mrfse_stage') for layer in model.layers)}")
    print(f"SSTM_STAGES={sum(layer.name.startswith('sstm_stage') for layer in model.layers)}")
    print(f"BFP_STAGES={sum(layer.name.startswith('bfp_stage') for layer in model.layers)}")
    print(f"TOTAL_PARAMS={total_params}")
    print(f"TRAINABLE_PARAMS={trainable_params}")
    print(f"NON_TRAINABLE_PARAMS={non_trainable_params}")
    print(f"PARAMETER_PARTITION_OK={total_params == trainable_params + non_trainable_params}")
    print(f"SYNTHETIC_INPUT_SEED={seed}")
    print(f"SYNTHETIC_INPUT_SHAPE={tuple(synthetic_input.shape)}")
    print(f"SYNTHETIC_INPUT_DTYPE={synthetic_input.dtype.name}")
    print(f"SYNTHETIC_INPUT_RANGE=[{float(tf.reduce_min(synthetic_input)):.9g}, {float(tf.reduce_max(synthetic_input)):.9g}]")
    print(f"OUTPUT_SHAPE={tuple(output.shape)}")
    print(f"OUTPUT_DTYPE={output.dtype.name}")
    print(f"OUTPUT_RANGE=[{float(np.min(output_np)):.9g}, {float(np.max(output_np)):.9g}]")
    print(f"OUTPUT_NAN_COUNT={output_nan_count}")
    print(f"OUTPUT_INF_COUNT={output_inf_count}")
    print(f"OUTPUT_ALL_FINITE={bool(np.isfinite(output_np).all())}")
    print(f"WEIGHT_TENSOR_COUNT={len(model.weights)}")
    print(f"WEIGHT_VALUE_COUNT={weight_value_count}")
    print(f"WEIGHT_NAN_COUNT={weight_nan_count}")
    print(f"WEIGHT_INF_COUNT={weight_inf_count}")
    print(f"WEIGHTS_ALL_FINITE={weight_nan_count == 0 and weight_inf_count == 0}")


if __name__ == "__main__":
    main()
