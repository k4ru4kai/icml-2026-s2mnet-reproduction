#!/usr/bin/env python3
"""Executable, build-only audit of Claim 1 architecture and parameter counts.

This diagnostic uses real TensorFlow constructors for the released S2M-Net
and the TransUNet approximation bundled with that release. The two pinned
upstream PyTorch baselines are represented by explicit per-tensor parameter
ledgers because PyTorch is not part of the reproduction environment.

No dataset, checkpoint, pretrained weight, optimizer, or training path is
used. The output deliberately contains no timestamps so identical source and
environment inputs produce byte-identical JSON.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence

# Never write import artifacts into the pinned reference checkout.
sys.dont_write_bytecode = True
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2mnet-matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = PROJECT_ROOT / "official_repo"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "audits" / "claim1_architecture_parameters.json"

AUDIT_DEVELOPMENT_BASE_PROJECT_HEAD = "45abc94e50f70d04c4bf11b5036480b9f58102e4"
EXPECTED_OFFICIAL_COMMIT = "3ec59668ab9b438ab9b170306d29b01e9270fd5a"
EXPECTED_SOURCE_SHA256 = {
    "docs/claim1_parameter_count_investigation.md":
        "3455e4378ca09e198e6f4c937a0eb7f831e1cc423736c3a959a6b86c140abc14",
    "official_repo/configs/default.yaml":
        "08ce1c7d52a79d3e584868685aefb6a970c0746f48d18d4e018fd40cda430cb6",
    "official_repo/s2mnet/models/baselines.py":
        "eaeff4dd074d6ccd399344907ac3335c296f6a4c64eba39c8b475a70dbbfaabf",
    "official_repo/s2mnet/models/blocks.py":
        "1cb072cb6abfc6e4ec7a201a15d1cd123f249bdf320b0cadd878493a8d6069d6",
    "official_repo/s2mnet/models/s2mnet.py":
        "fd2fcda4d3b54aaa6dbe6dc088d4fd132e24ac6a4e59666c42592cc82b565b4a",
}

S2MNET_COMMON_CONFIG: dict[str, Any] = {
    "num_classes": 1,
    "filters": (24, 32, 64, 80, 128),
    "use_mrfse": True,
    "mrfse_kernels": (3, 5, 7),
    "se_reduction": 16,
    "expand_ratio": 6,
    "use_sstm": True,
    "sstm_k": 32,
    "sstm_ssm_dim": 16,
    "sstm_stages": (True, True, True, True, True),
    "sstm_use_spectral": (True, True, True, True, True),
    "sstm_use_ssm": (False, False, True, True, True),
    "sstm_dropout": 0.1,
    "use_bfp": True,
    "bfp_routing": "soft",
    "dropout": 0.1,
    "l2_reg": 1e-4,
    "activation": "elu",
}

EXPECTED_COUNTS = {
    "s2mnet_352": {
        "total": 4_791_544,
        "trainable": 4_768_920,
        "non_trainable": 22_624,
    },
    "s2mnet_256": {
        "total": 4_766_008,
        "trainable": 4_743_384,
        "non_trainable": 22_624,
    },
    "bundled_transunet_352_tracked": {
        "total": 5_437_825,
        "trainable": 5_437_825,
        "non_trainable": 0,
    },
    "bundled_transunet_352_including_untracked_position": {
        "total": 5_933_441,
        "trainable": 5_933_441,
        "non_trainable": 0,
    },
    "official_transunet_224_9class": {
        "total": 105_277_081,
        "trainable": 105_277_081,
        "non_trainable": 0,
    },
    "official_swin_unet_224_9class": {
        "total": 27_168_900,
        "trainable": 27_168_900,
        "non_trainable": 0,
    },
}

TRANSUNET_COMMIT = "26de0c4d9a5145589ea249d169af7f7130823e03"
SWIN_UNET_COMMIT = "1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9"


@dataclass(frozen=True)
class LedgerEntry:
    """One actual trainable or non-trainable parameter tensor."""

    component: str
    name: str
    shape: tuple[int, ...]
    trainable: bool = True

    @property
    def count(self) -> int:
        return shape_numel(self.shape)

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["count"] = self.count
        return value


def shape_numel(shape: Sequence[Any]) -> int:
    """Return the product of a fully defined parameter shape."""
    dimensions: list[int] = []
    for dimension in shape:
        value = getattr(dimension, "value", dimension)
        if value is None:
            raise ValueError(f"Parameter shape is not fully defined: {shape}")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"Parameter shape has a negative dimension: {shape}")
        dimensions.append(integer)
    return int(math.prod(dimensions))


def sum_parameter_shapes(weights: Iterable[Any]) -> int:
    """Independently count parameters by multiplying every weight shape."""
    return sum(shape_numel(weight.shape) for weight in weights)


def parameter_partition(model: Any) -> dict[str, int]:
    """Count total/trainable/non-trainable parameters from a real model."""
    return {
        "total": int(model.count_params()),
        "trainable": sum_parameter_shapes(model.trainable_weights),
        "non_trainable": sum_parameter_shapes(model.non_trainable_weights),
    }


def ratio(numerator: int, denominator: int, decimal_places: int = 12) -> float:
    """Compute a stable, half-up rounded size ratio."""
    if denominator == 0:
        raise ZeroDivisionError("Parameter-count denominator must be non-zero")
    quantum = Decimal(1).scaleb(-decimal_places)
    value = (Decimal(numerator) / Decimal(denominator)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    return float(value)


def ratio_record(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": ratio(numerator, denominator),
    }


def json_compatible(value: Any) -> Any:
    """Recursively convert tuples and Paths to deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def deterministic_json_bytes(value: Any) -> bytes:
    """Serialize JSON canonically for stable reviewable result artifacts."""
    text = json.dumps(
        json_compatible(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def write_deterministic_json(path: Path, value: Any) -> bytes:
    """Write deterministic JSON and return the exact bytes written."""
    payload = deterministic_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return payload


def layer_shape(layer: Any) -> tuple[Any, ...]:
    shape = layer.output_shape
    if isinstance(shape, list):
        raise ValueError(f"Expected one output for {layer.name}, got {shape}")
    return tuple(shape)


def verify_encoder_architecture(
    model: Any,
    expected_channels: Sequence[int] = (24, 32, 64, 80, 128),
    expected_resolutions: Sequence[int] = (176, 88, 44, 22, 11),
) -> dict[str, Any]:
    """Inspect instantiated layers instead of trusting constructor arguments."""
    stage_pattern = re.compile(r"enc(\d+)_act")
    stage_layers: dict[int, Any] = {}
    model_layer_names = {layer.name for layer in model.layers}
    for layer in model.layers:
        match = stage_pattern.fullmatch(layer.name)
        if match:
            stage_layers[int(match.group(1))] = layer

    stages = []
    for stage_number in sorted(stage_layers):
        output_shape = layer_shape(stage_layers[stage_number])
        stages.append(
            {
                "stage": stage_number,
                "activation_layer": stage_layers[stage_number].name,
                "output_shape": list(output_shape),
                "resolution": [output_shape[1], output_shape[2]],
                "channels": output_shape[-1],
                "mrfse_layer_present": f"mrfse_stage{stage_number}" in model_layer_names,
                "sstm_layer_present": f"sstm_stage{stage_number}" in model_layer_names,
            }
        )

    observed_channels = [int(stage["channels"]) for stage in stages]
    observed_resolutions = [
        int(stage["resolution"][0]) for stage in stages
    ]
    observed_indices = [int(stage["stage"]) for stage in stages]
    expected_indices = list(range(1, len(expected_channels) + 1))
    checks = {
        "stage_indices": observed_indices == expected_indices,
        "stage_count": len(stages) == len(expected_channels),
        "channels": observed_channels == list(expected_channels),
        "resolutions": observed_resolutions == list(expected_resolutions),
        "mrfse_at_every_stage": all(
            bool(stage["mrfse_layer_present"]) for stage in stages
        ),
        "sstm_at_every_stage": all(
            bool(stage["sstm_layer_present"]) for stage in stages
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "expected": {
            "stage_count": len(expected_channels),
            "stage_indices": expected_indices,
            "channels": list(expected_channels),
            "resolutions": list(expected_resolutions),
        },
        "observed": {
            "stage_count": len(stages),
            "stage_indices": observed_indices,
            "channels": observed_channels,
            "resolutions": observed_resolutions,
            "stages": stages,
        },
        "checks": checks,
    }


def weight_shape_map(model: Any) -> dict[str, dict[str, Any]]:
    trainable_names = {weight.name for weight in model.trainable_weights}
    records: dict[str, dict[str, Any]] = {}
    for weight in model.weights:
        if weight.name in records:
            raise ValueError(f"Duplicate model weight name: {weight.name}")
        records[weight.name] = {
            "shape": tuple(int(value) for value in weight.shape),
            "count": shape_numel(weight.shape),
            "trainable": weight.name in trainable_names,
        }
    return records


def compare_resolution_dependent_variables(
    model_256: Any,
    model_352: Any,
) -> dict[str, Any]:
    """Locate every variable and top-level layer whose count changes."""
    weights_256 = weight_shape_map(model_256)
    weights_352 = weight_shape_map(model_352)
    changes = []
    for name in sorted(set(weights_256) | set(weights_352)):
        left = weights_256.get(name)
        right = weights_352.get(name)
        if left == right:
            continue
        count_256 = 0 if left is None else int(left["count"])
        count_352 = 0 if right is None else int(right["count"])
        changes.append(
            {
                "variable": name,
                "layer": name.split("/", 1)[0],
                "shape_256": None if left is None else list(left["shape"]),
                "shape_352": None if right is None else list(right["shape"]),
                "parameters_256": count_256,
                "parameters_352": count_352,
                "delta_352_minus_256": count_352 - count_256,
                "trainable_256": None if left is None else bool(left["trainable"]),
                "trainable_352": None if right is None else bool(right["trainable"]),
            }
        )

    layer_counts_256 = {
        layer.name: int(layer.count_params()) for layer in model_256.layers
    }
    layer_counts_352 = {
        layer.name: int(layer.count_params()) for layer in model_352.layers
    }
    layer_changes = []
    for name in sorted(set(layer_counts_256) | set(layer_counts_352)):
        count_256 = layer_counts_256.get(name, 0)
        count_352 = layer_counts_352.get(name, 0)
        if count_256 != count_352:
            layer_changes.append(
                {
                    "layer": name,
                    "parameters_256": count_256,
                    "parameters_352": count_352,
                    "delta_352_minus_256": count_352 - count_256,
                }
            )

    return {
        "total_delta_352_minus_256": (
            int(model_352.count_params()) - int(model_256.count_params())
        ),
        "delta_accounted_for_by_changed_variables": sum(
            int(change["delta_352_minus_256"]) for change in changes
        ),
        "changed_variables": changes,
        "changed_top_level_layers": layer_changes,
    }


def add_tensor(
    ledger: list[LedgerEntry],
    component: str,
    name: str,
    shape: Sequence[int],
    trainable: bool = True,
) -> None:
    ledger.append(
        LedgerEntry(
            component=component,
            name=name,
            shape=tuple(int(value) for value in shape),
            trainable=trainable,
        )
    )


def add_norm(
    ledger: list[LedgerEntry],
    component: str,
    name: str,
    features: int,
) -> None:
    add_tensor(ledger, component, f"{name}.weight", (features,))
    add_tensor(ledger, component, f"{name}.bias", (features,))


def add_linear(
    ledger: list[LedgerEntry],
    component: str,
    name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
) -> None:
    add_tensor(ledger, component, f"{name}.weight", (out_features, in_features))
    if bias:
        add_tensor(ledger, component, f"{name}.bias", (out_features,))


def add_conv2d(
    ledger: list[LedgerEntry],
    component: str,
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = True,
) -> None:
    add_tensor(
        ledger,
        component,
        f"{name}.weight",
        (out_channels, in_channels, kernel_size, kernel_size),
    )
    if bias:
        add_tensor(ledger, component, f"{name}.bias", (out_channels,))


def ledger_summary(entries: Sequence[LedgerEntry]) -> dict[str, Any]:
    components: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "trainable": 0, "non_trainable": 0, "tensors": 0}
    )
    trainable = 0
    non_trainable = 0
    for entry in entries:
        count = entry.count
        component = components[entry.component]
        component["total"] += count
        component["tensors"] += 1
        if entry.trainable:
            trainable += count
            component["trainable"] += count
        else:
            non_trainable += count
            component["non_trainable"] += count
    serialized_entries = [entry.to_json() for entry in entries]
    return {
        "parameters": {
            "total": trainable + non_trainable,
            "trainable": trainable,
            "non_trainable": non_trainable,
        },
        "parameter_tensor_count": len(entries),
        "components": dict(sorted(components.items())),
        "ledger_sha256": hashlib.sha256(
            deterministic_json_bytes(serialized_entries)
        ).hexdigest(),
        "entries": serialized_entries,
    }


def official_transunet_ledger(
    image_size: int = 224,
    num_classes: int = 9,
) -> list[LedgerEntry]:
    """Reproduce official R50-ViT-B_16 parameters from the pinned source."""
    if image_size <= 0 or image_size % 16:
        raise ValueError("Official TransUNet image_size must be positive and divisible by 16")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    ledger: list[LedgerEntry] = []
    resnet_component = "hybrid_resnet_v2"
    add_conv2d(ledger, resnet_component, "resnet.root.conv", 3, 64, 7, bias=False)
    add_norm(ledger, resnet_component, "resnet.root.gn", 64)

    stage_specs = (
        (3, 64, 256, 64),
        (4, 256, 512, 128),
        (9, 512, 1024, 256),
    )
    for stage, (units, first_input, output, middle) in enumerate(stage_specs, 1):
        for unit in range(1, units + 1):
            prefix = f"resnet.block{stage}.unit{unit}"
            input_channels = first_input if unit == 1 else output
            add_conv2d(
                ledger, resnet_component, f"{prefix}.conv1",
                input_channels, middle, 1, bias=False,
            )
            add_norm(ledger, resnet_component, f"{prefix}.gn1", middle)
            add_conv2d(
                ledger, resnet_component, f"{prefix}.conv2",
                middle, middle, 3, bias=False,
            )
            add_norm(ledger, resnet_component, f"{prefix}.gn2", middle)
            add_conv2d(
                ledger, resnet_component, f"{prefix}.conv3",
                middle, output, 1, bias=False,
            )
            add_norm(ledger, resnet_component, f"{prefix}.gn3", output)
            if unit == 1:
                add_conv2d(
                    ledger, resnet_component, f"{prefix}.downsample",
                    input_channels, output, 1, bias=False,
                )
                add_norm(ledger, resnet_component, f"{prefix}.gn_proj", output)

    add_conv2d(
        ledger,
        "hybrid_patch_embedding",
        "transformer.embeddings.patch_embeddings",
        1024,
        768,
        1,
        bias=True,
    )
    token_side = image_size // 16
    add_tensor(
        ledger,
        "position_embedding",
        "transformer.embeddings.position_embeddings",
        (1, token_side * token_side, 768),
    )

    for block in range(12):
        prefix = f"transformer.encoder.block{block}"
        component = "transformer_blocks"
        add_norm(ledger, component, f"{prefix}.attention_norm", 768)
        for projection in ("query", "key", "value", "out"):
            add_linear(
                ledger, component, f"{prefix}.attention.{projection}", 768, 768
            )
        add_norm(ledger, component, f"{prefix}.ffn_norm", 768)
        add_linear(ledger, component, f"{prefix}.ffn.fc1", 768, 3072)
        add_linear(ledger, component, f"{prefix}.ffn.fc2", 3072, 768)

    add_norm(
        ledger,
        "final_transformer_norm",
        "transformer.encoder.encoder_norm",
        768,
    )

    decoder_component = "cup_decoder"
    add_conv2d(
        ledger, decoder_component, "decoder.conv_more.conv",
        768, 512, 3, bias=False,
    )
    add_norm(ledger, decoder_component, "decoder.conv_more.bn", 512)
    decoder_specs = (
        (512, 512, 256),
        (256, 256, 128),
        (128, 64, 64),
        (64, 0, 16),
    )
    for block, (input_channels, skip_channels, output_channels) in enumerate(
        decoder_specs
    ):
        prefix = f"decoder.block{block}"
        add_conv2d(
            ledger, decoder_component, f"{prefix}.conv1",
            input_channels + skip_channels, output_channels, 3, bias=False,
        )
        add_norm(ledger, decoder_component, f"{prefix}.conv1.bn", output_channels)
        add_conv2d(
            ledger, decoder_component, f"{prefix}.conv2",
            output_channels, output_channels, 3, bias=False,
        )
        add_norm(ledger, decoder_component, f"{prefix}.conv2.bn", output_channels)

    add_conv2d(
        ledger,
        "segmentation_head",
        "segmentation_head.conv",
        16,
        num_classes,
        3,
        bias=True,
    )
    return ledger


def add_swin_block(
    ledger: list[LedgerEntry],
    component: str,
    prefix: str,
    dimension: int,
    heads: int,
    resolution: int,
    window_size: int = 7,
    mlp_ratio: int = 4,
) -> None:
    effective_window = min(window_size, resolution)
    add_norm(ledger, component, f"{prefix}.norm1", dimension)
    add_tensor(
        ledger,
        component,
        f"{prefix}.attn.relative_position_bias_table",
        ((2 * effective_window - 1) ** 2, heads),
    )
    add_linear(
        ledger, component, f"{prefix}.attn.qkv",
        dimension, 3 * dimension, bias=True,
    )
    add_linear(
        ledger, component, f"{prefix}.attn.proj",
        dimension, dimension, bias=True,
    )
    add_norm(ledger, component, f"{prefix}.norm2", dimension)
    hidden = dimension * mlp_ratio
    add_linear(ledger, component, f"{prefix}.mlp.fc1", dimension, hidden)
    add_linear(ledger, component, f"{prefix}.mlp.fc2", hidden, dimension)


def add_patch_expand(
    ledger: list[LedgerEntry],
    component: str,
    prefix: str,
    dimension: int,
) -> None:
    add_linear(
        ledger, component, f"{prefix}.expand",
        dimension, 2 * dimension, bias=False,
    )
    add_norm(ledger, component, f"{prefix}.norm", dimension // 2)


def official_swin_unet_ledger(
    image_size: int = 224,
    num_classes: int = 9,
) -> list[LedgerEntry]:
    """Reproduce the pinned official Tiny/lite Swin-Unet parameter ledger."""
    if image_size <= 0 or image_size % 32:
        raise ValueError("Official Swin-Unet image_size must be divisible by 32")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    ledger: list[LedgerEntry] = []
    patch_size = 4
    embed_dim = 96
    depths = (2, 2, 2, 2)
    heads = (3, 6, 12, 24)
    patch_resolution = image_size // patch_size

    add_conv2d(
        ledger,
        "patch_embedding",
        "patch_embed.proj",
        3,
        embed_dim,
        patch_size,
        bias=True,
    )
    add_norm(ledger, "patch_embedding", "patch_embed.norm", embed_dim)

    encoder_component = "encoder_bottleneck_stages"
    for stage in range(4):
        dimension = embed_dim * (2 ** stage)
        resolution = patch_resolution // (2 ** stage)
        for block in range(depths[stage]):
            add_swin_block(
                ledger,
                encoder_component,
                f"layers.{stage}.blocks.{block}",
                dimension,
                heads[stage],
                resolution,
            )
        if stage < 3:
            prefix = f"layers.{stage}.downsample"
            add_norm(ledger, encoder_component, f"{prefix}.norm", 4 * dimension)
            add_linear(
                ledger,
                encoder_component,
                f"{prefix}.reduction",
                4 * dimension,
                2 * dimension,
                bias=False,
            )

    add_norm(ledger, "final_encoder_norm", "norm", 768)

    decoder_component = "decoder_layers"
    add_patch_expand(ledger, decoder_component, "layers_up.0", 768)
    decoder_specs = (
        (1, 384, 12, patch_resolution // 4, True),
        (2, 192, 6, patch_resolution // 2, True),
        (3, 96, 3, patch_resolution, False),
    )
    for decoder_stage, dimension, head_count, resolution, upsample in decoder_specs:
        add_linear(
            ledger,
            "skip_concat_projections",
            f"concat_back_dim.{decoder_stage}",
            2 * dimension,
            dimension,
            bias=True,
        )
        for block in range(2):
            add_swin_block(
                ledger,
                decoder_component,
                f"layers_up.{decoder_stage}.blocks.{block}",
                dimension,
                head_count,
                resolution,
            )
        if upsample:
            add_patch_expand(
                ledger,
                decoder_component,
                f"layers_up.{decoder_stage}.upsample",
                dimension,
            )

    add_norm(ledger, "final_decoder_norm", "norm_up", embed_dim)
    add_linear(
        ledger,
        "final_patch_expand",
        "up.expand",
        embed_dim,
        16 * embed_dim,
        bias=False,
    )
    add_norm(ledger, "final_patch_expand", "up.norm", embed_dim)
    add_conv2d(
        ledger,
        "segmentation_head",
        "output",
        embed_dim,
        num_classes,
        1,
        bias=False,
    )
    return ledger


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verification(
    name: str,
    passed: bool,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "expected": json_compatible(expected),
        "observed": json_compatible(observed),
    }


def build_audit_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    observed_official_commit = git_output(OFFICIAL_REPO, "rev-parse", "HEAD")
    official_status = git_output(
        OFFICIAL_REPO, "status", "--porcelain=v1", "--untracked-files=all"
    )
    checks.append(
        verification(
            "official_repository_commit",
            observed_official_commit == EXPECTED_OFFICIAL_COMMIT,
            EXPECTED_OFFICIAL_COMMIT,
            observed_official_commit,
        )
    )
    checks.append(
        verification(
            "official_repository_clean",
            official_status == "",
            "",
            official_status,
        )
    )

    source_hashes: dict[str, dict[str, str]] = {}
    for relative_path, expected_hash in sorted(EXPECTED_SOURCE_SHA256.items()):
        observed_hash = sha256_file(PROJECT_ROOT / relative_path)
        source_hashes[relative_path] = {
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "status": "pass" if observed_hash == expected_hash else "fail",
        }
        checks.append(
            verification(
                f"source_sha256:{relative_path}",
                observed_hash == expected_hash,
                expected_hash,
                observed_hash,
            )
        )

    sys.path.insert(0, str(OFFICIAL_REPO))
    import tensorflow as tf  # noqa: PLC0415
    from s2mnet.models.baselines import TransUNet  # noqa: PLC0415
    from s2mnet.models.s2mnet import S2MNet  # noqa: PLC0415

    s2mnet_256 = S2MNet(input_size=256, **S2MNET_COMMON_CONFIG)
    s2mnet_352 = S2MNet(input_size=352, **S2MNET_COMMON_CONFIG)
    count_256 = parameter_partition(s2mnet_256)
    count_352 = parameter_partition(s2mnet_352)
    independent_256 = {
        "all_weights": sum_parameter_shapes(s2mnet_256.weights),
        "trainable_weights": sum_parameter_shapes(s2mnet_256.trainable_weights),
        "non_trainable_weights": sum_parameter_shapes(
            s2mnet_256.non_trainable_weights
        ),
    }
    independent_352 = {
        "all_weights": sum_parameter_shapes(s2mnet_352.weights),
        "trainable_weights": sum_parameter_shapes(s2mnet_352.trainable_weights),
        "non_trainable_weights": sum_parameter_shapes(
            s2mnet_352.non_trainable_weights
        ),
    }
    checks.extend(
        [
            verification(
                "s2mnet_256_expected_parameters",
                count_256 == EXPECTED_COUNTS["s2mnet_256"],
                EXPECTED_COUNTS["s2mnet_256"],
                count_256,
            ),
            verification(
                "s2mnet_352_expected_parameters",
                count_352 == EXPECTED_COUNTS["s2mnet_352"],
                EXPECTED_COUNTS["s2mnet_352"],
                count_352,
            ),
            verification(
                "s2mnet_256_constructor_vs_shape_sum",
                (
                    count_256["total"] == independent_256["all_weights"]
                    and count_256["trainable"]
                    == independent_256["trainable_weights"]
                    and count_256["non_trainable"]
                    == independent_256["non_trainable_weights"]
                ),
                count_256,
                independent_256,
            ),
            verification(
                "s2mnet_352_constructor_vs_shape_sum",
                (
                    count_352["total"] == independent_352["all_weights"]
                    and count_352["trainable"]
                    == independent_352["trainable_weights"]
                    and count_352["non_trainable"]
                    == independent_352["non_trainable_weights"]
                ),
                count_352,
                independent_352,
            ),
        ]
    )

    architecture = verify_encoder_architecture(s2mnet_352)
    checks.append(
        verification(
            "s2mnet_encoder_architecture",
            architecture["status"] == "pass",
            architecture["expected"],
            architecture["observed"],
        )
    )

    resolution_difference = compare_resolution_dependent_variables(
        s2mnet_256, s2mnet_352
    )
    expected_changed_variables = [
        {
            "variable": "sstm_stage4/freq_weights:0",
            "shape_256": [16, 16, 80],
            "shape_352": [22, 22, 80],
            "delta_352_minus_256": 18_240,
        },
        {
            "variable": "sstm_stage5/freq_weights:0",
            "shape_256": [8, 8, 128],
            "shape_352": [11, 11, 128],
            "delta_352_minus_256": 7_296,
        },
    ]
    observed_changed_variables = [
        {
            key: change[key]
            for key in (
                "variable",
                "shape_256",
                "shape_352",
                "delta_352_minus_256",
            )
        }
        for change in resolution_difference["changed_variables"]
    ]
    expected_changed_layers = [
        {"layer": "sstm_stage4", "delta_352_minus_256": 18_240},
        {"layer": "sstm_stage5", "delta_352_minus_256": 7_296},
    ]
    observed_changed_layers = [
        {
            "layer": change["layer"],
            "delta_352_minus_256": change["delta_352_minus_256"],
        }
        for change in resolution_difference["changed_top_level_layers"]
    ]
    checks.extend(
        [
            verification(
                "resolution_dependent_total_delta",
                resolution_difference["total_delta_352_minus_256"] == 25_536,
                25_536,
                resolution_difference["total_delta_352_minus_256"],
            ),
            verification(
                "resolution_dependent_variables",
                observed_changed_variables == expected_changed_variables,
                expected_changed_variables,
                observed_changed_variables,
            ),
            verification(
                "resolution_dependent_layers",
                observed_changed_layers == expected_changed_layers,
                expected_changed_layers,
                observed_changed_layers,
            ),
            verification(
                "resolution_delta_fully_accounted_for",
                (
                    resolution_difference[
                        "delta_accounted_for_by_changed_variables"
                    ]
                    == resolution_difference["total_delta_352_minus_256"]
                ),
                resolution_difference["total_delta_352_minus_256"],
                resolution_difference[
                    "delta_accounted_for_by_changed_variables"
                ],
            ),
        ]
    )

    bundled_transunet = TransUNet(input_size=352, num_classes=1)
    bundled_tracked = parameter_partition(bundled_transunet)
    bundled_independent = {
        "all_weights": sum_parameter_shapes(bundled_transunet.weights),
        "trainable_weights": sum_parameter_shapes(
            bundled_transunet.trainable_weights
        ),
        "non_trainable_weights": sum_parameter_shapes(
            bundled_transunet.non_trainable_weights
        ),
    }
    transformer_input_shape = tuple(
        bundled_transunet.get_layer("transformer_0").input_shape
    )
    position_shape = (
        1,
        int(transformer_input_shape[-2]),
        int(transformer_input_shape[-1]),
    )
    position_count = shape_numel(position_shape)
    tracked_position_variables = [
        weight.name
        for weight in bundled_transunet.weights
        if "pos_encoding" in weight.name
    ]
    bundled_inclusive = {
        "total": bundled_tracked["total"] + position_count,
        "trainable": bundled_tracked["trainable"] + position_count,
        "non_trainable": bundled_tracked["non_trainable"],
    }
    checks.extend(
        [
            verification(
                "bundled_transunet_tracked_parameters",
                (
                    bundled_tracked
                    == EXPECTED_COUNTS["bundled_transunet_352_tracked"]
                ),
                EXPECTED_COUNTS["bundled_transunet_352_tracked"],
                bundled_tracked,
            ),
            verification(
                "bundled_transunet_constructor_vs_shape_sum",
                (
                    bundled_tracked["total"]
                    == bundled_independent["all_weights"]
                    and bundled_tracked["trainable"]
                    == bundled_independent["trainable_weights"]
                    and bundled_tracked["non_trainable"]
                    == bundled_independent["non_trainable_weights"]
                ),
                bundled_tracked,
                bundled_independent,
            ),
            verification(
                "bundled_transunet_position_embedding_is_untracked",
                not tracked_position_variables,
                [],
                tracked_position_variables,
            ),
            verification(
                "bundled_transunet_inclusive_parameter_ledger",
                (
                    bundled_inclusive
                    == EXPECTED_COUNTS[
                        "bundled_transunet_352_including_untracked_position"
                    ]
                ),
                EXPECTED_COUNTS[
                    "bundled_transunet_352_including_untracked_position"
                ],
                bundled_inclusive,
            ),
        ]
    )

    transunet_224_1 = ledger_summary(
        official_transunet_ledger(image_size=224, num_classes=1)
    )
    transunet_224_2 = ledger_summary(
        official_transunet_ledger(image_size=224, num_classes=2)
    )
    transunet_224_9 = ledger_summary(
        official_transunet_ledger(image_size=224, num_classes=9)
    )
    transunet_352_1 = ledger_summary(
        official_transunet_ledger(image_size=352, num_classes=1)
    )
    transunet_352_9 = ledger_summary(
        official_transunet_ledger(image_size=352, num_classes=9)
    )
    checks.append(
        verification(
            "official_transunet_parameter_ledger",
            (
                transunet_224_9["parameters"]
                == EXPECTED_COUNTS["official_transunet_224_9class"]
            ),
            EXPECTED_COUNTS["official_transunet_224_9class"],
            transunet_224_9["parameters"],
        )
    )
    expected_transunet_components = {
        "hybrid_resnet_v2": 11_894_848,
        "hybrid_patch_embedding": 787_200,
        "position_embedding": 150_528,
        "transformer_blocks": 85_054_464,
        "final_transformer_norm": 1_536,
        "cup_decoder": 7_387_200,
        "segmentation_head": 1_305,
    }
    observed_transunet_components = {
        key: value["total"]
        for key, value in transunet_224_9["components"].items()
    }
    checks.append(
        verification(
            "official_transunet_component_ledger",
            observed_transunet_components == expected_transunet_components,
            expected_transunet_components,
            observed_transunet_components,
        )
    )

    swin_224_1 = ledger_summary(
        official_swin_unet_ledger(image_size=224, num_classes=1)
    )
    swin_224_9 = ledger_summary(
        official_swin_unet_ledger(image_size=224, num_classes=9)
    )
    swin_352_9 = ledger_summary(
        official_swin_unet_ledger(image_size=352, num_classes=9)
    )
    checks.extend(
        [
            verification(
                "official_swin_unet_parameter_ledger",
                (
                    swin_224_9["parameters"]
                    == EXPECTED_COUNTS["official_swin_unet_224_9class"]
                ),
                EXPECTED_COUNTS["official_swin_unet_224_9class"],
                swin_224_9["parameters"],
            ),
            verification(
                "official_swin_unet_parameter_tensor_count",
                swin_224_9["parameter_tensor_count"] == 218,
                218,
                swin_224_9["parameter_tensor_count"],
            ),
        ]
    )
    expected_swin_components = {
        "patch_embedding": 4_896,
        "encoder_bottleneck_stages": 20_406_954,
        "final_encoder_norm": 1_536,
        "decoder_layers": 6_219_066,
        "skip_concat_projections": 387_744,
        "final_decoder_norm": 192,
        "final_patch_expand": 147_648,
        "segmentation_head": 864,
    }
    observed_swin_components = {
        key: value["total"] for key, value in swin_224_9["components"].items()
    }
    checks.append(
        verification(
            "official_swin_unet_component_ledger",
            observed_swin_components == expected_swin_components,
            expected_swin_components,
            observed_swin_components,
        )
    )

    s2m_reference = count_352["total"]
    compared_counts = {
        "s2mnet_256_runtime_constructor": count_256["total"],
        "bundled_transunet_352_keras_tracked_runtime_constructor":
            bundled_tracked["total"],
        "bundled_transunet_352_constructor_plus_untracked_ledger":
            bundled_inclusive["total"],
        "transunet_readme_reported_60_0m": 60_000_000,
        "official_transunet_224_1class_ledger":
            transunet_224_1["parameters"]["total"],
        "official_transunet_224_2class_ledger":
            transunet_224_2["parameters"]["total"],
        "official_transunet_224_9class_ledger":
            transunet_224_9["parameters"]["total"],
        "official_transunet_352_1class_sensitivity_ledger":
            transunet_352_1["parameters"]["total"],
        "official_transunet_352_9class_sensitivity_ledger":
            transunet_352_9["parameters"]["total"],
        "swin_unet_readme_reported_27_0m": 27_000_000,
        "official_swin_unet_224_1class_sensitivity_ledger":
            swin_224_1["parameters"]["total"],
        "official_swin_unet_224_9class_ledger":
            swin_224_9["parameters"]["total"],
        "official_swin_unet_352_9class_sensitivity_ledger":
            swin_352_9["parameters"]["total"],
    }
    ratios = {
        key: ratio_record(value, s2m_reference)
        for key, value in sorted(compared_counts.items())
    }
    ratios["readme_display_60_0m_divided_by_4_7m"] = ratio_record(
        60_000_000, 4_700_000
    )
    ratios["readme_display_27_0m_divided_by_4_7m"] = ratio_record(
        27_000_000, 4_700_000
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "audit": {
            "name": "Claim 1 architecture and parameter-count audit",
            "scope": "build-only; no dataset, checkpoint, pretrained weights, or training",
            "output_is_deterministic": True,
            "audit_development_base_project_head":
                AUDIT_DEVELOPMENT_BASE_PROJECT_HEAD,
            "audit_script_sha256": sha256_file(Path(__file__)),
            "official_repository_commit": observed_official_commit,
            "official_repository_expected_commit": EXPECTED_OFFICIAL_COMMIT,
            "frameworks": {
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "pytorch_available": importlib.util.find_spec("torch") is not None,
            },
            "source_sha256": source_hashes,
        },
        "evidence_classes": {
            "runtime_model_constructor": (
                "A real framework model constructor was executed and its tracked "
                "variables were counted."
            ),
            "transparent_parameter_ledger": (
                "Every parameter tensor shape was derived from a pinned source "
                "configuration and summed; the upstream constructor was not executed."
            ),
            "runtime_constructor_plus_ledger": (
                "A runtime model count was supplemented by a separately identified "
                "tensor that the constructor creates but Keras does not track."
            ),
            "reported_external_value": (
                "A rounded value stated by the paper or released README without an "
                "executable matching configuration."
            ),
        },
        "models": {
            "s2mnet_352": {
                "evidence_class": "runtime_model_constructor",
                "constructor_executed": True,
                "framework": f"TensorFlow {tf.__version__}",
                "constructor": "official_repo/s2mnet/models/s2mnet.py:S2MNet",
                "source_repository": (
                    "https://github.com/sanaullah-ashfat/"
                    "S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation"
                ),
                "pinned_commit": EXPECTED_OFFICIAL_COMMIT,
                "configuration": {
                    "input_shape": [352, 352, 3],
                    **json_compatible(S2MNET_COMMON_CONFIG),
                },
                "counting_method": {
                    "primary": "tf.keras.Model.count_params()",
                    "independent": "sum(product(variable.shape))",
                },
                "expected_parameters": EXPECTED_COUNTS["s2mnet_352"],
                "observed_parameters": count_352,
                "independent_shape_sum": independent_352,
                "parameter_tensors": {
                    "all": len(s2mnet_352.weights),
                    "trainable": len(s2mnet_352.trainable_weights),
                    "non_trainable": len(s2mnet_352.non_trainable_weights),
                },
                "status": (
                    "pass"
                    if count_352 == EXPECTED_COUNTS["s2mnet_352"]
                    else "fail"
                ),
            },
            "s2mnet_256": {
                "evidence_class": "runtime_model_constructor",
                "constructor_executed": True,
                "framework": f"TensorFlow {tf.__version__}",
                "constructor": "official_repo/s2mnet/models/s2mnet.py:S2MNet",
                "source_repository": (
                    "https://github.com/sanaullah-ashfat/"
                    "S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation"
                ),
                "pinned_commit": EXPECTED_OFFICIAL_COMMIT,
                "configuration": {
                    "input_shape": [256, 256, 3],
                    **json_compatible(S2MNET_COMMON_CONFIG),
                },
                "counting_method": {
                    "primary": "tf.keras.Model.count_params()",
                    "independent": "sum(product(variable.shape))",
                },
                "expected_parameters": EXPECTED_COUNTS["s2mnet_256"],
                "observed_parameters": count_256,
                "independent_shape_sum": independent_256,
                "parameter_tensors": {
                    "all": len(s2mnet_256.weights),
                    "trainable": len(s2mnet_256.trainable_weights),
                    "non_trainable": len(s2mnet_256.non_trainable_weights),
                },
                "status": (
                    "pass"
                    if count_256 == EXPECTED_COUNTS["s2mnet_256"]
                    else "fail"
                ),
            },
            "bundled_transunet_352_keras_tracked": {
                "evidence_class": "runtime_model_constructor",
                "constructor_executed": True,
                "framework": f"TensorFlow {tf.__version__}",
                "constructor": (
                    "official_repo/s2mnet/models/baselines.py:TransUNet"
                ),
                "source_repository": (
                    "https://github.com/sanaullah-ashfat/"
                    "S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation"
                ),
                "pinned_commit": EXPECTED_OFFICIAL_COMMIT,
                "configuration": {
                    "input_shape": [352, 352, 3],
                    "num_classes": 1,
                    "cnn_channels": [64, 128, 256],
                    "transformer_hidden_size": 256,
                    "transformer_layers": 2,
                    "attention_heads": 8,
                },
                "counting_method": {
                    "primary": "tf.keras.Model.count_params()",
                    "independent": "sum(product(tracked_variable.shape))",
                    "scope": "Keras-tracked variables only",
                },
                "expected_parameters": EXPECTED_COUNTS[
                    "bundled_transunet_352_tracked"
                ],
                "observed_parameters": bundled_tracked,
                "independent_shape_sum": bundled_independent,
                "parameter_tensors": {
                    "all": len(bundled_transunet.weights),
                    "trainable": len(bundled_transunet.trainable_weights),
                    "non_trainable": len(
                        bundled_transunet.non_trainable_weights
                    ),
                },
                "status": (
                    "pass"
                    if bundled_tracked
                    == EXPECTED_COUNTS["bundled_transunet_352_tracked"]
                    else "fail"
                ),
            },
            "bundled_transunet_352_including_untracked_position": {
                "evidence_class": "runtime_constructor_plus_ledger",
                "constructor_executed": True,
                "framework": f"TensorFlow {tf.__version__}",
                "constructor": (
                    "official_repo/s2mnet/models/baselines.py:TransUNet"
                ),
                "configuration": {
                    "input_shape": [352, 352, 3],
                    "num_classes": 1,
                },
                "counting_method": (
                    "Keras-tracked runtime count plus product of the untracked "
                    "raw tf.Variable position-embedding shape"
                ),
                "keras_tracked_parameters": bundled_tracked,
                "untracked_position_embedding": {
                    "name_in_source": "pos_encoding",
                    "shape": list(position_shape),
                    "parameters": position_count,
                    "declared_trainable": True,
                    "present_in_model_weights": bool(tracked_position_variables),
                },
                "expected_parameters": EXPECTED_COUNTS[
                    "bundled_transunet_352_including_untracked_position"
                ],
                "observed_parameters": bundled_inclusive,
                "status": (
                    "pass"
                    if bundled_inclusive
                    == EXPECTED_COUNTS[
                        "bundled_transunet_352_including_untracked_position"
                    ]
                    else "fail"
                ),
            },
            "official_transunet_r50_vit_b16_224_9class": {
                "evidence_class": "transparent_parameter_ledger",
                "constructor_executed": False,
                "constructor_execution_limitation": (
                    "PyTorch is unavailable in the project environment."
                ),
                "source_repository": "https://github.com/Beckschen/TransUNet",
                "pinned_commit": TRANSUNET_COMMIT,
                "source_files": [
                    "networks/vit_seg_configs.py",
                    "networks/vit_seg_modeling.py",
                    "networks/vit_seg_modeling_resnet_skip.py",
                    "train.py",
                ],
                "configuration": {
                    "variant": "R50-ViT-B_16",
                    "input_shape": [224, 224, 3],
                    "num_classes": 9,
                    "resnet_width": 64,
                    "resnet_block_units": [3, 4, 9],
                    "vit_hidden_size": 768,
                    "vit_layers": 12,
                    "vit_heads": 12,
                    "vit_mlp_dimension": 3072,
                    "decoder_channels": [256, 128, 64, 16],
                    "skip_channels": [512, 256, 64, 0],
                    "n_skip": 3,
                },
                "counting_method": (
                    "Explicit sum of every PyTorch parameter tensor shape; "
                    "buffers excluded as model.parameters() excludes them."
                ),
                "expected_parameters": EXPECTED_COUNTS[
                    "official_transunet_224_9class"
                ],
                "observed_parameters": transunet_224_9["parameters"],
                "parameter_tensor_count": transunet_224_9[
                    "parameter_tensor_count"
                ],
                "component_ledger": transunet_224_9["components"],
                "ledger_sha256": transunet_224_9["ledger_sha256"],
                "parameter_ledger": transunet_224_9["entries"],
                "sensitivity_counts": {
                    "224_1class": transunet_224_1["parameters"],
                    "224_2class": transunet_224_2["parameters"],
                    "352_1class": transunet_352_1["parameters"],
                    "352_9class": transunet_352_9["parameters"],
                },
                "status": (
                    "pass"
                    if transunet_224_9["parameters"]
                    == EXPECTED_COUNTS["official_transunet_224_9class"]
                    else "fail"
                ),
            },
            "official_swin_unet_tiny_lite_224_9class": {
                "evidence_class": "transparent_parameter_ledger",
                "constructor_executed": False,
                "constructor_execution_limitation": (
                    "PyTorch is unavailable in the project environment."
                ),
                "source_repository": (
                    "https://github.com/HuCaoFighting/Swin-Unet"
                ),
                "pinned_commit": SWIN_UNET_COMMIT,
                "source_files": [
                    "configs/swin_tiny_patch4_window7_224_lite.yaml",
                    "config.py",
                    "train.sh",
                    "train.py",
                    "networks/vision_transformer.py",
                    (
                        "networks/"
                        "swin_transformer_unet_skip_expand_decoder_sys.py"
                    ),
                ],
                "configuration": {
                    "variant": "swin_tiny_patch4_window7_224_lite",
                    "input_shape": [224, 224, 3],
                    "num_classes": 9,
                    "patch_size": 4,
                    "embed_dimension": 96,
                    "encoder_depths": [2, 2, 2, 2],
                    "configured_decoder_depths_unused": [2, 2, 2, 1],
                    "effective_decoder_transformer_depths": [2, 2, 2],
                    "attention_heads": [3, 6, 12, 24],
                    "window_size": 7,
                    "mlp_ratio": 4,
                    "qkv_bias": True,
                    "absolute_position_embedding": False,
                    "patch_normalization": True,
                    "segmentation_head_bias": False,
                },
                "counting_method": (
                    "Explicit sum of every PyTorch parameter tensor shape; "
                    "registered attention masks and indices are buffers and excluded."
                ),
                "expected_parameters": EXPECTED_COUNTS[
                    "official_swin_unet_224_9class"
                ],
                "observed_parameters": swin_224_9["parameters"],
                "parameter_tensor_count": swin_224_9["parameter_tensor_count"],
                "component_ledger": swin_224_9["components"],
                "ledger_sha256": swin_224_9["ledger_sha256"],
                "parameter_ledger": swin_224_9["entries"],
                "sensitivity_counts": {
                    "224_1class": swin_224_1["parameters"],
                    "352_9class": swin_352_9["parameters"],
                },
                "status": (
                    "pass"
                    if swin_224_9["parameters"]
                    == EXPECTED_COUNTS["official_swin_unet_224_9class"]
                    else "fail"
                ),
            },
        },
        "architecture_verification": architecture,
        "resolution_dependent_difference": resolution_difference,
        "reported_values": {
            "paper_s2mnet": {
                "evidence_class": "reported_external_value",
                "value": "4.7M",
                "exact_integer_available": False,
                "conventional_one_decimal_rounding_of_observed_exact_count":
                    "4.8M",
            },
            "paper_transunet_ratio": {
                "evidence_class": "reported_external_value",
                "value": "approximately 13x smaller",
                "explicit_baseline_count_in_paper": None,
            },
            "readme_transunet": {
                "evidence_class": "reported_external_value",
                "parameters": 60_000_000,
                "displayed_ratio": 12.8,
                "matching_executable_configuration": None,
            },
            "paper_swin_unet_ratio": {
                "evidence_class": "reported_external_value",
                "value": "approximately 6x smaller",
                "explicit_baseline_count_in_paper": None,
                "matching_citation": None,
            },
            "readme_swin_unet": {
                "evidence_class": "reported_external_value",
                "parameters": 27_000_000,
                "matching_executable_configuration": None,
            },
        },
        "ratios_against_s2mnet_352": ratios,
        "claim_subassertions": {
            "s2mnet_has_approximately_4_7m_parameters": {
                "status": "partially_verified",
                "observed_exact_count": count_352["total"],
                "assessment": (
                    "The exact released 352x352 count is verified, but 4,791,544 "
                    "conventionally rounds to 4.8M rather than 4.7M."
                ),
            },
            "s2mnet_has_five_encoder_stages_with_channels_24_32_64_80_128": {
                "status": (
                    "verified" if architecture["status"] == "pass" else "failed"
                ),
                "observed_stage_count": architecture["observed"]["stage_count"],
                "observed_channels": architecture["observed"]["channels"],
            },
            "s2mnet_is_approximately_13x_smaller_than_transunet": {
                "status": "partially_verified",
                "reported_60m_ratio_using_exact_s2mnet": ratios[
                    "transunet_readme_reported_60_0m"
                ]["value"],
                "official_default_ratio_using_exact_s2mnet": ratios[
                    "official_transunet_224_9class_ledger"
                ]["value"],
                "assessment": (
                    "The reported 60.0M arithmetic is approximately 13x at "
                    "integer precision, but its TransUNet configuration is absent. "
                    "The pinned official default ledger gives about 22x instead."
                ),
            },
            "s2mnet_is_approximately_6x_smaller_than_swin_unet": {
                "status": "partially_verified",
                "reported_27m_ratio_using_exact_s2mnet": ratios[
                    "swin_unet_readme_reported_27_0m"
                ]["value"],
                "official_tiny_lite_ratio_using_exact_s2mnet": ratios[
                    "official_swin_unet_224_9class_ledger"
                ]["value"],
                "assessment": (
                    "Both ratios round to 6x at integer precision, and the "
                    "official Tiny/lite ledger is close to 27.0M, but the S2M-Net "
                    "materials do not identify that baseline configuration."
                ),
            },
        },
        "overall_claim1_assessment": {
            "status": "partially_verified",
            "recommended_verdict_change": False,
            "reason": (
                "The executable audit strengthens reproducibility of the exact "
                "released architecture and independent baseline counts, but it "
                "cannot recover the paper's TransUNet or Swin-Unet configurations "
                "and does not turn 4,791,544 into conventional 4.7M rounding."
            ),
        },
        "limitations_and_unresolved_ambiguities": [
            (
                "The released S2M-Net parameter count is resolution-dependent "
                "because SSTM allocates feature-size-clamped frequency grids."
            ),
            (
                "The paper's exact TransUNet configuration and the provenance of "
                "the README's rounded 60.0M value remain unavailable."
            ),
            (
                "The paper does not cite or configure the 2D Swin-Unet behind its "
                "comparison; the official Tiny/lite configuration is plausible "
                "but inferential."
            ),
            (
                "PyTorch is unavailable, so the two upstream constructors were "
                "not executed in this environment; their counts are transparent "
                "per-tensor ledgers from pinned source, not runtime claims."
            ),
            (
                "The bundled TensorFlow TransUNet creates its positional embedding "
                "as an untracked raw tf.Variable; Keras count_params excludes it."
            ),
            (
                "Paper/README values are rounded reports and do not provide "
                "trainable/non-trainable partitions."
            ),
            (
                "This is a build-only parameter audit. It does not validate "
                "training, checkpoints, data, segmentation quality, FLOPs, latency, "
                "or memory."
            ),
        ],
        "internal_verifications": checks,
    }
    report["audit"]["status"] = (
        "pass"
        if all(check["status"] == "pass" for check in checks)
        else "fail"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Deterministic JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    report = build_audit_report()
    payload = write_deterministic_json(output, report)
    parsed = json.loads(payload)
    round_trip = deterministic_json_bytes(parsed)
    if round_trip != payload:
        print("DETERMINISTIC_JSON_ROUND_TRIP=FAIL")
        raise SystemExit(1)

    relative_output = (
        output.relative_to(PROJECT_ROOT)
        if output.is_relative_to(PROJECT_ROOT)
        else output
    )
    print(f"CLAIM1_AUDIT_STATUS={report['audit']['status'].upper()}")
    print(f"OUTPUT={relative_output}")
    print(f"OUTPUT_SHA256={hashlib.sha256(payload).hexdigest()}")
    print(
        "S2MNET_352_PARAMETERS="
        f"{report['models']['s2mnet_352']['observed_parameters']['total']}"
    )
    print(
        "S2MNET_256_PARAMETERS="
        f"{report['models']['s2mnet_256']['observed_parameters']['total']}"
    )
    print(
        "OFFICIAL_TRANSUNET_LEDGER_PARAMETERS="
        f"{report['models']['official_transunet_r50_vit_b16_224_9class']['observed_parameters']['total']}"
    )
    print(
        "OFFICIAL_SWIN_UNET_LEDGER_PARAMETERS="
        f"{report['models']['official_swin_unet_tiny_lite_224_9class']['observed_parameters']['total']}"
    )
    if report["audit"]["status"] != "pass":
        failures = [
            check["name"]
            for check in report["internal_verifications"]
            if check["status"] != "pass"
        ]
        print("FAILED_VERIFICATIONS=" + ",".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
