#!/usr/bin/env python3
"""Executable audit for S2M-Net Claim 2.

This diagnostic keeps three questions separate:

1. What the released SSTM implementation actually does.
2. What a paper-intended centred spatial K x K crop retains.
3. Whether the reported 63% cost reduction can be reconstructed.

The default audit is deterministic and does not require TensorFlow, OpenCV,
datasets, or checkpoints. Optional ``--image`` arguments run the released
raw-RGB spectral-energy convention on explicitly named, non-hidden images.
No directory is scanned automatically.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "3ec59668ab9b438ab9b170306d29b01e9270fd5a"
BLOCKS_PATH = REPO_ROOT / "official_repo/s2mnet/models/blocks.py"
DEFAULT_CONFIG_PATH = REPO_ROOT / "official_repo/configs/default.yaml"

CANONICAL_INPUT_SIZE = 352
CANONICAL_STAGE_SHAPES = (
    (176, 176, 24),
    (88, 88, 32),
    (44, 44, 64),
    (22, 22, 80),
    (11, 11, 128),
)
CONFIGURED_K = 32


class Claim2AuditError(RuntimeError):
    """Raised when a required source or audit invariant is unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise Claim2AuditError(f"Class {name!r} not found in {BLOCKS_PATH}")


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise Claim2AuditError(f"Method {name!r} not found in {class_node.name}")


def _calls(node: ast.AST) -> list[str]:
    return [
        _call_name(candidate.func)
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
    ]


def _subscripted_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Subscript):
            continue
        value = candidate.value
        if isinstance(value, ast.Name):
            names.append(value.id)
        elif isinstance(value, ast.Attribute):
            names.append(_call_name(value))
    return names


def _parse_yaml_int(text: str, key: str) -> int:
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise Claim2AuditError(f"Integer key {key!r} not found in default config")
    return int(match.group(1))


def _parse_yaml_bool_list(text: str, key: str) -> list[bool]:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*:\s*\[([^\]]*)\]\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        raise Claim2AuditError(f"Boolean-list key {key!r} not found in default config")
    values = [item.strip().lower() for item in match.group(1).split(",")]
    if any(value not in {"true", "false"} for value in values):
        raise Claim2AuditError(f"Unexpected boolean list for {key!r}: {values}")
    return [value == "true" for value in values]


def audit_released_sstm() -> dict[str, Any]:
    """Audit the pinned released SSTM source without importing TensorFlow."""

    if not BLOCKS_PATH.is_file() or not DEFAULT_CONFIG_PATH.is_file():
        raise Claim2AuditError(
            "Pinned official_repo submodule is unavailable; run "
            "`git submodule update --init --recursive`."
        )

    blocks_text = BLOCKS_PATH.read_text(encoding="utf-8")
    config_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(blocks_text, filename=str(BLOCKS_PATH))
    sstm_class = _find_class(tree, "SpectralSelectiveTokenMixer")
    build = _find_method(sstm_class, "build")
    spectral = _find_method(sstm_class, "_spectral_path")
    selective = _find_method(sstm_class, "_ssm_path")

    spectral_calls = _calls(spectral)
    build_calls = _calls(build)
    selective_calls = _calls(selective)
    forbidden_shift_or_padding = {
        "tf.signal.fftshift",
        "tf.signal.ifftshift",
        "tf.pad",
        "np.fft.fftshift",
        "np.fft.ifftshift",
        "numpy.fft.fftshift",
        "numpy.fft.ifftshift",
    }

    configured_k = _parse_yaml_int(config_text, "sstm_k")
    spectral_mask = _parse_yaml_bool_list(config_text, "sstm_use_spectral")
    selective_mask = _parse_yaml_bool_list(config_text, "sstm_use_ssm")
    stage_records = []
    for index, (height, width, channels) in enumerate(CANONICAL_STAGE_SHAPES):
        effective_k = min(configured_k, height, width)
        stage_records.append(
            {
                "stage": index + 1,
                "input_shape_hwc": [height, width, channels],
                "configured_k": configured_k,
                "effective_k": effective_k,
                "operational_resampled_shape_hwc": [
                    effective_k,
                    effective_k,
                    channels,
                ],
                "paper_crop_coefficient_fraction": (
                    effective_k * effective_k / (height * width)
                ),
                "spectral_branch": spectral_mask[index],
                "selective_branch": selective_mask[index],
            }
        )

    checks = {
        "default_k_is_32": configured_k == CONFIGURED_K,
        "spectral_enabled_all_five_stages": spectral_mask == [True] * 5,
        "selective_enabled_only_stages_3_to_5": selective_mask
        == [False, False, True, True, True],
        "calls_fft2d": spectral_calls.count("tf.signal.fft2d") == 1,
        "calls_ifft2d": spectral_calls.count("tf.signal.ifft2d") == 1,
        "does_not_transpose_before_fft": not any(
            call in {"tf.transpose", "np.transpose", "numpy.transpose"}
            for call in spectral_calls
        ),
        "uses_four_bilinear_resize_calls": spectral_calls.count("tf.image.resize") == 4,
        "contains_no_fft_shift_or_zero_pad": forbidden_shift_or_padding.isdisjoint(
            spectral_calls
        ),
        "does_not_slice_fft_tensor_X": "X" not in _subscripted_names(spectral),
        "creates_real_frequency_weights": (
            "self.add_weight" in build_calls
            and "dtype=tf.complex64" not in ast.get_source_segment(blocks_text, build)
        ),
        "selective_path_uses_dense_channel_projection": (
            selective_calls.count("self.ssm_gate") == 1
            and selective_calls.count("self.ssm_proj") == 1
        ),
        "selective_path_does_not_use_state_dim": "ssm_state_dim"
        not in ast.get_source_segment(blocks_text, selective),
        "effective_k_matches_32_32_32_22_11": [
            record["effective_k"] for record in stage_records
        ]
        == [32, 32, 32, 22, 11],
    }

    return {
        "source": {
            "upstream_commit": UPSTREAM_COMMIT,
            "blocks_path": str(BLOCKS_PATH.relative_to(REPO_ROOT)),
            "blocks_sha256": sha256_file(BLOCKS_PATH),
            "default_config_path": str(DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)),
            "default_config_sha256": sha256_file(DEFAULT_CONFIG_PATH),
        },
        "configured_k": configured_k,
        "canonical_input_size": CANONICAL_INPUT_SIZE,
        "fft_call_rank4_layout": "B,H,W,C",
        "tensorflow_fft2d_innermost_axes": ["W", "C"],
        "literal_retained_subset_in_released_path": False,
        "stage_records": stage_records,
        "checks": checks,
        "audit_pass": all(checks.values()),
    }


def centered_spatial_energy_retention(array: np.ndarray, k: int) -> float:
    """Return the paper-intended centred spatial K x K energy ratio.

    ``array`` must be HxW or HxWxC. Multichannel values are the arithmetic
    mean of per-channel ratios, matching the released analysis utility.
    """

    values = np.asarray(array)
    if values.ndim not in {2, 3}:
        raise ValueError(f"Expected HxW or HxWxC array, got shape {values.shape}")
    if k <= 0:
        raise ValueError("k must be positive")

    if values.ndim == 2:
        channels = [values]
    else:
        channels = [values[:, :, index] for index in range(values.shape[2])]

    ratios: list[float] = []
    for channel in channels:
        height, width = channel.shape
        effective_k = min(k, height, width)
        spectrum = np.fft.fftshift(np.fft.fft2(channel))
        start_y = height // 2 - effective_k // 2
        start_x = width // 2 - effective_k // 2
        crop = spectrum[
            start_y : start_y + effective_k,
            start_x : start_x + effective_k,
        ]
        total = float(np.sum(np.abs(spectrum) ** 2, dtype=np.float64))
        retained = float(np.sum(np.abs(crop) ** 2, dtype=np.float64))
        ratios.append(retained / (total + 1e-10))
    return float(np.mean(ratios))


def _synthetic_energy_self_checks() -> dict[str, Any]:
    constant = np.ones((64, 64), dtype=np.float32)
    checkerboard = (
        2 * (np.indices((64, 64), dtype=np.int32).sum(axis=0) % 2) - 1
    ).astype(np.float32)
    constant_ratio = centered_spatial_energy_retention(constant, 32)
    checkerboard_ratio = centered_spatial_energy_retention(checkerboard, 32)
    checks = {
        "constant_dc_energy_is_retained": math.isclose(
            constant_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "high_frequency_checkerboard_is_not_95_percent_retained": (
            checkerboard_ratio < 0.95
        ),
    }
    return {
        "constant_64x64_k32": constant_ratio,
        "checkerboard_64x64_k32": checkerboard_ratio,
        "checks": checks,
        "self_check_pass": all(checks.values()),
    }


def _reduction(reference: float, candidate: float) -> float:
    if reference <= 0:
        raise ValueError("reference cost must be positive")
    return 1.0 - candidate / reference


def audit_cost_claim() -> dict[str, Any]:
    """Recompute all non-equivalent arithmetic interpretations in the report."""

    alternatives = {
        "k32_square_vs_352_square_coefficient_count": _reduction(
            352.0**2, 32.0**2
        ),
        "32_vs_unexplained_256_coefficient_count": _reduction(256.0, 32.0),
        "paper_reported_4_2x_speedup_as_reduction": 1.0 - 1.0 / 4.2,
        "readme_whole_model_flops_11_2_vs_45_2": _reduction(45.2, 11.2),
        "readme_whole_model_runtime_10_1_vs_42_3": _reduction(42.3, 10.1),
    }

    stage_proxy = []
    attention_total = 0.0
    sstm_total = 0.0
    for stage, (height, width, channels) in enumerate(CANONICAL_STAGE_SHAPES, 1):
        tokens = height * width
        attention = float(tokens * tokens * channels)
        sstm = float(
            tokens * channels * math.log2(tokens)
            + tokens * channels * channels
        )
        attention_total += attention
        sstm_total += sstm
        stage_proxy.append(
            {
                "stage": stage,
                "shape_hwc": [height, width, channels],
                "unit_coefficient_proxy_reduction": _reduction(attention, sstm),
            }
        )

    matches_63 = {
        key: math.isclose(value, 0.63, rel_tol=0.0, abs_tol=5e-4)
        for key, value in alternatives.items()
    }
    return {
        "claim_target_reduction": 0.63,
        "matched_attention_baseline_specified": False,
        "named_cost_metric_specified": False,
        "alternatives_are_not_like_for_like_reproductions": True,
        "alternative_reductions": alternatives,
        "alternative_matches_63_percent": matches_63,
        "unit_coefficient_asymptotic_stage_proxy": stage_proxy,
        "unit_coefficient_asymptotic_aggregate_reduction": _reduction(
            attention_total, sstm_total
        ),
        "claim_reproducible_from_released_materials": False,
    }


def _load_explicit_image(path: Path, input_size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.shape[:2] != (input_size, input_size):
            raise Claim2AuditError(
                f"{path} has shape {image.shape}; .npy inputs must already be "
                f"{input_size}x{input_size}."
            )
        return np.asarray(image, dtype=np.float32)

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise Claim2AuditError(
            "OpenCV is required for image files; install requirements-repro.txt "
            "or pass preprocessed .npy arrays."
        ) from exc

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise Claim2AuditError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size))
    return resized.astype(np.float32) / 255.0


def analyze_explicit_images(
    paths: Iterable[Path], k: int = CONFIGURED_K, input_size: int = 352
) -> dict[str, Any] | None:
    """Analyze only explicitly supplied images; never discover a dataset."""

    records = []
    for path in paths:
        if not path.is_file():
            raise Claim2AuditError(f"Explicit image does not exist: {path}")
        image = _load_explicit_image(path, input_size)
        ratio = centered_spatial_energy_retention(image, k)
        records.append(
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "retained_energy_ratio": ratio,
            }
        )

    if not records:
        return None
    values = [record["retained_energy_ratio"] for record in records]
    return {
        "protocol": "released_raw_rgb_analysis_convention",
        "input_size": input_size,
        "k": k,
        "images": records,
        "summary": {
            "n": len(values),
            "mean": mean(values),
            "median": median(values),
            "minimum": min(values),
            "maximum": max(values),
            "sample_sd": stdev(values) if len(values) > 1 else None,
            "mean_exceeds_95_percent": mean(values) > 0.95,
        },
    }


def build_audit(explicit_images: Iterable[Path] = ()) -> dict[str, Any]:
    mechanism = audit_released_sstm()
    energy_self_checks = _synthetic_energy_self_checks()
    image_diagnostic = analyze_explicit_images(explicit_images)
    cost = audit_cost_claim()
    procedure_pass = mechanism["audit_pass"] and energy_self_checks["self_check_pass"]
    return {
        "schema_version": 1,
        "audit": "claim2_sstm_mechanism_energy_cost",
        "claim": (
            "SSTM uses a truncated 2D FFT with K=32, retains more than 95% "
            "spectral energy, and reduces computational cost by 63% relative "
            "to full spatial attention."
        ),
        "mechanism": mechanism,
        "energy": {
            "paper_intended_formula": (
                "sum(abs(center_crop_k(fftshift(fft2_hw(x))))**2) / "
                "sum(abs(fftshift(fft2_hw(x)))**2)"
            ),
            "released_forward_retained_subset_defined": False,
            "synthetic_self_checks": energy_self_checks,
            "explicit_image_diagnostic": image_diagnostic,
        },
        "cost": cost,
        "procedure": {
            "audit_pass": procedure_pass,
            "meaning": (
                "The deterministic verification procedure passed. This is "
                "not a PASS verdict for the scientific claim."
            ),
        },
        "component_verdicts": {
            "truncated_spatial_fft_k32": (
                "paper specification partially verified; contradicted by "
                "released forward implementation"
            ),
            "more_than_95_percent_energy": (
                "not verified for released SSTM; data-dependent diagnostic "
                "not run" if image_diagnostic is None else
                "limited explicit-image diagnostic only"
            ),
            "63_percent_cost_reduction": (
                "not verified; metric and matched attention baseline absent"
            ),
        },
        "scientific_verdict": "Not verified",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/audits/claim2_sstm_audit.json",
        help="JSON audit output path.",
    )
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit non-hidden image path. Repeat for each permitted image. "
            "Directories are intentionally unsupported."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    audit = build_audit(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"CLAIM2_AUDIT_PASS={str(audit['procedure']['audit_pass']).lower()}")
    print(f"SCIENTIFIC_VERDICT={audit['scientific_verdict']}")
    print(f"OUTPUT={args.output}")
    return 0 if audit["procedure"]["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
