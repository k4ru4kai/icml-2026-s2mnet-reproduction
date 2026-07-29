#!/usr/bin/env python3
"""Targeted Claim 3 smoke checks; never launches a full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro.claim3.data import (
    build_inventory,
    combine_type_masks,
    prepare_sample,
    records_for_split,
)
from repro.claim3.losses import tensorflow_common_loss, torch_common_loss
from repro.claim3.metrics import PooledForegroundMetrics
from repro.claim3.models.s2mnet_adapter import build_s2mnet_logits
from repro.claim3.training import (
    apply_tensorflow_weight_decay,
    materialize_effective_batch,
    tensorflow_one_step,
    torch_one_step,
    torch_optimizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/sarah/Datasets/EndoVis17_HF"),
    )
    parser.add_argument("--skip-s2mnet", action="store_true")
    parser.add_argument("--skip-umamba", action="store_true")
    parser.add_argument("--real-optimizer-step", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict = {}
    records = build_inventory(args.dataset_root)
    representatives = [
        records_for_split(records, split)[0]
        for split in ("train", "validation", "test")
    ]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    samples = [
        prepare_sample(record, mean, std, size=384) for record in representatives
    ]
    real_images, real_targets = materialize_effective_batch(
        records_for_split(records, "train")[:4],
        mean,
        std,
        seed=42,
        training_pass=0,
        first_position=0,
    )
    report["real_loader"] = {
        record.split: {
            "key": record.key,
            "image_shape": list(sample[0].shape),
            "target_shape": list(sample[1].shape),
            "target_values": np.unique(sample[1]).tolist(),
        }
        for record, sample in zip(representatives, samples)
    }

    controlled_target = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    controlled_prediction = np.array([[0, 1], [2, 7]], dtype=np.uint8)
    evaluator = PooledForegroundMetrics()
    evaluator.update(controlled_target, controlled_prediction)
    report["controlled_evaluator"] = evaluator.result()

    if not args.skip_s2mnet:
        import tensorflow as tf

        model = build_s2mnet_logits()
        optimizer = tf.keras.optimizers.Adam(1e-4)
        if args.real_optimizer_step:
            report["s2mnet"] = tensorflow_one_step(
                model, optimizer, real_images, real_targets
            )
            apply_tensorflow_weight_decay(model, 1e-4)
            report["s2mnet"]["data"] = "real_train_frames_000-003"
        else:
            image = tf.zeros((1, 384, 384, 3), tf.float32)
            target = tf.zeros((1, 384, 384), tf.uint8)
            with tf.GradientTape() as tape:
                logits = model(image, training=True)
                loss, ce, dice = tensorflow_common_loss(logits, target)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            )
            apply_tensorflow_weight_decay(model, 1e-4)
            report["s2mnet"] = {
                "output_shape": list(logits.shape),
                "loss": float(loss.numpy()),
                "ce": float(ce.numpy()),
                "dice_loss": float(dice.numpy()),
                "data": "synthetic",
            }
        try:
            report["s2mnet"]["peak_gpu_bytes"] = int(
                tf.config.experimental.get_memory_info("GPU:0")["peak"]
            )
        except (ValueError, RuntimeError):
            report["s2mnet"]["peak_gpu_bytes"] = None

    if not args.skip_umamba:
        import torch

        from repro.claim3.models.umamba_adapter import build_umamba_bot

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_umamba_bot().to(device)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        optimizer = torch_optimizer(model)
        if args.real_optimizer_step:
            report["umamba"] = torch_one_step(
                model, optimizer, real_images, real_targets, device
            )
            report["umamba"]["data"] = "real_train_frames_000-003"
        else:
            image = torch.zeros((1, 3, 384, 384), dtype=torch.float32, device=device)
            target = torch.zeros((1, 384, 384), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, ce, dice = torch_common_loss(logits, target)
            loss.backward()
            optimizer.step()
            report["umamba"] = {
                "output_shape": list(logits.shape),
                "loss": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "dice_loss": float(dice.detach().cpu()),
                "data": "synthetic",
            }
        report["umamba"]["device"] = str(device)
        report["umamba"]["peak_gpu_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
