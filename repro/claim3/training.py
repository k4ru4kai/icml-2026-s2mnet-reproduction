"""Shared batching and framework-specific one-step primitives.

The full campaign controller intentionally requires an explicit confirmation
flag in ``repro.experiments.claim3_train``. These primitives are also used by
the targeted smoke tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .contract import CONTRACT, learning_rate
from .data import FrameRecord, prepare_sample
from .losses import tensorflow_common_loss, torch_common_loss


def materialize_effective_batch(
    records: Sequence[FrameRecord],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    seed: int,
    training_pass: int,
    first_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(records) != CONTRACT.effective_batch_size:
        raise ValueError("An optimizer step requires exactly four samples")
    samples = [
        prepare_sample(
            record,
            mean,
            std,
            seed=seed,
            training_pass=training_pass,
            shuffled_position=first_position + offset,
        )
        for offset, record in enumerate(records)
    ]
    return (
        np.stack([sample[0] for sample in samples]).astype(np.float32),
        np.stack([sample[1] for sample in samples]).astype(np.uint8),
    )


def tensorflow_one_step(
    model,
    optimizer,
    images: np.ndarray,
    targets: np.ndarray,
    micro_batch: int = 2,
):
    import tensorflow as tf

    if micro_batch <= 0 or CONTRACT.effective_batch_size % micro_batch:
        raise ValueError("micro_batch must divide the fixed effective batch size")
    with tf.GradientTape() as tape:
        micro_logits = []
        for start in range(0, CONTRACT.effective_batch_size, micro_batch):
            micro_logits.append(
                model(images[start : start + micro_batch], training=True)
            )
        logits = tf.concat(micro_logits, axis=0)
        total, ce, dice = tensorflow_common_loss(logits, targets)
        tf.debugging.check_numerics(total, "non-finite Claim 3 loss")
    gradients = tape.gradient(total, model.trainable_variables)
    gradients, norm = tf.clip_by_global_norm(gradients, CONTRACT.gradient_clip_norm)
    tf.debugging.check_numerics(norm, "non-finite Claim 3 gradient norm")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {
        "loss": float(total.numpy()),
        "ce": float(ce.numpy()),
        "dice_loss": float(dice.numpy()),
        "gradient_norm": float(norm.numpy()),
        "output_shape": tuple(logits.shape),
    }


def apply_tensorflow_weight_decay(model, learning_rate_value) -> None:
    for variable in model.trainable_variables:
        lower = variable.name.lower()
        if len(variable.shape) > 1 and not any(
            token in lower for token in ("bias", "norm", "bn", "beta", "gamma")
        ):
            variable.assign_sub(
                learning_rate_value * CONTRACT.weight_decay * variable
            )


def torch_one_step(
    model,
    optimizer,
    images: np.ndarray,
    targets: np.ndarray,
    device,
    micro_batch: int = 2,
):
    import torch

    if micro_batch <= 0 or CONTRACT.effective_batch_size % micro_batch:
        raise ValueError("micro_batch must divide the fixed effective batch size")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tensors = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    target_tensor = torch.from_numpy(targets.astype(np.int64)).to(device)
    micro_logits = []
    for start in range(0, CONTRACT.effective_batch_size, micro_batch):
        micro_logits.append(model(tensors[start : start + micro_batch]))
    logits = torch.cat(micro_logits, dim=0)
    total, ce, dice = torch_common_loss(logits, target_tensor)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite Claim 3 loss")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), CONTRACT.gradient_clip_norm
    )
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("non-finite Claim 3 gradient norm")
    optimizer.step()
    return {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "gradient_norm": float(norm.detach().cpu()),
        "output_shape": tuple(logits.shape),
    }


def torch_optimizer(model, step: int = 1):
    import torch

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": CONTRACT.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate(step),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
