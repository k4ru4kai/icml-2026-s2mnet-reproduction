"""Equivalent TensorFlow and PyTorch implementations of the common loss."""

from __future__ import annotations


def tensorflow_common_loss(logits, target, ignore_index: int = 255, eps: float = 1e-6):
    import tensorflow as tf

    target = tf.cast(target, tf.int32)
    valid = tf.not_equal(target, ignore_index)
    safe_target = tf.where(valid, target, tf.zeros_like(target))
    per_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=safe_target, logits=logits
    )
    valid_float = tf.cast(valid, tf.float32)
    ce = tf.reduce_sum(per_pixel * valid_float) / tf.maximum(
        tf.reduce_sum(valid_float), 1.0
    )
    probabilities = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
    dice_sum = tf.constant(0.0, tf.float32)
    present_count = tf.constant(0.0, tf.float32)
    for class_id in range(1, 8):
        truth = tf.cast(tf.equal(safe_target, class_id) & valid, tf.float32)
        ground_truth_sum = tf.reduce_sum(truth)
        prediction = probabilities[..., class_id] * valid_float
        intersection = tf.reduce_sum(prediction * truth)
        dice = (2.0 * intersection + eps) / (
            tf.reduce_sum(prediction) + ground_truth_sum + eps
        )
        present = tf.cast(ground_truth_sum > 0, tf.float32)
        dice_sum += present * dice
        present_count += present
    dice_loss = tf.where(
        present_count > 0, 1.0 - dice_sum / present_count, tf.constant(0.0)
    )
    return ce + dice_loss, ce, dice_loss


def torch_common_loss(logits, target, ignore_index: int = 255, eps: float = 1e-6):
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 4 or logits.shape[1] != 8:
        raise ValueError(f"Expected NCHW logits with 8 channels: {tuple(logits.shape)}")
    target = target.long()
    valid = target != ignore_index
    if not bool(valid.any()):
        raise ValueError("Loss batch has no valid pixels")
    ce = functional.cross_entropy(logits, target, ignore_index=ignore_index)
    probabilities = torch.softmax(logits.float(), dim=1)
    dice_values = []
    for class_id in range(1, 8):
        truth = ((target == class_id) & valid).float()
        if not bool(truth.any()):
            continue
        prediction = probabilities[:, class_id] * valid.float()
        intersection = torch.sum(prediction * truth)
        dice_values.append(
            (2.0 * intersection + eps)
            / (torch.sum(prediction) + torch.sum(truth) + eps)
        )
    dice_loss = (
        logits.new_tensor(0.0, dtype=torch.float32)
        if not dice_values
        else 1.0 - torch.stack(dice_values).mean()
    )
    return ce + dice_loss, ce, dice_loss
