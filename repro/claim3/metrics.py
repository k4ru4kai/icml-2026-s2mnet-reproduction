"""Independent native-grid evaluator and controlled NumPy loss."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .contract import CLASS_NAMES, IGNORE_INDEX


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def common_loss_numpy(
    logits: np.ndarray, target: np.ndarray, eps: float = 1e-6
) -> tuple[float, float, float]:
    if logits.shape[:-1] != target.shape or logits.shape[-1] != 8:
        raise ValueError(f"Incompatible logits/target: {logits.shape}/{target.shape}")
    valid = target != IGNORE_INDEX
    if not np.any(valid):
        raise ValueError("Loss batch has no valid pixels")
    probabilities = softmax_numpy(logits.astype(np.float64))
    valid_target = target[valid].astype(np.int64)
    ce = -np.log(
        np.clip(probabilities[valid][np.arange(valid_target.size), valid_target], eps, 1.0)
    ).mean()
    dice_values = []
    for class_id in range(1, 8):
        truth = (target == class_id) & valid
        if not np.any(truth):
            continue
        prediction = probabilities[..., class_id][valid]
        truth_valid = truth[valid].astype(np.float64)
        dice_values.append(
            (2.0 * np.sum(prediction * truth_valid) + eps)
            / (np.sum(prediction) + np.sum(truth_valid) + eps)
        )
    dice_loss = 0.0 if not dice_values else 1.0 - float(np.mean(dice_values))
    return float(ce + dice_loss), float(ce), float(dice_loss)


@dataclass
class PooledForegroundMetrics:
    tp: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.int64))
    fp: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.int64))
    fn: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.int64))

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        if target.shape != prediction.shape:
            raise ValueError(f"Shape mismatch: {target.shape}/{prediction.shape}")
        valid = target != IGNORE_INDEX
        for offset, class_id in enumerate(range(1, 8)):
            truth = (target == class_id) & valid
            predicted = (prediction == class_id) & valid
            self.tp[offset] += np.count_nonzero(truth & predicted)
            self.fp[offset] += np.count_nonzero(~truth & predicted)
            self.fn[offset] += np.count_nonzero(truth & ~predicted)

    def result(self) -> dict:
        classes = {}
        dice_for_macro = []
        iou_for_macro = []
        for offset, class_id in enumerate(range(1, 8)):
            tp, fp, fn = (
                int(self.tp[offset]),
                int(self.fp[offset]),
                int(self.fn[offset]),
            )
            if tp + fp + fn == 0:
                dice = None
                iou = None
            else:
                dice = 2.0 * tp / (2 * tp + fp + fn)
                iou = tp / (tp + fp + fn)
                dice_for_macro.append(dice)
                iou_for_macro.append(iou)
            classes[str(class_id)] = {
                "name": CLASS_NAMES[class_id],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": dice,
                "iou": iou,
            }
        return {
            "classes": classes,
            "foreground_macro_dice": (
                None if not dice_for_macro else float(np.mean(dice_for_macro))
            ),
            "foreground_macro_iou": (
                None if not iou_for_macro else float(np.mean(iou_for_macro))
            ),
        }


def native_prediction(probabilities: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    if probabilities.ndim != 3 or probabilities.shape[-1] != 8:
        raise ValueError(f"Expected HWC probabilities with 8 channels: {probabilities.shape}")
    height, width = native_shape
    resized = np.stack(
        [
            cv2.resize(
                probabilities[..., class_id],
                (width, height),
                interpolation=cv2.INTER_LINEAR_EXACT,
            )
            for class_id in range(8)
        ],
        axis=-1,
    )
    return np.argmax(resized, axis=-1).astype(np.uint8)
