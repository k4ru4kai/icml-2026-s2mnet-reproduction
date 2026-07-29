"""Executable paired training, selection, and held-out evaluation engine.

This module is intentionally imported by one framework-specific Python
environment at a time. Dataset, transforms, loss semantics, and evaluation stay
shared; only model/optimizer/checkpoint plumbing branches.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from .contract import CONTRACT, IGNORE_INDEX, learning_rate
from .data import (
    FrameRecord,
    build_inventory,
    combine_type_masks,
    effective_batches,
    load_rgb,
    prepare_sample,
    records_for_split,
)
from .metrics import PooledForegroundMetrics, native_prediction
from .models.s2mnet_adapter import build_s2mnet_logits
from .runtime import frozen_run_payload, run_directory, write_json
from .training import (
    apply_tensorflow_weight_decay,
    materialize_effective_batch,
    tensorflow_one_step,
    torch_one_step,
    torch_optimizer,
)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _probability_statistics(
    probabilities: np.ndarray,
    target: np.ndarray,
    accumulator: dict,
    eps: float = 1e-6,
) -> None:
    valid = target != IGNORE_INDEX
    safe = np.where(valid, target, 0).astype(np.int64)
    selected = probabilities[
        np.indices(target.shape)[0],
        np.indices(target.shape)[1],
        np.indices(target.shape)[2],
        safe,
    ]
    accumulator["ce_sum"] += float(
        -np.log(np.clip(selected[valid], eps, 1.0)).sum(dtype=np.float64)
    )
    accumulator["valid_count"] += int(valid.sum())
    for class_id in range(1, 8):
        truth = ((target == class_id) & valid).astype(np.float64)
        prediction = probabilities[..., class_id] * valid
        accumulator["intersection"][class_id - 1] += float(
            np.sum(prediction * truth, dtype=np.float64)
        )
        accumulator["prediction"][class_id - 1] += float(
            np.sum(prediction, dtype=np.float64)
        )
        accumulator["truth"][class_id - 1] += float(
            np.sum(truth, dtype=np.float64)
        )


def _pooled_loss(accumulator: dict, eps: float = 1e-6) -> dict:
    ce = accumulator["ce_sum"] / accumulator["valid_count"]
    present = accumulator["truth"] > 0
    dice = (
        2.0 * accumulator["intersection"][present] + eps
    ) / (
        accumulator["prediction"][present]
        + accumulator["truth"][present]
        + eps
    )
    dice_loss = 0.0 if not np.any(present) else 1.0 - float(np.mean(dice))
    return {"loss": float(ce + dice_loss), "ce": float(ce), "dice_loss": dice_loss}


def _empty_probability_accumulator() -> dict:
    return {
        "ce_sum": 0.0,
        "valid_count": 0,
        "intersection": np.zeros(7, dtype=np.float64),
        "prediction": np.zeros(7, dtype=np.float64),
        "truth": np.zeros(7, dtype=np.float64),
    }


def validate_model(
    model,
    model_name: str,
    records: tuple[FrameRecord, ...],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int = 2,
) -> dict:
    accumulator = _empty_probability_accumulator()
    validation = records_for_split(records, "validation")
    for start in range(0, len(validation), batch_size):
        samples = [prepare_sample(record, mean, std) for record in validation[start : start + batch_size]]
        images = np.stack([sample[0] for sample in samples]).astype(np.float32)
        targets = np.stack([sample[1] for sample in samples]).astype(np.uint8)
        if model_name == "s2mnet":
            import tensorflow as tf

            logits = model(images, training=False).numpy()
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        else:
            import torch

            device = next(model.parameters()).device
            with torch.no_grad():
                tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
                logits = model(tensor)
                probabilities = (
                    torch.softmax(logits.float(), dim=1)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
        _probability_statistics(probabilities, targets, accumulator)
    result = _pooled_loss(accumulator)
    result["frames"] = len(validation)
    return result


def evaluate_selected(
    model,
    model_name: str,
    records: tuple[FrameRecord, ...],
    mean: np.ndarray,
    std: np.ndarray,
    prediction_root: Path,
) -> dict:
    pooled = PooledForegroundMetrics()
    per_sequence = {}
    prediction_hashes = {}
    timings = []
    prediction_root.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records_for_split(records, "test")):
        image, _ = prepare_sample(record, mean, std)
        start = time.perf_counter()
        if model_name == "s2mnet":
            import tensorflow as tf

            probabilities = tf.nn.softmax(
                model(image[None], training=False), axis=-1
            ).numpy()[0]
        else:
            import torch

            device = next(model.parameters()).device
            with torch.no_grad():
                tensor = (
                    torch.from_numpy(image[None]).permute(0, 3, 1, 2).to(device)
                )
                probabilities = (
                    torch.softmax(model(tensor).float(), dim=1)[0]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
        if model_name == "umamba":
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        native_target, _ = combine_type_masks(record.masks)
        prediction = native_prediction(probabilities, native_target.shape)
        pooled.update(native_target, prediction)
        sequence_metric = per_sequence.setdefault(
            record.sequence, PooledForegroundMetrics()
        )
        sequence_metric.update(native_target, prediction)
        sequence_dir = prediction_root / f"instrument_dataset_{record.sequence}"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = sequence_dir / f"frame{record.frame_id:03d}.png"
        if not cv2.imwrite(str(prediction_path), prediction):
            raise OSError(f"Failed to write prediction for {record.key}")
        prediction_hashes[record.key] = hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest()
    result = pooled.result()
    result["per_sequence"] = {
        str(sequence): metric.result() for sequence, metric in per_sequence.items()
    }
    result["prediction_sha256"] = prediction_hashes
    measured = timings[20:]
    result["runtime"] = {
        "warmup_frames": min(20, len(timings)),
        "measured_frames": len(measured),
        "mean_seconds": None if not measured else float(np.mean(measured)),
        "std_seconds": None if not measured else float(np.std(measured, ddof=1)),
    }
    return result


def _set_common_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_campaign(output_root: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((output_root / "campaign.json").read_text())
    return (
        np.asarray(payload["normalization_mean"], dtype=np.float32),
        np.asarray(payload["normalization_std"], dtype=np.float32),
    )


def _selected_state(run_root: Path) -> tuple[float, int | None]:
    selection_path = run_root / "checkpoints" / "best" / "selection.json"
    if not selection_path.is_file():
        return float("inf"), None
    selection = json.loads(selection_path.read_text())
    return float(selection["validation"]["loss"]), int(selection["step"])


def train_s2mnet(
    dataset_root: Path, output_root: Path, seed: int, micro_batch: int = 2
) -> dict:
    import tensorflow as tf

    _set_common_seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.reset_memory_stats("GPU:0")
    records = build_inventory(dataset_root)
    mean, std = _load_campaign(output_root)
    run_root = run_directory(output_root, "s2mnet", seed)
    log_path = run_root / "logs" / "metrics.jsonl"
    model = build_s2mnet_logits()
    learning_rate_variable = tf.Variable(learning_rate(1), trainable=False)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_variable, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    )
    checkpoint_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        step=checkpoint_step, model=model, optimizer=optimizer
    )
    latest = tf.train.CheckpointManager(
        checkpoint, str(run_root / "checkpoints" / "latest"), max_to_keep=2
    )
    best_loss, best_step = _selected_state(run_root)
    resumed_from_step = 0
    if latest.latest_checkpoint:
        checkpoint.restore(latest.latest_checkpoint).assert_existing_objects_matched()
        resumed_from_step = int(checkpoint_step.numpy())
    started = time.time()
    for step, training_pass, batch_records in effective_batches(records, seed):
        if step <= resumed_from_step:
            continue
        learning_rate_variable.assign(learning_rate(step))
        position = ((step - 1) % CONTRACT.validation_frequency) * 4
        images, targets = materialize_effective_batch(
            batch_records,
            mean,
            std,
            seed=seed,
            training_pass=training_pass,
            first_position=position,
        )
        train_metrics = tensorflow_one_step(
            model, optimizer, images, targets, micro_batch=micro_batch
        )
        apply_tensorflow_weight_decay(model, learning_rate_variable)
        if step <= 5 or step % 25 == 0:
            _append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": step,
                    "training_pass": training_pass + 1,
                    "learning_rate": learning_rate(step),
                    "train": train_metrics,
                },
            )
        if step % CONTRACT.validation_frequency == 0:
            validation = validate_model(model, "s2mnet", records, mean, std)
            row = {
                "event": "validation",
                "step": step,
                "training_pass": training_pass + 1,
                "learning_rate": learning_rate(step),
                "train": train_metrics,
                "validation": validation,
            }
            _append_jsonl(log_path, row)
            checkpoint_step.assign(step)
            latest.save(checkpoint_number=step)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_step = step
                best_dir = run_root / "checkpoints" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_weights(str(best_dir / "weights"))
                write_json(
                    best_dir / "selection.json",
                    {"step": step, "validation": validation},
                )
    status = {
        "status": "trained",
        "model": "s2mnet",
        "seed": seed,
        "optimizer_steps": CONTRACT.maximum_optimizer_steps,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "micro_batch_size": micro_batch,
        "gradient_accumulation": CONTRACT.effective_batch_size // micro_batch,
        "resumed_from_step": resumed_from_step,
        "parameter_count": int(model.count_params()),
        "peak_gpu_bytes": int(
            tf.config.experimental.get_memory_info("GPU:0")["peak"]
        ),
    }
    write_json(run_root / "status.json", status)
    return status


def train_umamba(
    dataset_root: Path, output_root: Path, seed: int, micro_batch: int = 2
) -> dict:
    import torch

    from .models.umamba_adapter import build_umamba_bot

    _set_common_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    records = build_inventory(dataset_root)
    mean, std = _load_campaign(output_root)
    run_root = run_directory(output_root, "umamba", seed)
    log_path = run_root / "logs" / "metrics.jsonl"
    model = build_umamba_bot().to(device)
    optimizer = torch_optimizer(model)
    best_loss, best_step = _selected_state(run_root)
    resumed_from_step = 0
    recovery_path = run_root / "checkpoints" / "latest" / "checkpoint.pt"
    if recovery_path.is_file():
        recovery = torch.load(recovery_path, map_location=device)
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        torch.set_rng_state(recovery["torch_rng"])
        torch.cuda.set_rng_state_all(recovery["cuda_rng"])
        np.random.set_state(recovery["numpy_rng"])
        random.setstate(recovery["python_rng"])
        resumed_from_step = int(recovery["step"])
    started = time.time()
    for step, training_pass, batch_records in effective_batches(records, seed):
        if step <= resumed_from_step:
            continue
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        position = ((step - 1) % CONTRACT.validation_frequency) * 4
        images, targets = materialize_effective_batch(
            batch_records,
            mean,
            std,
            seed=seed,
            training_pass=training_pass,
            first_position=position,
        )
        train_metrics = torch_one_step(
            model, optimizer, images, targets, device, micro_batch=micro_batch
        )
        if step <= 5 or step % 25 == 0:
            _append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": step,
                    "training_pass": training_pass + 1,
                    "learning_rate": learning_rate(step),
                    "train": train_metrics,
                },
            )
        if step % CONTRACT.validation_frequency == 0:
            validation = validate_model(model, "umamba", records, mean, std)
            row = {
                "event": "validation",
                "step": step,
                "training_pass": training_pass + 1,
                "learning_rate": learning_rate(step),
                "train": train_metrics,
                "validation": validation,
            }
            _append_jsonl(log_path, row)
            latest_dir = run_root / "checkpoints" / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                },
                latest_dir / "checkpoint.pt",
            )
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_step = step
                best_dir = run_root / "checkpoints" / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_dir / "weights.pt")
                write_json(
                    best_dir / "selection.json",
                    {"step": step, "validation": validation},
                )
    status = {
        "status": "trained",
        "model": "umamba",
        "seed": seed,
        "optimizer_steps": CONTRACT.maximum_optimizer_steps,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "micro_batch_size": micro_batch,
        "gradient_accumulation": CONTRACT.effective_batch_size // micro_batch,
        "resumed_from_step": resumed_from_step,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(run_root / "status.json", status)
    return status


def evaluate_worker(
    model_name: str, dataset_root: Path, output_root: Path, seed: int
) -> dict:
    records = build_inventory(dataset_root)
    mean, std = _load_campaign(output_root)
    run_root = run_directory(output_root, model_name, seed)
    if model_name == "s2mnet":
        model = build_s2mnet_logits()
        model.load_weights(str(run_root / "checkpoints" / "best" / "weights"))
    else:
        import torch

        from .models.umamba_adapter import build_umamba_bot

        model = build_umamba_bot().to("cuda")
        state = torch.load(
            run_root / "checkpoints" / "best" / "weights.pt",
            map_location="cuda",
        )
        model.load_state_dict(state)
        model.eval()
    result = evaluate_selected(
        model,
        model_name,
        records,
        mean,
        std,
        run_root / "predictions" / "test",
    )
    write_json(run_root / "metrics" / "test.json", result)
    status_path = run_root / "status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "completed"
    status["test_foreground_macro_dice"] = result["foreground_macro_dice"]
    write_json(status_path, status)
    return result
