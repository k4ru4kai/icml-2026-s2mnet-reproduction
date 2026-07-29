"""Run manifests, environment capture, and framework-neutral campaign paths."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .contract import CONTRACT


def run_directory(output_root: Path, model: str, seed: int) -> Path:
    if model not in {"s2mnet", "umamba"}:
        raise ValueError(model)
    return output_root / f"{model}_seed{seed}"


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def environment_payload(project_root: Path) -> dict:
    gpu = None
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu,
        "project_head": _git_value(project_root, "rev-parse", "HEAD"),
        "project_status": _git_value(project_root, "status", "--short"),
        "s2mnet_head": _git_value(project_root / "official_repo", "rev-parse", "HEAD"),
        "umamba_head": _git_value(
            project_root / "baselines" / "U-Mamba", "rev-parse", "HEAD"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def frozen_run_payload(model: str, seed: int, micro_batch: int = 2) -> dict:
    payload = {
        "schema_version": 1,
        "model": model,
        "seed": seed,
        "contract": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(CONTRACT).items()
        },
        "micro_batch_size": micro_batch,
        "gradient_accumulation": CONTRACT.effective_batch_size // micro_batch,
        "mixed_precision": False,
        "deep_supervision": False,
        "tta": False,
    }
    payload["config_sha256"] = canonical_json_hash(payload)
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
