"""Claim 3 paired-campaign entry point.

The launcher freezes manifests and run directories but refuses a campaign
unless the caller explicitly confirms that the published protocol was reviewed.
Training-loop completion is intentionally model-dispatched so each framework
can preserve native checkpoint state while sharing the scientific pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from repro.claim3.contract import CONTRACT, SEEDS, run_matrix
from repro.claim3.data import (
    build_inventory,
    compute_training_normalization,
    manifest_payload,
    write_manifest,
)
from repro.claim3.runtime import (
    environment_payload,
    frozen_run_payload,
    run_directory,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=CONTRACT.output_root)
    parser.add_argument(
        "--models", nargs="+", choices=("s2mnet", "umamba"), default=("s2mnet", "umamba")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--confirm-full-campaign", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-model", choices=("s2mnet", "umamba"))
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--worker-phase", choices=("train", "evaluate"))
    return parser.parse_args()


def prepare_campaign(args: argparse.Namespace) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    records = build_inventory(args.dataset_root)
    mean, std = compute_training_normalization(records)
    campaign = {
        "schema_version": 1,
        "protocol": "docs/claim3_endovis17_matched_protocol.md",
        "dataset_root": str(args.dataset_root.resolve()),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "run_matrix": [
            {"model": model, "seed": seed}
            for model, seed in run_matrix()
            if model in args.models and seed in args.seeds
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.output_root / "manifest" / "endovis17_eligible.json",
        manifest_payload(args.dataset_root, records, include_hashes=True),
    )
    write_json(args.output_root / "campaign.json", campaign)
    write_json(args.output_root / "environment.json", environment_payload(project_root))
    for row in campaign["run_matrix"]:
        run_root = run_directory(args.output_root, row["model"], row["seed"])
        write_json(
            run_root / "config" / "frozen.json",
            frozen_run_payload(row["model"], row["seed"]),
        )
    return campaign


def summarize_campaign(output_root: Path) -> dict:
    runs = []
    paired_differences = []
    for seed in SEEDS:
        paired = {}
        for model in ("s2mnet", "umamba"):
            run_root = run_directory(output_root, model, seed)
            status = json.loads((run_root / "status.json").read_text())
            metrics = json.loads((run_root / "metrics" / "test.json").read_text())
            if status.get("status") != "completed":
                raise RuntimeError(f"Incomplete run cannot be summarized: {model}/{seed}")
            dice = metrics["foreground_macro_dice"]
            paired[model] = dice
            runs.append({"model": model, "seed": seed, "foreground_macro_dice": dice})
        paired_differences.append(
            {
                "seed": seed,
                "s2mnet": paired["s2mnet"],
                "umamba": paired["umamba"],
                "s2mnet_minus_umamba": paired["s2mnet"] - paired["umamba"],
            }
        )
    differences = [row["s2mnet_minus_umamba"] for row in paired_differences]
    summary = {
        "schema_version": 1,
        "runs": runs,
        "paired_differences": paired_differences,
        "paired_difference_mean": statistics.mean(differences),
        "paired_difference_sample_std": statistics.stdev(differences),
    }
    summary_root = output_root / "summary"
    write_json(summary_root / "paired_results.json", summary)
    summary_root.mkdir(parents=True, exist_ok=True)
    with (summary_root / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(paired_differences[0]))
        writer.writeheader()
        writer.writerows(paired_differences)
    return summary


def main() -> int:
    args = parse_args()
    if args.worker_model:
        if args.worker_seed is None or args.worker_phase is None:
            raise SystemExit("Worker mode requires --worker-seed and --worker-phase")
        from repro.claim3.engine import evaluate_worker, train_s2mnet, train_umamba

        try:
            if args.worker_phase == "train":
                function = (
                    train_s2mnet if args.worker_model == "s2mnet" else train_umamba
                )
                result = function(
                    args.dataset_root, args.output_root, args.worker_seed
                )
            else:
                result = evaluate_worker(
                    args.worker_model,
                    args.dataset_root,
                    args.output_root,
                    args.worker_seed,
                )
        except Exception as error:
            failure = {
                "status": "failed",
                "model": args.worker_model,
                "seed": args.worker_seed,
                "phase": args.worker_phase,
                "failed_at_unix_seconds": time.time(),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(
                run_directory(args.output_root, args.worker_model, args.worker_seed)
                / "failure.json",
                failure,
            )
            raise
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not args.prepare_only and not args.confirm_full_campaign:
        raise SystemExit(
            "Refusing to train: pass --confirm-full-campaign only after campaign review"
        )
    if not args.prepare_only and tuple(args.models) != ("s2mnet", "umamba"):
        raise SystemExit(
            "The full paired campaign requires --models s2mnet umamba."
        )
    if not args.prepare_only and tuple(args.seeds) != SEEDS:
        seeds = " ".join(map(str, SEEDS))
        raise SystemExit(f"The full paired campaign requires --seeds {seeds}.")
    campaign = prepare_campaign(args)
    if args.prepare_only:
        print(json.dumps(campaign, indent=2))
        return 0
    project_root = Path(__file__).resolve().parents[2]
    interpreters = {
        "s2mnet": project_root / ".venv" / "bin" / "python",
        "umamba": project_root / ".venv-umamba" / "bin" / "python",
    }
    environment = {
        **os.environ,
        "PYTHONPATH": str(project_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Train all six declared runs before unsealing the held-out test.
    for phase in ("train", "evaluate"):
        for row in campaign["run_matrix"]:
            command = [
                str(interpreters[row["model"]]),
                "-m",
                "repro.experiments.claim3_train",
                "--dataset-root",
                str(args.dataset_root),
                "--output-root",
                str(args.output_root),
                "--worker-model",
                row["model"],
                "--worker-seed",
                str(row["seed"]),
                "--worker-phase",
                phase,
            ]
            subprocess.run(command, check=True, cwd=project_root, env=environment)
    summary = summarize_campaign(args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
