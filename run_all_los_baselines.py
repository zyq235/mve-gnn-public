"""Run the six-fold LOS baseline experiments serially.

Each model's existing entry point trains on development folds and evaluates
the fixed test cohort.  This is not a checkpoint-only inference command.
Completed outputs are skipped by default; use --overwrite for a full rerun.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_ENTRY = ROOT / "run_los_main_table.py"
ISTRATS_ENTRY = ROOT / "istrats" / "train_strats_los.py"
GCT_GRAM_ENTRY = ROOT / "run_gct_gram_los.py"
DATASET_DIRS = {"mimic3": "mimic3", "mimic4": "mimic-iv", "eicu": "eicu"}
MODELS = (
    "gct", "gram", "lr", "svm", "xgboost", "gru", "grud", "tcn", "sand", "istrats",
    "mtgnn", "time2graphplus", "raindrop", "mstdgnn",
    "mstdgnn_multiscale",
)
CLASSICAL = {"lr", "svm", "xgboost", "time2graphplus"}
SUMMARY_METRICS = {
    "linear_kappa", "rps", "mae_days", "macro_f1", "accuracy",
    "within_1_bin_accuracy",
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_models(values: list[str], parser: argparse.ArgumentParser) -> list[str]:
    models = [part.strip().lower() for value in values for part in value.split(",")]
    if not models or any(model not in MODELS for model in models):
        parser.error(f"--models must contain names from: {', '.join(MODELS)}")
    if len(set(models)) != len(models):
        parser.error("--models contains duplicate names")
    return models


def istrats_device(device: str) -> tuple[int, bool]:
    if device == "cpu":
        return 0, True
    if device in ("auto", "cuda"):
        return 0, False
    if device.startswith("cuda:") and device[5:].isdecimal():
        return int(device[5:]), False
    raise ValueError("--device must be auto, cpu, cuda, or cuda:<GPU index>")


def time2graph_child_device(device: str, environment: dict[str, str]) -> str:
    if not device.startswith("cuda:") or not device[5:].isdecimal():
        return device
    index = int(device[5:])
    if index == 0:
        return device
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        token = str(index)
    else:
        devices = [part.strip() for part in visible.split(",")]
        if index >= len(devices) or not devices[index]:
            raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        token = devices[index]
    environment["CUDA_VISIBLE_DEVICES"] = token
    return "cuda:0"


def command_for_model(model: str, args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    if model == "istrats":
        index, force_cpu = istrats_device(args.device)
        if force_cpu:
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        command = [
            sys.executable, "-u", str(ISTRATS_ENTRY),
            "--dataset", args.dataset, "--device", str(index),
            "--num-folds", str(args.n_folds),
            "--window-size", str(args.window_size),
            "--train-batch-size", str(args.batch_size or 16),
        ]
        if args.data_root:
            command += ["--data-root", str(Path(args.data_root) / DATASET_DIRS[args.dataset])]
        if args.istrats_pretrained_checkpoint:
            command += ["--pretrained-checkpoint", args.istrats_pretrained_checkpoint]
        if args.no_istrats_pretrained:
            command.append("--no-pretrained")
    else:
        child_device = (time2graph_child_device(args.device, environment)
                        if model == "time2graphplus" else args.device)
        command = [
            sys.executable, "-u", str(GCT_GRAM_ENTRY if model in {"gct", "gram"} else MAIN_ENTRY),
            "--model", model, "--dataset", args.dataset,
            "--device", child_device, "--n-folds", str(args.n_folds),
            "--window-size", str(args.window_size),
        ]
        if model in {"gct", "gram"}:
            command += ["--repeats", str(args.gct_gram_repeats)]
        if model == "time2graphplus":
            command += ["--time2graph-workers", str(args.time2graph_workers)]
        if args.data_root:
            command += ["--data-root", args.data_root]
        if args.batch_size is not None and model not in CLASSICAL:
            command += ["--batch-size", str(args.batch_size)]
    command += [
        "--landmark-hours", str(args.landmark_hours),
        "--eval-batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--results-root", str(project_path(args.results_root)),
        "--checkpoint-root", str(project_path(args.checkpoint_root)),
    ]
    if args.max_folds is not None:
        command += ["--max-folds", str(args.max_folds)]
    if args.max_train_steps is not None:
        command += ["--max-train-steps", str(args.max_train_steps)]
    if args.smoke_samples_per_class is not None:
        command += ["--smoke-samples-per-class", str(args.smoke_samples_per_class)]
    if args.epochs is not None and model not in CLASSICAL:
        command += ["--epochs", str(args.epochs)]
    if args.model_config_dir and model not in {"istrats", "gct", "gram"}:
        config = Path(args.model_config_dir) / f"{model}.json"
        if config.is_file():
            command += ["--model-config", str(config)]
    for option, value in (
        ("--pkl-path", args.pkl_path),
        ("--statistics-path", args.statistics_path),
        ("--los-labels-path", args.los_labels_path),
    ):
        if value is not None:
            command += [option, value]
    return command, environment


def output_paths(model: str, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    prefix = f"{model}_{args.landmark_hours}h"
    output_dir = project_path(args.results_root) / args.dataset / "los"
    return (
        output_dir / f"{prefix}_fold_metrics.csv",
        output_dir / f"{prefix}_summary.csv",
        output_dir / f"{prefix}_test_predictions.csv",
    )


def checkpoint_directory(model: str, args: argparse.Namespace) -> Path:
    root = project_path(args.checkpoint_root)
    if model == "istrats":
        return root / "istrats" / args.dataset
    return root / args.dataset / model


def fold_metric_rows(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [int(row["fold"]) for row in csv.DictReader(handle)]


def predictions_complete(path: Path, expected_folds: set[int]) -> bool:
    """Every fold must cover the same held-out stays exactly once."""

    by_fold = {fold: set() for fold in expected_folds}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fold, stay = int(row["fold"]), int(row["ts_id"])
            if fold not in by_fold or stay in by_fold[fold]:
                return False
            by_fold[fold].add(stay)
    reference = by_fold[min(expected_folds)]
    return bool(reference) and all(stays == reference for stays in by_fold.values())


def outputs_complete(model: str, args: argparse.Namespace) -> bool:
    fold_path, summary_path, predictions_path = output_paths(model, args)
    if not all(path.is_file() and path.stat().st_size > 0 for path in (
        fold_path, summary_path, predictions_path,
    )):
        return False
    expected = set(range(1, (args.max_folds or args.n_folds) + 1))
    try:
        if sorted(fold_metric_rows(fold_path)) != sorted(expected) or not predictions_complete(predictions_path, expected):
            return False
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return (
            len(rows) == len(SUMMARY_METRICS)
            and {row["metric"] for row in rows} == SUMMARY_METRICS
            and all(int(row["n_folds"]) == len(expected) for row in rows)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def has_existing_artifacts(model: str, args: argparse.Namespace) -> bool:
    if any(path.exists() for path in output_paths(model, args)):
        return True
    checkpoint_dir = checkpoint_directory(model, args)
    return checkpoint_dir.is_dir() and any(path.is_file() for path in checkpoint_dir.rglob("*"))


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("model", "status", "exit_code", "seconds", "log_path", "detail")
        )
        writer.writeheader()
        writer.writerows(rows)


def run_models(args: argparse.Namespace, runner=subprocess.run) -> int:
    if args.dry_run:
        for model in args.models:
            command, _ = command_for_model(model, args)
            print(f"[{model}] {' '.join(command)}", flush=True)
        return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_dir = project_path(args.results_root) / args.dataset / "los" / "baseline_runs" / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    manifest = log_dir / "run_manifest.csv"
    rows = []
    for index, model in enumerate(args.models, start=1):
        print(f"[{index}/{len(args.models)}] {model}", flush=True)
        log_path = log_dir / f"{model}.log"
        row = {"model": model, "status": "", "exit_code": "", "seconds": "",
               "log_path": "", "detail": ""}
        if outputs_complete(model, args) and not args.overwrite:
            row["status"] = "skipped_complete"
            row["detail"] = "Six-fold result CSVs already exist"
            print(f"  skipped: {row['detail']}", flush=True)
        elif has_existing_artifacts(model, args) and not args.overwrite and model != "time2graphplus":
            row["status"] = "blocked_partial"
            row["detail"] = "Existing partial outputs/checkpoints; use --overwrite to rerun"
            print(f"  blocked: {row['detail']}", flush=True)
        else:
            command, environment = command_for_model(model, args)
            started = time.monotonic()
            try:
                with log_path.open("w", encoding="utf-8") as stream:
                    result = runner(
                        command, cwd=ROOT, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, check=False,
                    )
                row["exit_code"] = result.returncode
                row["log_path"] = str(log_path.resolve())
                row["status"] = (
                    "completed" if result.returncode == 0 and outputs_complete(model, args)
                    else "failed"
                )
                if row["status"] == "failed":
                    row["detail"] = (f"exit code {result.returncode}" if result.returncode
                                     else "Process exited successfully but six-fold CSVs are incomplete")
            except (OSError, ValueError) as exc:
                row["status"] = "failed"
                row["detail"] = str(exc)
            row["seconds"] = round(time.monotonic() - started, 1)
            print(f"  {row['status']}: {row['detail'] or 'OK'}; log: {log_path}", flush=True)
        rows.append(row)
        write_manifest(manifest, rows)
    print(f"Run manifest: {manifest}", flush=True)
    return 1 if any(row["status"] in {"failed", "blocked_partial"} for row in rows) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_DIRS))
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help="Model names in execution order; default: every LOS baseline")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, help="Override neural-model epochs")
    parser.add_argument("--gct-gram-repeats", type=int, default=5)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--smoke-samples-per-class", type=int)
    parser.add_argument("--model-config-dir")
    parser.add_argument("--batch-size", type=int, help="Override neural-model training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time2graph-workers", type=int, default=1,
                        help="Time2Graph+ shapelet workers; default 1 limits peak memory")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--data-root", help="Parent of mimic3/mimic-iv/eicu data directories")
    parser.add_argument("--pkl-path")
    parser.add_argument("--statistics-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--istrats-pretrained-checkpoint")
    parser.add_argument("--no-istrats-pretrained", action="store_true")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--checkpoint-root", default="checkpoints/los")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow full reruns to overwrite existing results/checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.models = parse_models(args.models, parser)
    try:
        istrats_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    if args.n_folds < 2 or not 1 <= args.window_size < args.n_folds:
        parser.error("Require n-folds >= 2 and 1 <= window-size < n-folds")
    if args.max_folds is not None and not 1 <= args.max_folds <= args.n_folds:
        parser.error("--max-folds must be between 1 and --n-folds")
    if args.landmark_hours != 24:
        parser.error("Current baseline inputs support only the 24h landmark")
    for name in ("epochs", "batch_size", "eval_batch_size", "max_train_steps", "smoke_samples_per_class", "gct_gram_repeats", "time2graph_workers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return run_models(args)


if __name__ == "__main__":
    raise SystemExit(main())
