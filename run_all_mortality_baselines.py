"""Train mortality baselines on the finetune_kfold.py six-fold protocol.

The original pickle train and validation stay IDs are concatenated in their
saved order, split into six consecutive chunks, and used in cyclic 3-train /
1-validation windows.  Every fold evaluates the same untouched pickle test
set.  This intentionally matches finetune_kfold.py's *stay-level* folds; it
does not substitute the patient-shuffled folds used by some older baselines.
Each baseline is fitted once per fold (not repeat-5 ensembling).

Edit BASELINE_MODELS to select models and their launch priority. Each model
runs in a separate process. With --gpus, GPU models run concurrently on
different cards while classical models share one CPU slot. MVE-GNN is not a
baseline and is not included.

For an interrupted Time2Graph+ run, pass --only time2graphplus and
--resume-run-dir with that run's existing results/<dataset>/mortality/<run-id>
directory. Completed per-fold Time2Graph+ stages are reused there.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FOLD_COUNT = 6
TRAIN_CHUNKS = 3

# This is the authoritative selection and execution order.  These names use
# the existing three-dataset adapters in run_patient_bootstrap.py.
BASELINE_MODELS = [
    "sapsii",
    "lr",
    "svm",
    "xgboost",
    "time2graphplus",
    "gru",
    "grud",
    "sand",
    "tcn",
    "istrats",
    "mtgnn",
    "raindrop",
    "mstdgnn",
    "mstdgnn_multiscale",
    "gct",
    "gram",
]
CLASSICAL_MODELS = frozenset(
    {"sapsii", "lr", "svm", "xgboost", "time2graphplus"}
)
CPU_MODELS = frozenset({"sapsii", "lr", "svm", "xgboost"})
GRAPH_MODELS = frozenset({"gct", "gram"})
METRICS = ("auc", "prauc", "pr09")


def finetune_folds(ids, n_splits: int = FOLD_COUNT, window_size: int = TRAIN_CHUNKS):
    """Mirror finetune_kfold.get_sliding_window_splits without importing it."""

    ids = tuple(map(int, ids))
    if n_splits < 2 or not 1 <= window_size < n_splits:
        raise ValueError("Invalid fold count or training window")
    if len(ids) < n_splits or len(set(ids)) != len(ids):
        raise ValueError("Development IDs must be unique and cover every chunk")
    quotient, remainder = divmod(len(ids), n_splits)
    chunks = []
    offset = 0
    for index in range(n_splits):
        size = quotient + (index < remainder)
        chunks.append(ids[offset:offset + size])
        offset += size
    folds = []
    for index in range(n_splits):
        train = tuple(
            stay_id
            for shift in range(window_size)
            for stay_id in chunks[(index + shift) % n_splits]
        )
        validation = chunks[(index + window_size) % n_splits]
        folds.append((train, validation))
    return tuple(folds)


def normalize_device(value: str) -> str:
    value = value.strip().lower()
    if value.isdecimal():
        return f"cuda:{int(value)}"
    if value == "cuda":
        return "cuda:0"
    if value in {"auto", "cpu"}:
        return value
    if value.startswith("cuda:") and value[5:].isdecimal():
        return f"cuda:{int(value[5:])}"
    raise ValueError("--device must be auto, cpu, cuda, cuda:N, or a GPU index")


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError("--gpus must be a comma-separated list such as 0,1,3")
    gpu_ids = tuple(int(part) for part in parts)
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpus must not contain duplicate GPU indices")
    return gpu_ids


def visible_gpu_token(index: int, environment: dict[str, str]) -> str:
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(index)
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if index >= len(devices):
        raise ValueError(
            f"GPU index {index} is outside CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return devices[index]


def selected_models(only: list[str] | None) -> list[str]:
    if len(BASELINE_MODELS) != len(set(BASELINE_MODELS)):
        raise ValueError("BASELINE_MODELS contains duplicate names")
    requested = set(only) if only else set(BASELINE_MODELS)
    unknown = requested - set(BASELINE_MODELS)
    if unknown:
        raise ValueError(f"Models absent from BASELINE_MODELS: {sorted(unknown)}")
    return [model for model in BASELINE_MODELS if model in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=("mimic3", "mimic4", "eicu"))
    parser.add_argument("--device", default="auto", help="GPU index, cuda:N, cpu, or auto")
    parser.add_argument("--gpus", "--gpu-ids", "--gpu_ids", dest="gpu_ids",
                        help="GPU indices for parallel neural models, e.g. 0,1,3; "
                             "indices refer to visible GPUs when CUDA_VISIBLE_DEVICES is set")
    parser.add_argument("--max-task-num", "--max_task_num", type=int, default=None,
                        help="Maximum concurrent model processes, CPU and GPU combined; "
                             "defaults to GPU count + 1 with --gpus, otherwise 1")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints",
                        help="Existing pretrained checkpoints, used by iStraTS")
    parser.add_argument("--strats-pretrained-checkpoint", type=Path,
                        help="Override the iStraTS pretrained checkpoint path")
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results",
                        help="Base directory; outputs go to {results-root}/{dataset}/mortality/{run-id}")
    parser.add_argument("--resume-run-dir", type=Path,
                        help="Resume Time2Graph+ in an existing mortality run directory; use with --only time2graphplus")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Optional override for every neural baseline")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=FOLD_COUNT,
                        help="Use 1 for a smoke test; the full experiment uses 6")
    parser.add_argument("--only", nargs="+", choices=BASELINE_MODELS,
                        help="Optional subset; execution still follows BASELINE_MODELS")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print model commands without training")
    parser.add_argument("--single-model", choices=BASELINE_MODELS,
                        help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.device = normalize_device(args.device)
        args.gpu_ids = (parse_gpu_ids(args.gpu_ids)
                        if args.gpu_ids is not None else None)
        args.models = selected_models(args.only)
        if args.gpu_ids is not None:
            for gpu_id in args.gpu_ids:
                visible_gpu_token(gpu_id, os.environ)
    except ValueError as error:
        parser.error(str(error))
    if args.max_task_num is None:
        args.max_task_num = len(args.gpu_ids) + 1 if args.gpu_ids else 1
    if args.max_task_num < 1:
        parser.error("--max-task-num must be positive")
    if args.gpu_ids is None and args.max_task_num > 1:
        parser.error("--max-task-num > 1 requires --gpus")
    for name in ("epochs", "batch_size", "eval_batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or not 1 <= args.max_folds <= FOLD_COUNT:
        parser.error("--num-workers must be nonnegative and --max-folds in [1, 6]")
    if args.single_model and args.run_dir is None:
        parser.error("--single-model requires --run-dir")
    if args.resume_run_dir is not None:
        if args.models != ["time2graphplus"]:
            parser.error("--resume-run-dir requires --only time2graphplus")
        args.resume_run_dir = args.resume_run_dir.expanduser().resolve()
        if not args.resume_run_dir.is_dir():
            parser.error(f"Resume directory does not exist: {args.resume_run_dir}")
    for name in ("data_root", "checkpoint_root", "results_root", "strats_pretrained_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            path = value.expanduser()
            setattr(args, name, path if path.is_absolute() else ROOT / path)
    if args.resume_run_dir is not None:
        expected_root = (args.results_root / args.dataset / "mortality").resolve()
        if (args.resume_run_dir == expected_root
                or not args.resume_run_dir.is_relative_to(expected_root)):
            parser.error(f"--resume-run-dir must be inside {expected_root}")
    if args.run_dir is not None:
        args.run_dir = args.run_dir.expanduser().resolve()
    return args


def command_for_model(model: str, args: argparse.Namespace, run_dir: Path,
                      *, device: str | None = None) -> list[str]:
    # Time2Graph+ calls .cuda() without a device index. Its subprocess sees
    # only the selected physical GPU, so cuda:0 is correct inside that process.
    child_device = device or args.device
    if model in CPU_MODELS:
        child_device = "cpu"
    elif model == "time2graphplus" and child_device.startswith("cuda:") and device is None:
        child_device = "cuda:0"
    command = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--dataset", args.dataset, "--device", child_device,
        "--data-root", str(args.data_root),
        "--checkpoint-root", str(args.checkpoint_root),
        "--results-root", str(args.results_root),
        "--seed", str(args.seed), "--num-workers", str(args.num_workers),
        "--max-folds", str(args.max_folds),
        "--single-model", model, "--run-dir", str(run_dir),
    ]
    if model == "istrats" and args.strats_pretrained_checkpoint is not None:
        command.extend(("--strats-pretrained-checkpoint", str(args.strats_pretrained_checkpoint)))
    if model not in CLASSICAL_MODELS:
        for option, value in (
            ("--epochs", args.epochs),
            ("--batch-size", args.batch_size),
            ("--eval-batch-size", args.eval_batch_size),
        ):
            if value is not None:
                command.extend((option, str(value)))
    return command


def environment_for_model(model: str, args: argparse.Namespace,
                          *, gpu_id: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if model in CPU_MODELS:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_gpu_token(gpu_id, environment)
    elif model == "time2graphplus" and args.device.startswith("cuda:"):
        index = int(args.device.split(":", 1)[1])
        environment["CUDA_VISIBLE_DEVICES"] = visible_gpu_token(index, environment)
    return environment


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "status", "exit_code", "seconds", "log", "summary"),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_model_summary(path: Path, rows: list[dict]) -> None:
    fields = ["model", "status", "folds"] + [
        f"test_{metric}_{statistic}"
        for metric in METRICS for statistic in ("mean", "std")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in rows:
            item = {"model": run["model"], "status": run["status"], "folds": ""}
            if run["status"] == "completed":
                with Path(run["summary"]).open("r", encoding="utf-8") as source:
                    summary = json.load(source)
                item["folds"] = summary["n_folds_completed"]
                item.update({
                    f"test_{metric}_{statistic}": summary["test_summary"][metric][statistic]
                    for metric in METRICS for statistic in ("mean", "std")
                })
            writer.writerow(item)


def prepare_run_directory(args: argparse.Namespace) -> tuple[Path, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    if args.resume_run_dir is None:
        run_dir = args.results_root / args.dataset / "mortality" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log_dir = run_dir / "logs"
    else:
        run_dir = args.resume_run_dir
        log_dir = run_dir / "logs" / f"resume_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, log_dir


def run_all(args: argparse.Namespace, runner=subprocess.run,
            popen_factory=subprocess.Popen, poll_interval: float = 0.5) -> int:
    if args.dry_run:
        preview_dir = (args.resume_run_dir or
                       args.results_root / args.dataset / "mortality" / "<new-run-id>")
        gpu_count = 0
        for model in args.models:
            gpu_id = None
            if model not in CPU_MODELS and args.gpu_ids is not None:
                gpu_id = args.gpu_ids[gpu_count % len(args.gpu_ids)]
                gpu_count += 1
            environment = environment_for_model(model, args, gpu_id=gpu_id)
            child_device = "cuda:0" if gpu_id is not None else None
            visibility = (
                f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']!r} "
                if gpu_id is not None or model in CPU_MODELS
                or model == "time2graphplus" and args.device.startswith("cuda:")
                else ""
            )
            command = command_for_model(model, args, preview_dir, device=child_device)
            print(f"[{model}] {visibility}{subprocess.list2cmdline(command)}")
        return 0

    if args.gpu_ids is not None:
        return run_all_parallel(args, popen_factory=popen_factory,
                                poll_interval=poll_interval)

    run_dir, log_dir = prepare_run_directory(args)
    manifest_path = run_dir / "run_manifest.csv"
    rows: list[dict] = []
    for index, model in enumerate(args.models, start=1):
        log_path = log_dir / f"{model}.log"
        summary_path = run_dir / model / "summary.json"
        print(f"[{index}/{len(args.models)}] {model}; log={log_path}", flush=True)
        started = time.monotonic()
        try:
            command = command_for_model(model, args, run_dir)
            environment = environment_for_model(model, args)
            with log_path.open("w", encoding="utf-8") as stream:
                result = runner(command, cwd=ROOT, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
            exit_code = int(result.returncode)
            status = "completed" if exit_code == 0 and summary_path.is_file() else "failed"
        except (OSError, ValueError) as error:
            exit_code = -1
            status = f"failed: {error}"
        rows.append({
            "model": model, "status": status, "exit_code": exit_code,
            "seconds": round(time.monotonic() - started, 1),
            "log": str(log_path),
            "summary": str(summary_path) if summary_path.is_file() else "",
        })
        write_manifest(manifest_path, rows)
        write_model_summary(run_dir / "model_summary.csv", rows)
        print(f"  {status} ({rows[-1]['seconds']}s)", flush=True)
    print(f"Run manifest: {manifest_path}", flush=True)
    return int(any(row["status"] != "completed" for row in rows))


def run_all_parallel(args: argparse.Namespace, *, popen_factory=subprocess.Popen,
                     poll_interval: float = 0.5) -> int:
    """Use one CPU slot and one slot per selected GPU, with a global task cap."""

    run_dir, log_dir = prepare_run_directory(args)
    manifest_path = run_dir / "run_manifest.csv"
    summary_csv = run_dir / "model_summary.csv"
    pending = list(args.models)
    running: dict[str, dict] = {}
    records: dict[str, dict] = {}

    def persist() -> None:
        ordered = [records[model] for model in args.models if model in records]
        write_manifest(manifest_path, ordered)
        write_model_summary(summary_csv, ordered)

    try:
        while pending or running:
            while pending and len(running) < args.max_task_num:
                cpu_busy = any(job["gpu_id"] is None for job in running.values())
                used_gpus = {job["gpu_id"] for job in running.values()
                             if job["gpu_id"] is not None}
                available_gpus = [gpu_id for gpu_id in args.gpu_ids
                                  if gpu_id not in used_gpus]
                choice = None
                for index, model in enumerate(pending):
                    if model in CPU_MODELS and not cpu_busy:
                        choice = (index, None)
                        break
                    if model not in CPU_MODELS and available_gpus:
                        choice = (index, available_gpus[0])
                        break
                if choice is None:
                    break

                index, gpu_id = choice
                model = pending.pop(index)
                log_path = log_dir / f"{model}.log"
                summary_path = run_dir / model / "summary.json"
                started = time.monotonic()
                assignment = "CPU" if gpu_id is None else f"GPU {gpu_id}"
                print(f"[{args.models.index(model) + 1}/{len(args.models)}] "
                      f"start {model} on {assignment}; log={log_path}", flush=True)
                try:
                    command = command_for_model(
                        model, args, run_dir,
                        device="cuda:0" if gpu_id is not None else None,
                    )
                    environment = environment_for_model(model, args, gpu_id=gpu_id)
                    with log_path.open("w", encoding="utf-8") as stream:
                        process = popen_factory(
                            command, cwd=ROOT, env=environment, stdout=stream,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        )
                except (OSError, ValueError) as error:
                    records[model] = {
                        "model": model, "status": f"failed: {error}",
                        "exit_code": -1,
                        "seconds": round(time.monotonic() - started, 1),
                        "log": str(log_path), "summary": "",
                    }
                    print(f"  {model}: {records[model]['status']}", flush=True)
                else:
                    running[model] = {
                        "process": process, "gpu_id": gpu_id,
                        "started": started, "summary_path": summary_path,
                        "log_path": log_path,
                    }
                    records[model] = {
                        "model": model, "status": "running", "exit_code": "",
                        "seconds": "", "log": str(log_path), "summary": "",
                    }
                persist()

            finished = []
            for model, job in running.items():
                exit_code = job["process"].poll()
                if exit_code is None:
                    continue
                summary_path = job["summary_path"]
                status = ("completed" if int(exit_code) == 0 and summary_path.is_file()
                          else "failed")
                records[model] = {
                    "model": model, "status": status,
                    "exit_code": int(exit_code),
                    "seconds": round(time.monotonic() - job["started"], 1),
                    "log": str(job["log_path"]),
                    "summary": str(summary_path) if summary_path.is_file() else "",
                }
                finished.append(model)
                print(f"  {model}: {status} ({records[model]['seconds']}s)",
                      flush=True)
            for model in finished:
                del running[model]
            if finished:
                persist()
            elif running:
                time.sleep(poll_interval)
            elif pending:
                raise RuntimeError("Pending models cannot use the available CPU/GPU slots")
    except BaseException:
        for job in running.values():
            try:
                job["process"].terminate()
            except OSError:
                pass
        raise

    print(f"Run manifest: {manifest_path}", flush=True)
    return int(any(record["status"] != "completed" for record in records.values()))


def _validate_predictions(frame, expected_ids, label_by_stay) -> None:
    expected = set(map(int, expected_ids))
    received = frame["stay_id"].astype(int).tolist()
    if len(received) != len(expected) or set(received) != expected:
        raise ValueError("Predictions do not cover the requested stays exactly once")
    for row in frame.itertuples(index=False):
        if int(row.label) != int(label_by_stay[int(row.stay_id)]):
            raise ValueError(f"Wrong label for stay {row.stay_id}")


def _metrics(frame):
    from patient_bootstrap import metric_values

    return metric_values(frame["label"].to_numpy(), frame["probability"].to_numpy())


def _train_neural(model_name, paths, protocol, fold_dir, args, seed):
    import numpy as np
    import torch

    from ehr_baseline_utils import cpu_state_dict, resolve_device, set_seed
    from run_patient_bootstrap import build_neural_adapter, make_loader, predict_model

    set_seed(seed)
    device = resolve_device(args.device)
    adapter = build_neural_adapter(
        model_name, paths, {}, args, protocol, fold_dir, device
    )
    train_set = adapter.dataset(protocol.train_ids)
    validation_set = adapter.dataset(protocol.validation_ids)
    demo_dim = getattr(train_set, "demo_dim", 2)
    if demo_dim != getattr(validation_set, "demo_dim", 2):
        raise ValueError("Demographic dimensions differ between training and validation")
    train_loader = make_loader(
        adapter, train_set, args.batch_size or adapter.default_batch_size,
        args.num_workers, device, shuffle=True,
    )
    validation_loader = make_loader(
        adapter, validation_set,
        args.eval_batch_size or adapter.default_eval_batch_size,
        args.num_workers, device, shuffle=False,
    )
    model = adapter.build_model(device, demo_dim)
    epochs = args.epochs or adapter.default_epochs
    learning_rate = adapter.default_learning_rate
    weight_decay = getattr(adapter, "default_weight_decay", 0.0)
    gradient_clip = getattr(adapter, "default_gradient_clip", 0.5)
    optimizer_kind = getattr(adapter, "optimizer_kind", "adamw")
    optimizer_type = torch.optim.Adam if optimizer_kind == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    scheduler_kind = getattr(adapter, "scheduler_kind", "cosine")
    if scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )
    else:
        scheduler = None
    best_score = -float("inf")
    best_validation = None
    best_epoch = None
    checkpoint = fold_dir / "best_validation.pt"
    history = []
    for epoch in range(1, epochs + 1):
        loss = adapter.train_epoch(model, train_loader, optimizer, device, gradient_clip)
        validation_frame = predict_model(
            adapter, model, validation_loader, device, protocol.validation_ids
        )
        _validate_predictions(
            validation_frame, protocol.validation_ids, protocol.label_by_stay
        )
        validation = _metrics(validation_frame)
        score = validation["auc"] + validation["prauc"]
        history.append({"epoch": epoch, "train_loss": loss, **validation})
        if score > best_score:
            best_score = score
            best_validation = validation
            best_epoch = epoch
            torch.save({
                "model_state": cpu_state_dict(model), "model": model_name,
                "dataset": args.dataset, "epoch": epoch, "seed": seed,
                "validation": validation,
            }, checkpoint)
        if scheduler_kind == "cosine":
            scheduler.step()
        elif scheduler_kind == "plateau":
            probabilities = np.clip(validation_frame["probability"].to_numpy(),
                                    1e-7, 1 - 1e-7)
            labels = validation_frame["label"].to_numpy()
            log_loss = -np.mean(
                labels * np.log(probabilities)
                + (1 - labels) * np.log(1 - probabilities)
            )
            scheduler.step(float(log_loss))
        print(
            f"  epoch={epoch}/{epochs} loss={loss:.4f} "
            f"val_auc={validation['auc']:.4f} val_prauc={validation['prauc']:.4f}",
            flush=True,
        )
    from ehr_baseline_utils import save_json

    save_json(fold_dir / "history.json", history)
    if best_validation is None:
        raise RuntimeError("No validation checkpoint was selected")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"], strict=True)
    model.to(device)
    test_set = adapter.dataset(protocol.test_ids)
    if demo_dim != getattr(test_set, "demo_dim", 2):
        raise ValueError("Demographic dimensions differ in the test split")
    test_loader = make_loader(
        adapter, test_set, args.eval_batch_size or adapter.default_eval_batch_size,
        args.num_workers, device, shuffle=False,
    )
    prediction = predict_model(adapter, model, test_loader, device, protocol.test_ids)
    return prediction, {"validation": best_validation, "selected_epoch": best_epoch,
                        "checkpoint": str(checkpoint), "epochs": epochs}


def run_single_model(args: argparse.Namespace) -> None:
    from dataset.ehr_cached_graph_dataset import (
        load_protocol_splits, resolve_ehr_dataset_paths,
    )
    from ehr_baseline_utils import save_json

    model_name = args.single_model
    assert model_name is not None and args.run_dir is not None
    paths = resolve_ehr_dataset_paths(
        args.dataset, args.data_root,
        require_graph_cache=model_name in GRAPH_MODELS,
        require_metadata=model_name != "sapsii",
    )
    protocol = load_protocol_splits(paths)
    folds = finetune_folds(protocol.development_ids)
    test_ids = set(protocol.test_ids)
    for fold_number, (train_ids, validation_ids) in enumerate(folds, start=1):
        if (set(train_ids) & set(validation_ids)
                or set(train_ids) & test_ids or set(validation_ids) & test_ids):
            raise ValueError(f"Stay-ID leakage in fold {fold_number}")
        for part, ids in (("train", train_ids), ("validation", validation_ids)):
            labels = {protocol.label_by_stay[stay] for stay in ids}
            if labels != {0, 1}:
                raise ValueError(f"Fold {fold_number} {part} needs both outcome classes")

    model_dir = args.run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=model_name == "time2graphplus")
    split_payload = {
        "development_ids": protocol.development_ids,
        "test_ids": protocol.test_ids,
        "folds": folds,
    }
    split_digest = hashlib.sha256(
        json.dumps(split_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rows = []
    for fold_index, (train_ids, validation_ids) in enumerate(folds[:args.max_folds]):
        fold_number = fold_index + 1
        fold_dir = model_dir / f"fold_{fold_number}"
        fold_dir.mkdir(exist_ok=model_name == "time2graphplus")
        fold_seed = args.seed + fold_index * 1000
        fold_protocol = replace(
            protocol, train_ids=train_ids, validation_ids=validation_ids, folds=()
        )
        shared_patients = (
            {protocol.patient_by_stay[stay] for stay in train_ids}
            & {protocol.patient_by_stay[stay] for stay in validation_ids}
        )
        if shared_patients:
            print(
                f"WARNING fold={fold_number}: {len(shared_patients)} patients "
                "span training and validation stays under the reference "
                "stay-level split",
                flush=True,
            )
        print(
            f"[{model_name}] fold={fold_number}/{FOLD_COUNT} "
            f"train={len(train_ids)} validation={len(validation_ids)} "
            f"test={len(protocol.test_ids)} seed={fold_seed}",
            flush=True,
        )
        if model_name in CLASSICAL_MODELS:
            from main_table_classical import fit_tabular_baseline
            from ehr_baseline_utils import resolve_device

            classical_args = argparse.Namespace(
                device=str(resolve_device(args.device)), dataset=args.dataset,
                data_root=args.data_root, seed=fold_seed, sapsii_csv=None,
            )
            prediction, details = fit_tabular_baseline(
                model_name, paths, fold_protocol, classical_args, {}, fold_dir
            )
            validation = details["validation_metrics"]
            selected_epoch = details.get("selected_epoch")
            artifacts = list(fold_dir.glob("selected_model.*"))
            if model_name == "time2graphplus":
                artifacts.append(fold_dir / "selected_xgb_head.json")
            checkpoint = str(artifacts[0]) if artifacts else None
        else:
            prediction, details = _train_neural(
                model_name, paths, fold_protocol, fold_dir, args, fold_seed
            )
            validation = details["validation"]
            selected_epoch = details["selected_epoch"]
            checkpoint = details["checkpoint"]
        _validate_predictions(prediction, protocol.test_ids, protocol.label_by_stay)
        test_metrics = _metrics(prediction)
        prediction.insert(0, "fold", fold_number)
        prediction.insert(1, "patient_id", prediction["stay_id"].map(
            lambda stay: str(protocol.patient_by_stay[int(stay)])
        ))
        prediction.to_csv(fold_dir / "test_predictions.csv", index=False)
        record = {
            "fold": fold_number, "seed": fold_seed,
            "train_stays": len(train_ids), "validation_stays": len(validation_ids),
            "test_stays": len(protocol.test_ids),
            "train_validation_patient_overlap": len(shared_patients),
            "selected_epoch": selected_epoch, "checkpoint": checkpoint,
            "validation": validation, "test": test_metrics,
        }
        save_json(fold_dir / "metrics.json", record)
        rows.append(record)
        with (model_dir / "fold_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = (
                "fold", "seed", "train_stays", "validation_stays",
                "test_stays", "train_validation_patient_overlap",
                "selected_epoch", "checkpoint", "val_auc", "val_prauc",
                "val_pr09", "test_auc", "test_prauc", "test_pr09",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow({
                    **{key: item[key] for key in fieldnames[:8]},
                    **{f"val_{metric}": item["validation"][metric]
                       for metric in METRICS},
                    **{f"test_{metric}": item["test"][metric]
                       for metric in METRICS},
                })
        print(
            f"  test_auc={test_metrics['auc']:.4f} "
            f"test_prauc={test_metrics['prauc']:.4f} "
            f"test_pr09={test_metrics['pr09']:.4f}",
            flush=True,
        )
    summary = {
        "dataset": args.dataset, "model": model_name,
        "fold_protocol": "finetune_kfold_stay_chunks_6x3",
        "split_digest": split_digest, "test_is_fixed": True,
        "n_folds_completed": len(rows), "seed": args.seed,
        "folds": rows,
        "test_summary": {
            metric: {
                "mean": statistics.mean(row["test"][metric] for row in rows),
                "std": statistics.pstdev(row["test"][metric] for row in rows),
            }
            for metric in METRICS
        },
    }
    save_json(model_dir / "summary.json", summary)
    print(f"[{model_name}] summary={model_dir / 'summary.json'}", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.single_model:
        run_single_model(args)
        return 0
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
