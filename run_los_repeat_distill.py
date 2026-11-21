"""Run or re-evaluate three LOS views, then fuse their test probabilities.

Each view runs in its own process on one physical GPU.  Stage 2 starts only
after all three Stage-1 processes succeed and every expected teacher
checkpoint exists.  When all distilled students already exist, Stage 3
re-evaluates them without repeating either training stage.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VIEWS = ("type", "time", "diff_pool")
REPEATS = 5


def parse_devices(values: list[str]) -> tuple[int, int, int]:
    tokens = [token.strip() for value in values for token in value.split(",")]
    if len(tokens) != len(VIEWS) or any(not token.isdecimal() for token in tokens):
        raise argparse.ArgumentTypeError(
            "--device requires three GPU indices in type,time,diff order, e.g. 0,1,2"
        )
    devices = tuple(int(token) for token in tokens)
    if len(set(devices)) != len(VIEWS):
        raise argparse.ArgumentTypeError("--device requires three distinct GPU indices")
    return devices


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def child_command(args: argparse.Namespace, stage: int, view: str) -> list[str]:
    """Use the same cohort protocol and checkpoint root in both stages."""

    script = (
        "finetune_los_kfold.py" if stage == 1
        else "finetune_los_kfold_ensemble_distillation.py"
    )
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / script),
        "--dataset", args.dataset,
        "--pool-mode", view,
        # CUDA_VISIBLE_DEVICES exposes just this view's physical GPU.
        "--device", "0",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--n-folds", str(args.n_folds),
        "--window-size", str(args.window_size),
        "--landmark-hours", str(args.landmark_hours),
        "--lambda-confidence", str(args.lambda_confidence),
        "--settings-root", str(project_path(args.settings_root)),
        "--results-root", str(project_path(args.results_root)),
    ]
    if stage == 1:
        command += [
            "--repeat-time", str(REPEATS),
            "--max-epoch", str(args.teacher_epochs),
            "--checkpoint-root", str(project_path(args.teacher_checkpoint_root)),
        ]
    elif stage in (2, 3):
        command += [
            "--teacher-repeats", str(REPEATS),
            "--student-lr", str(args.student_lr),
            "--distillation-weight", str(args.distillation_weight),
            "--temperature", str(args.temperature),
            "--checkpoint-root", str(project_path(args.distill_checkpoint_root)),
        ]
        if stage == 2:
            command += [
                "--max-epoch", str(args.distill_epochs),
                "--teacher-batch-size", str(args.teacher_batch_size),
                "--teacher-checkpoint-root", str(project_path(args.teacher_checkpoint_root)),
            ]
        else:
            command.append("--evaluate-only")
        if args.experiment_name is not None:
            command += ["--experiment-name", args.experiment_name]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    if args.max_folds is not None:
        command += ["--max-folds", str(args.max_folds)]
    for option, value in (
        ("--data-root", args.data_root),
        ("--pkl-path", args.pkl_path),
        ("--los-labels-path", args.los_labels_path),
        ("--pretrained-checkpoint", args.pretrained_checkpoint),
        ("--model-var2ind-path", args.model_var2ind_path),
        ("--pretrained-var2ind-path", args.pretrained_var2ind_path),
    ):
        if value is not None:
            command += [option, str(value)]
    return command


def check_gpus(devices: tuple[int, int, int]) -> None:
    """Prevent a missing GPU from silently turning a view into a CPU job."""

    for gpu in devices:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        check = subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1"],
            cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True,
        )
        if check.returncode != 0:
            raise RuntimeError(
                f"GPU {gpu} is unavailable to the selected Python environment. "
                "Check --device and CUDA_VISIBLE_DEVICES before starting training."
            )


def stop_processes(processes: dict[str, subprocess.Popen]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_stage(
    args: argparse.Namespace, stage: int, devices: tuple[int, int, int], log_dir: Path
) -> None:
    processes: dict[str, subprocess.Popen] = {}
    streams = {}
    try:
        for view, gpu in zip(VIEWS, devices):
            log_path = log_dir / f"stage{stage}_{view}.log"
            stream = log_path.open("w", encoding="utf-8")
            streams[view] = stream
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = child_command(args, stage, view)
            processes[view] = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
            print(
                f"Stage {stage}: {view} on GPU {gpu}, PID {processes[view].pid}; "
                f"log: {log_path}", flush=True,
            )

        pending = set(VIEWS)
        while pending:
            for view in VIEWS:
                if view not in pending:
                    continue
                exit_code = processes[view].poll()
                if exit_code is None:
                    continue
                pending.remove(view)
                streams[view].close()
                if exit_code != 0:
                    raise RuntimeError(
                        f"Stage {stage} {view} failed (exit {exit_code}); "
                        f"see {log_dir / f'stage{stage}_{view}.log'}"
                    )
                print(f"Stage {stage}: {view} completed", flush=True)
            if pending:
                time.sleep(1)
    except BaseException:
        stop_processes(processes)
        raise
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()


def verify_teachers(args: argparse.Namespace) -> None:
    root = project_path(args.teacher_checkpoint_root) / args.dataset
    fold_count = args.max_folds or args.n_folds
    missing = [
        root / view / f"fold_{fold}" / f"repeat_{repeat}" / "best.pt"
        for view in VIEWS
        for fold in range(1, fold_count + 1)
        for repeat in range(1, REPEATS + 1)
        if not (root / view / f"fold_{fold}" / f"repeat_{repeat}" / "best.pt").is_file()
    ]
    if missing:
        detail = "\n".join(map(str, missing[:10]))
        raise FileNotFoundError(
            f"Stage 1 is missing {len(missing)} teacher checkpoints; "
            f"Stage 2 will not start:\n{detail}"
        )


def view_locations(args: argparse.Namespace, view: str) -> tuple[Path, Path, str]:
    """Use the distiller's own tag and paths so resume matches its checkpoints."""

    from finetune_los_kfold_ensemble_distillation import (
        _run_tag, checkpoint_directory, result_directory,
    )

    view_args = argparse.Namespace(**vars(args))
    view_args.pool_mode = view
    view_args.checkpoint_root = str(project_path(args.distill_checkpoint_root))
    view_args.results_root = str(project_path(args.results_root))
    return checkpoint_directory(view_args), result_directory(view_args), _run_tag(view_args)


def all_distilled_students_exist(args: argparse.Namespace) -> bool:
    """A complete run has one selected student per view and fold."""

    fold_count = args.max_folds or args.n_folds
    for view in VIEWS:
        checkpoint_dir, _, _ = view_locations(args, view)
        if any(
            not (checkpoint_dir / f"fold_{fold}" / "best.pt").is_file()
            for fold in range(1, fold_count + 1)
        ):
            return False
    return True


def fuse_view_test_results(args: argparse.Namespace) -> Path:
    """Equal-weight soft voting of aligned student probabilities, per fold."""

    import numpy as np
    import pandas as pd
    import torch

    from ML.los_metrics import NUM_LOS_CLASSES, compute_los_metrics

    columns = [f"prob_{index}" for index in range(NUM_LOS_CLASSES)]
    identity = ["fold", "ts_id"]
    metadata = [
        "true_class", "remaining_los_days", "icu_death",
        "in_hospital_mortality",
    ]
    expected_folds = list(range(1, (args.max_folds or args.n_folds) + 1))
    by_view = {}
    checkpoint_dirs = {}
    tags = {}
    for view in VIEWS:
        checkpoint_dir, result_dir, tag = view_locations(args, view)
        source = result_dir / "test_predictions.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing {view} student predictions: {source}")
        frame = pd.read_csv(source)
        required = set(identity + metadata + columns + ["dataset", "pool_mode"])
        if not required.issubset(frame.columns):
            raise ValueError(f"Incomplete {view} student predictions: {source}")
        if frame.empty or frame.duplicated(identity).any():
            raise ValueError(f"Empty or duplicate fold/stay rows in {source}")
        if sorted(frame["fold"].unique().tolist()) != expected_folds:
            raise ValueError(f"Expected folds {expected_folds} in {source}")
        if not frame["dataset"].eq(args.dataset).all() or not frame["pool_mode"].eq(view).all():
            raise ValueError(f"Dataset or view mismatch in {source}")
        frame = frame.sort_values(identity).reset_index(drop=True)
        probs = frame[columns].to_numpy(dtype=np.float64)
        if (not np.isfinite(probs).all() or (probs < 0).any()
                or not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7)):
            raise ValueError(f"Invalid probabilities in {source}")
        by_view[view] = frame
        checkpoint_dirs[view] = checkpoint_dir
        tags[view] = tag

    reference = by_view[VIEWS[0]]
    for view in VIEWS[1:]:
        frame = by_view[view]
        if not frame[identity].equals(reference[identity]):
            raise ValueError(f"{view} fold/stay IDs differ from {VIEWS[0]}")
        for column in metadata:
            left, right = reference[column].to_numpy(), frame[column].to_numpy()
            equal = (np.allclose(left, right, rtol=0, atol=1e-6)
                     if column == "remaining_los_days" else np.array_equal(left, right))
            if not equal:
                raise ValueError(f"{view} {column} differs from {VIEWS[0]}")

    fused_probs = np.mean(
        np.stack([by_view[view][columns].to_numpy(dtype=np.float64) for view in VIEWS]),
        axis=0,
    )
    fold_rows, prediction_frames = [], []
    for fold in expected_folds:
        selected = reference["fold"].eq(fold).to_numpy()
        if not selected.any():
            raise ValueError(f"No test predictions for fold {fold}")
        representatives = None
        for view in VIEWS:
            checkpoint = checkpoint_dirs[view] / f"fold_{fold}" / "best.pt"
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            if "representative_days" not in payload:
                raise ValueError(f"Missing representative days in {checkpoint}")
            candidate = np.asarray(payload["representative_days"], dtype=np.float64)
            if candidate.shape != (NUM_LOS_CLASSES,) or not np.isfinite(candidate).all():
                raise ValueError(f"Invalid representative days in {checkpoint}")
            if representatives is None:
                representatives = candidate
            elif not np.allclose(candidate, representatives, rtol=0, atol=1e-6):
                raise ValueError(f"Fold {fold} representative days differ across views")
        probs = fused_probs[selected]
        labels = reference.loc[selected, "true_class"].to_numpy(dtype=np.int64)
        days = reference.loc[selected, "remaining_los_days"].to_numpy(dtype=np.float64)
        metrics = compute_los_metrics(labels, probs, days, representatives)
        fold_rows.append({
            "dataset": args.dataset, "pool_mode": "three_view_fusion",
            "landmark_hours": args.landmark_hours, "fold": fold,
            "fusion_method": "equal_probability_mean",
            **{f"test_{name}": value for name, value in metrics.items()},
        })
        predictions = reference.loc[selected, identity + metadata].copy()
        predictions.insert(0, "dataset", args.dataset)
        predictions.insert(1, "pool_mode", "three_view_fusion")
        predictions["pred_class"] = probs.argmax(axis=1)
        predictions["predicted_los_days"] = representatives[
            predictions["pred_class"].to_numpy(dtype=np.int64)
        ]
        for index, column in enumerate(columns):
            predictions[column] = probs[:, index]
        prediction_frames.append(predictions)

    result_dir = (
        project_path(args.results_root) / args.dataset / "los" / "distillation"
        / "three_view_fusion" / tags["diff_pool"]
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(result_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        result_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame([
        {
            "dataset": args.dataset, "pool_mode": "three_view_fusion",
            "metric": metric, "mean": fold_frame[f"test_{metric}"].mean(),
            "std": fold_frame[f"test_{metric}"].std(ddof=0),
            "n_folds": len(fold_frame),
        }
        for metric in (
            "linear_kappa", "rps", "mae_days", "macro_f1", "accuracy",
            "within_1_bin_accuracy",
        )
    ]).to_csv(result_dir / "summary.csv", index=False)
    return result_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mimic3", "mimic4", "eicu"))
    parser.add_argument(
        "--device", "--devices", nargs="+", required=True,
        help="Three distinct physical GPU indices for type,time,diff_pool, e.g. 0,1,2",
    )
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--distill-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--distillation-weight", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--student-lr", type=float, default=5e-5)
    parser.add_argument("--lambda-confidence", type=float, default=0.0044)
    parser.add_argument("--settings-root", default="settings")
    parser.add_argument("--data-root")
    parser.add_argument("--pkl-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--model-var2ind-path")
    parser.add_argument("--pretrained-var2ind-path")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--teacher-checkpoint-root", default="checkpoints/los")
    parser.add_argument("--distill-checkpoint-root", default="checkpoints/los_distilled")
    parser.add_argument("--experiment-name")
    parser.add_argument("--log-dir", help="Directory for six per-view process logs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without using GPUs")
    return parser


def run(args: argparse.Namespace) -> None:
    devices = parse_devices(args.device)
    if min(args.teacher_epochs, args.distill_epochs, args.batch_size,
           args.teacher_batch_size, args.n_folds, args.window_size) < 1:
        raise ValueError("Epochs, batch sizes, n-folds, and window-size must be positive")
    if args.num_workers < 0 or (args.max_folds is not None and
                                not 1 <= args.max_folds <= args.n_folds):
        raise ValueError("num-workers must be nonnegative and max-folds within n-folds")
    if not 0 <= args.distillation_weight <= 1 or args.temperature <= 0:
        raise ValueError("Invalid distillation weight or temperature")
    if args.student_lr <= 0:
        raise ValueError("student-lr must be positive")
    if args.lambda_confidence < 0:
        raise ValueError("lambda-confidence must be nonnegative")
    if args.experiment_name and ("/" in args.experiment_name or "\\" in args.experiment_name):
        raise ValueError("experiment-name must be a single directory name")

    if args.dry_run:
        for stage in (1, 2):
            for view, gpu in zip(VIEWS, devices):
                print(f"Stage {stage} {view}: CUDA_VISIBLE_DEVICES={gpu}")
                print(" ".join(child_command(args, stage, view)))
        return

    check_gpus(devices)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_dir = (
        project_path(args.log_dir) if args.log_dir else
        project_path(args.results_root) / args.dataset / "los" / "pipeline_logs" / run_id
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset={args.dataset}; type,time,diff_pool GPUs={devices}", flush=True)
    if all_distilled_students_exist(args):
        print("All three views have distilled students for every fold; re-evaluating test only", flush=True)
        run_stage(args, 3, devices, log_dir)
    else:
        run_stage(args, 1, devices, log_dir)
        verify_teachers(args)
        print("All repeat-5 teachers verified; starting distillation", flush=True)
        run_stage(args, 2, devices, log_dir)
    fusion_dir = fuse_view_test_results(args)
    print(f"Three-view LOS fusion complete: {fusion_dir}; logs: {log_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
