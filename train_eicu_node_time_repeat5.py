"""Train the eICU node-time view with six folds and five repeats per fold.

The command is intentionally narrow: it always creates the 30 teacher
checkpoints expected by ``finetune_kfold_ensemble_distillation.py``:

    eicu_gat_ft_pool_node_time_fold{1..6}_turn{1..5}_auc/
        checkpoint_fold{fold}_turn{turn}_best.pt

Every graph node is a Transformer token because the selected model forward
mode is ``node_time``.  Use ``--dry-run`` to inspect all output paths without
loading the dataset or allocating a model.  ``--start-fold`` and
``--start-repeat`` resume at a fold/repeat boundary; the boundary repeat always
restarts at epoch 1.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import finetune_kfold as training
from logger import Logger


ROOT = Path(__file__).resolve().parent
DATASET = "eicu"
POOL_MODE = "node_time"
N_FOLDS = 6
N_REPEATS = 5


def resolve_path(value: Path) -> Path:
    path = value.expanduser()
    return path if path.is_absolute() else ROOT / path


def expected_checkpoints(output_root: Path) -> list[Path]:
    paths: list[Path] = []
    for fold_number in range(1, N_FOLDS + 1):
        for turn_number in range(1, N_REPEATS + 1):
            task_name = training.fine_tune_task_name(
                "eicu", POOL_MODE, fold_number, turn_number
            )
            paths.append(
                output_root
                / "checkpoints"
                / task_name
                / (
                    f"checkpoint_fold{fold_number}_turn{turn_number}_best.pt"
                )
            )
    return paths


def checkpoint_partitions(
    output_root: Path,
    start_fold: int,
    start_repeat: int,
) -> tuple[list[Path], list[Path]]:
    """Split the fixed protocol into retained and (re)trained checkpoints."""

    completed: list[Path] = []
    remaining: list[Path] = []
    for index, path in enumerate(expected_checkpoints(output_root)):
        fold_number = index // N_REPEATS + 1
        repeat_number = index % N_REPEATS + 1
        if (fold_number, repeat_number) < (start_fold, start_repeat):
            completed.append(path)
        else:
            remaining.append(path)
    return completed, remaining


def build_training_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=DATASET,
        pool_mode=POOL_MODE,
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=args.device,
        repeat_time=N_REPEATS,
        unfreeze_epoch=5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        n_folds=N_FOLDS,
        window_size=args.window_size,
        max_folds=None,
        start_fold=args.start_fold,
        start_repeat=args.start_repeat,
        cuda_debug=args.cuda_debug,
        cuda_sync=args.cuda_sync,
        max_node_time_tokens=args.max_node_time_tokens,
        data_root=resolve_path(args.data_root),
        settings_root=resolve_path(args.settings_root),
        checkpoint_root=resolve_path(args.checkpoint_root),
        output_root=resolve_path(args.output_root),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 6-fold x 5-repeat eICU node_time teachers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument(
        "--start-fold",
        "--start_fold",
        dest="start_fold",
        type=int,
        default=1,
        help="1-based fold at which training restarts.",
    )
    parser.add_argument(
        "--start-repeat",
        "--start_repeat",
        dest="start_repeat",
        type=int,
        default=1,
        help=(
            "1-based repeat at which the start fold restarts; later folds "
            "always start at repeat 1."
        ),
    )
    parser.add_argument(
        "--cuda-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist last-batch metadata and a JSON CUDA failure report.",
    )
    parser.add_argument(
        "--cuda-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize CUDA phases so the report identifies the failing phase.",
    )
    parser.add_argument(
        "--hard-exit-on-cuda-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After diagnostics are flushed, terminate immediately with exit "
            "code 70 when a CUDA/driver error reaches the launcher."
        ),
    )
    parser.add_argument(
        "--max-node-time-tokens",
        "--max_node_time_tokens",
        dest="max_node_time_tokens",
        type=int,
        default=0,
        help=(
            "Abort before GPU forward if a patient graph exceeds this many "
            "node_time tokens; 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        help="Override the default eICU pretrained encoder checkpoint.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--settings-root", type=Path, default=Path("settings"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow retraining existing checkpoints after the requested resume "
            "boundary. The boundary checkpoint itself may be overwritten "
            "without this flag."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed protocol and expected checkpoint paths only.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.device < 0:
        raise ValueError("--device must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 1 <= args.window_size < N_FOLDS:
        raise ValueError(f"--window-size must lie in [1, {N_FOLDS - 1}]")
    if not 1 <= args.start_fold <= N_FOLDS:
        raise ValueError(f"--start-fold must lie in [1, {N_FOLDS}]")
    if not 1 <= args.start_repeat <= N_REPEATS:
        raise ValueError(f"--start-repeat must lie in [1, {N_REPEATS}]")
    if args.max_node_time_tokens < 0:
        raise ValueError("--max-node-time-tokens cannot be negative")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    training_args = build_training_args(args)
    output_root = Path(training_args.output_root)
    planned = expected_checkpoints(output_root)
    completed, remaining = checkpoint_partitions(
        output_root,
        args.start_fold,
        args.start_repeat,
    )

    print(
        f"dataset={DATASET} pool_mode={POOL_MODE} folds={N_FOLDS} "
        f"repeats={N_REPEATS} total_checkpoints={len(planned)} "
        f"device=cuda:{args.device} start_fold={args.start_fold} "
        f"start_repeat={args.start_repeat}"
    )
    for path in planned:
        print(path)

    if args.dry_run:
        return

    missing_completed = [path for path in completed if not path.is_file()]
    if missing_completed:
        preview = "\n".join(str(path) for path in missing_completed[:10])
        print(
            f"WARNING: {len(missing_completed)} checkpoint(s) before the resume "
            "boundary are missing. They will remain missing and will not block "
            f"training. First paths:\n{preview}"
        )

    resume_requested = (args.start_fold, args.start_repeat) != (1, 1)
    boundary_checkpoint = remaining[0] if remaining else None
    conflicting = [
        path
        for path in remaining
        if path.is_file()
        and not (resume_requested and path == boundary_checkpoint)
    ]
    if conflicting and not args.overwrite:
        preview = "\n".join(str(path) for path in conflicting[:10])
        raise FileExistsError(
            f"{len(conflicting)} checkpoint(s) in the requested training range "
            f"already exist. Use --overwrite to retrain them. First paths:\n{preview}"
        )

    if boundary_checkpoint is not None and boundary_checkpoint.is_file():
        print(
            "Restarting the interrupted boundary repeat from epoch 1 and "
            f"overwriting its checkpoint: {boundary_checkpoint}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    training.logger = Logger(str(output_root / "eicu_node_time_repeat5.log"))
    training.main(training_args)

    missing = [path for path in remaining if not path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(
            f"Training ended without {len(missing)} checkpoint(s) requested in "
            "the resumed range. "
            f"First missing paths:\n{preview}"
        )
    print(
        f"Completed all {len(remaining)} checkpoint(s) in the requested eICU "
        "node_time training range."
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except RuntimeError as error:
        if not (
            args.hard_exit_on_cuda_error
            and training.is_cuda_runtime_error(error)
        ):
            raise

        failure_path = (
            resolve_path(args.output_root)
            / "logs"
            / "eicu_node_time_process_cuda_failure.json"
        )
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "start_fold": args.start_fold,
            "start_repeat": args.start_repeat,
            "batch_size": args.batch_size,
            "max_node_time_tokens": args.max_node_time_tokens,
            "device": args.device,
            "exit_code": 70,
            "linux_nvidia_diagnostics": training.linux_nvidia_diagnostics(),
            "traceback": traceback.format_exc(),
        }
        try:
            training.atomic_write_json(failure_path, payload)
            write_status = f"Process CUDA diagnostic: {failure_path}"
        except Exception as write_error:
            write_status = (
                "Unable to write process CUDA diagnostic: "
                f"{type(write_error).__name__}: {write_error}"
            )

        sys.stderr.write(
            "CUDA/driver failure detected. Diagnostics were flushed; "
            f"terminating this process with exit code 70. {write_status}\n"
        )
        sys.stderr.flush()
        sys.stdout.flush()
        logging.shutdown()
        os._exit(70)


if __name__ == "__main__":
    main()
