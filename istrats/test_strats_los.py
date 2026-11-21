"""Evaluate saved iStraTS LOS folds and export CSV results."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.los_patient_dataset import load_los_label_frame
from dataset.strats_los_dataset import StratsLOSDataset
from istrats.train_strats_los import (
    DATASET_CONFIGS,
    _torch_load,
    build_model_args,
    evaluate,
    prediction_frame,
    resolve_paths,
    validate_paths,
    write_summary,
)
from models.StraTS.modeling_strats_los import StratsLOS


LOGGER = logging.getLogger("test_strats_los")


def main(args):
    paths = resolve_paths(args)
    validate_paths(paths, no_pretrained=True)
    with paths.pkl_path.open("rb") as handle:
        events, outcomes, _, _, test_ids = pickle.load(handle)
    test_ids = np.asarray(test_ids, dtype=np.int64)
    targets = load_los_label_frame(
        paths.los_labels_path,
        test_ids,
        dataset_name=args.dataset,
        landmark_hours=args.landmark_hours,
    )
    test_dataset = StratsLOSDataset(
        paths.pkl_path,
        paths.statistics_path,
        test_ids,
        targets,
        total_hours=args.landmark_hours,
        max_seq_len=args.max_seq_len,
        events=events,
        training=False,
    )
    del events, outcomes
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device_object.type == "cuda",
    )

    fold_rows = []
    prediction_frames = []
    for fold_index in range(1, args.num_folds + 1):
        checkpoint_path = paths.checkpoint_dir / f"best_model_fold{fold_index}.pt"
        if not checkpoint_path.exists():
            LOGGER.warning("Checkpoint not found, skipping fold %d: %s", fold_index, checkpoint_path)
            continue
        checkpoint = _torch_load(checkpoint_path, args.device_object)
        if "model_state_dict" not in checkpoint or "representative_days" not in checkpoint:
            raise ValueError(
                f"Not an iStraTS LOS checkpoint with metric metadata: {checkpoint_path}"
            )
        expected_features = int(checkpoint.get("num_features", test_dataset.num_features))
        if expected_features != test_dataset.num_features:
            raise ValueError(
                f"Checkpoint expects {expected_features} variables but dataset has "
                f"{test_dataset.num_features}: {checkpoint_path}"
            )
        model = StratsLOS(
            build_model_args(test_dataset.num_features, args)
        ).to(args.device_object)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        representative_days = np.asarray(
            checkpoint["representative_days"].cpu(), dtype=np.float64
        )
        metrics, arrays = evaluate(
            model, test_loader, args, representative_days
        )
        fold_rows.append(
            {
                "dataset": args.dataset,
                "model": "istrats",
                "landmark_hours": args.landmark_hours,
                "fold": fold_index,
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
        prediction_frames.append(
            prediction_frame(
                arrays, representative_days, args.dataset, fold_index
            )
        )
        LOGGER.info(
            "fold=%d kappa=%.4f rps=%.4f mae_days=%.4f",
            fold_index,
            metrics["linear_kappa"],
            metrics["rps"],
            metrics["mae_days"],
        )

    if not fold_rows:
        raise FileNotFoundError(
            f"No iStraTS LOS fold checkpoints found in {paths.checkpoint_dir}"
        )
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"istrats_{args.landmark_hours}h_test"
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(paths.result_dir / f"{prefix}_fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        paths.result_dir / f"{prefix}_predictions.csv", index=False
    )
    targets.to_csv(
        paths.result_dir / f"istrats_los{args.landmark_hours}_test_labels.csv",
        index=False,
    )
    write_summary(fold_frame, paths.result_dir / f"{prefix}_summary.csv")
    LOGGER.info("Test results written to %s", paths.result_dir.resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate iStraTS ten-class remaining ICU LOS checkpoints"
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="mimic3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=880)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--num-folds", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--attention-dropout", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data-root")
    parser.add_argument("--pkl-path")
    parser.add_argument("--statistics-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--checkpoint-root", default="checkpoints/los")
    parser.add_argument("--results-root", default="results")
    parser.set_defaults(no_pretrained=True)
    args = parser.parse_args()
    if args.max_seq_len <= 0 or args.num_folds <= 0:
        raise ValueError("max_seq_len and num_folds must be positive")
    args.device_object = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main(parse_args())
