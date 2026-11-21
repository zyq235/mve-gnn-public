"""Evaluate every saved fold for a classic time-series baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.mimic_timeseries_baseline_dataset import (
    MIMICTimeseriesDataStore,
    MIMICTimeseriesDataset,
    collate_grud,
)
from timeseries_baselines.common import (
    MODEL_CLASSES,
    build_model,
    calculate_metrics,
    load_state_dict_file,
    move_batch,
)


def build_parser(default_model=None):
    parser = argparse.ArgumentParser(
        description="Evaluate all fold checkpoints for a time-series baseline."
    )
    if default_model is None:
        parser.add_argument(
            "--model-type", choices=sorted(MODEL_CLASSES), default="gru"
        )
    else:
        parser.set_defaults(model_type=default_model)
    parser.add_argument("--pkl-path", default="data/mimic3/mimic_iii.pkl")
    parser.add_argument("--json-path", default="data/mimic3/var_mean_std.json")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-hours", type=float, default=24)
    parser.add_argument("--time-step-minutes", type=float, default=60)
    parser.add_argument("--max-timesteps", type=int, default=880)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--hid-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dropout", type=float, default=0.2)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--r", type=int, default=24)
    parser.add_argument("--M", type=int, default=12)
    parser.add_argument("--pos-class-weight", type=float, default=1.0)
    return parser


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def get_predictions(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            probs = model(**move_batch(batch, device, include_labels=False))
            probabilities.extend(probs.detach().cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def run(args):
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/ft_{args.model_type}_kfold_mm3"
    device = resolve_device(args.device)
    args.device = str(device)
    print(f"Testing {args.model_type} on {device}")
    print("Loading and compacting the processed MIMIC cohort once...")
    store = MIMICTimeseriesDataStore(
        args.pkl_path, args.json_path, total_hours=args.total_hours
    )
    test_set = MIMICTimeseriesDataset(
        store,
        store.test_ids,
        model_type=args.model_type,
        time_step_minutes=args.time_step_minutes,
        max_timesteps=args.max_timesteps,
        truncation_seed=args.seed,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_grud if args.model_type == "grud" else None,
    )

    fold_metrics = []
    for fold_index in range(1, args.num_folds + 1):
        checkpoint_path = (
            Path(args.checkpoint_dir) / f"best_model_fold{fold_index}.pt"
        )
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found. Skipping Fold {fold_index}.")
            continue
        model = build_model(args.model_type, store, args).to(device)
        model.load_state_dict(load_state_dict_file(checkpoint_path, device))
        y_true, y_probs = get_predictions(model, test_loader, device)
        metrics = calculate_metrics(y_true, y_probs)
        fold_metrics.append(metrics)
        print(
            f"Fold {fold_index}: AUC {metrics['auc']:.4f} | "
            f"PR-AUC {metrics['prauc']:.4f} | PR@0.8 {metrics['pr08']:.4f} | "
            f"PR@0.9 {metrics['pr09']:.4f}"
        )

    if not fold_metrics:
        raise FileNotFoundError(
            f"No fold checkpoints were found in {args.checkpoint_dir}"
        )
    print(
        f"\n{'=' * 20} {args.model_type} Final Results "
        f"({len(fold_metrics)} Folds) {'=' * 20}"
    )
    for key in ("auc", "prauc", "pr08", "pr09", "f1", "minrp"):
        values = np.asarray([metrics[key] for metrics in fold_metrics])
        print(f"{key.upper().ljust(10)}: {values.mean():.4f} ± {values.std():.4f}")
    return fold_metrics
    return fold_metrics


def main(default_model=None):
    args = build_parser(default_model).parse_args()
    run(args)


if __name__ == "__main__":
    main()
