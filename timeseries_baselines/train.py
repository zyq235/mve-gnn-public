"""Shared six-fold trainer for GRU, GRU-D, SAnD, and TCN."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset.mimic_timeseries_baseline_dataset import (
    MIMICTimeseriesDataStore,
    MIMICTimeseriesDataset,
    collate_grud,
)
from split_protocol import (
    assert_split_integrity,
    get_sliding_window_splits,
    patient_mapping_from_outcomes,
)
from timeseries_baselines.common import (
    MODEL_CLASSES,
    build_model,
    calculate_metrics,
    load_matching_weights,
    move_batch,
)


def build_parser(default_model=None):
    parser = argparse.ArgumentParser(
        description="Train a classic time-series baseline with patient-safe folds."
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
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-hours", type=float, default=24)
    parser.add_argument("--time-step-minutes", type=float, default=60)
    parser.add_argument("--max-timesteps", type=int, default=880)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=443)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--num-workers", type=int, default=0)
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


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args, training=False, fold_seed=None):
    sampler = None
    if training:
        labels = dataset.labels_array()
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = torch.as_tensor(
            class_weights[labels], dtype=torch.double
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed if fold_seed is None else fold_seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.train_batch_size if training else args.eval_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=resolve_device(args.device).type == "cuda",
        collate_fn=collate_grud if dataset.model_type == "grud" else None,
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**move_batch(batch, device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            probs = model(**move_batch(batch, device, include_labels=False))
            total_loss += float(criterion(probs, labels).item())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / max(len(loader), 1), calculate_metrics(all_labels, all_probs)


def _dataset(store, ids, args, truncation_seed):
    return MIMICTimeseriesDataset(
        store,
        ids,
        model_type=args.model_type,
        time_step_minutes=args.time_step_minutes,
        max_timesteps=args.max_timesteps,
        truncation_seed=truncation_seed,
    )


def run(args):
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/ft_{args.model_type}_kfold_mm3"
    device = resolve_device(args.device)
    args.device = str(device)
    set_all_seeds(args.seed)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | Model: {args.model_type}")
    print("Loading and compacting the processed MIMIC cohort once...")
    store = MIMICTimeseriesDataStore(
        args.pkl_path, args.json_path, total_hours=args.total_hours
    )
    id_to_patient = patient_mapping_from_outcomes(store.outcomes)
    combined_ids = np.concatenate((store.train_ids, store.val_ids))
    folds = get_sliding_window_splits(
        combined_ids,
        id_to_patient,
        n_splits=args.num_folds,
        window_size=args.window_size,
        seed=args.split_seed,
    )

    config_record = dict(vars(args))
    config_record.update(
        num_features=store.num_features,
        num_demo=store.num_demo,
        num_steps=int(np.ceil(store.max_minute / args.time_step_minutes)),
        selection_metric="validation_auc_plus_prauc",
    )
    with (Path(args.checkpoint_dir) / "training_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config_record, handle, indent=2, ensure_ascii=False)

    test_set = _dataset(store, store.test_ids, args, truncation_seed=args.seed)
    test_loader = make_loader(test_set, args, training=False)

    for fold_index, (fold_train_ids, fold_val_ids) in enumerate(folds, start=1):
        assert_split_integrity(
            fold_train_ids, fold_val_ids, store.test_ids, id_to_patient
        )
        print(
            f"\n{'=' * 10} {args.model_type} Fold {fold_index} / "
            f"{args.num_folds} {'=' * 10}"
        )
        train_set = _dataset(store, fold_train_ids, args, truncation_seed=None)
        val_set = _dataset(
            store, fold_val_ids, args, truncation_seed=args.seed + fold_index
        )
        train_loader = make_loader(
            train_set, args, training=True, fold_seed=args.seed + fold_index
        )
        val_loader = make_loader(val_set, args, training=False)

        model = build_model(args.model_type, store, args).to(device)
        if args.init_checkpoint:
            if not os.path.exists(args.init_checkpoint):
                raise FileNotFoundError(args.init_checkpoint)
            matched, total = load_matching_weights(
                model, args.init_checkpoint, device
            )
            print(
                f"Initialized {matched}/{total} compatible tensors from "
                f"{args.init_checkpoint}"
            )
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        best_score = -np.inf
        best_val_metrics = None
        checkpoint_path = Path(args.checkpoint_dir) / f"best_model_fold{fold_index}.pt"

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_metrics = evaluate(model, val_loader, device)
            score = val_metrics["auc"] + val_metrics["prauc"]
            saved = score > best_score
            if saved:
                best_score = score
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), checkpoint_path)
            suffix = " *Saved*" if saved else ""
            print(
                f"  Epoch {epoch}: Train Loss {train_loss:.4f} | "
                f"Val Loss {val_loss:.4f} | Val AUC {val_metrics['auc']:.4f} | "
                f"Val PR-AUC {val_metrics['prauc']:.4f}{suffix}"
            )

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        _, test_metrics = evaluate(model, test_loader, device)
        print(
            f"Fold {fold_index} Finished. Best Val AUC: "
            f"{best_val_metrics['auc']:.4f} | Best Val PR-AUC: "
            f"{best_val_metrics['prauc']:.4f} | Test AUC: "
            f"{test_metrics['auc']:.4f} | Test PR-AUC: "
            f"{test_metrics['prauc']:.4f}"
        )


def main(default_model=None):
    args = build_parser(default_model).parse_args()
    run(args)


if __name__ == "__main__":
    main()
