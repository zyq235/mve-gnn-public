"""Fine-tune iStraTS for ten-class remaining ICU LOS prediction."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.los_patient_dataset import LOS_CLASS_NAMES, load_los_label_frame
from dataset.strats_los_dataset import StratsLOSDataset
from ML.los_metrics import (
    NUM_LOS_CLASSES,
    class_representative_days,
    compute_los_metrics,
)
from ML.los_smoke import subset_per_class
from models.StraTS.modeling_strats_los import StratsLOS
from split_protocol import assert_split_integrity, get_sliding_window_splits


LOGGER = logging.getLogger("train_strats_los")

DATASET_CONFIGS = {
    "mimic3": {
        "data_root": "data/mimic3",
        "pickle": "mimic_iii.pkl",
        "statistics": "var_mean_std.json",
        "pretrained": "checkpoints/finetune_strats_mm3_extend/checkpoint_best.bin",
    },
    "mimic4": {
        "data_root": "data/mimic-iv",
        "pickle": "mimic_iv.pkl",
        "statistics": "var_mean_std.json",
        "pretrained": "checkpoints/finetune_strats_mm4/checkpoint_best.bin",
    },
    "eicu": {
        "data_root": "data/eicu",
        "pickle": "eicu.pkl",
        "statistics": "var_mean_std.json",
        "pretrained": "checkpoints/finetune_strats_mm4/checkpoint_best.bin",
    },
}


def resolve_paths(args):
    defaults = DATASET_CONFIGS[args.dataset]
    data_root = Path(args.data_root or defaults["data_root"])
    return SimpleNamespace(
        data_root=data_root,
        pkl_path=Path(args.pkl_path or data_root / defaults["pickle"]),
        statistics_path=Path(
            args.statistics_path or data_root / defaults["statistics"]
        ),
        los_labels_path=Path(
            args.los_labels_path
            or data_root / f"los_{args.landmark_hours}h_labels.csv"
        ),
        pretrained_path=Path(
            args.pretrained_checkpoint or defaults["pretrained"]
        ),
        checkpoint_dir=Path(args.checkpoint_root) / "istrats" / args.dataset,
        result_dir=Path(args.results_root) / args.dataset / "los",
    )


def validate_paths(paths, no_pretrained=False):
    required = [paths.pkl_path, paths.statistics_path, paths.los_labels_path]
    if not no_pretrained:
        required.append(paths.pretrained_path)
    missing = [os.fspath(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required paths are missing:\n" + "\n".join(missing))


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_args(num_features, args):
    # A non-None value preserves the pretrained forecast projection between
    # the patient embedding and the new LOS head.
    return SimpleNamespace(
        model_type="istrats",
        V=int(num_features),
        D=2,
        hid_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        dropout=args.dropout,
        pretrain=0,
        load_ckpt_path="los_finetune_architecture",
        pos_class_weight=1.0,
        M=12,
        kernel_size=4,
        r=24,
    )


def load_matching_pretrained(model, checkpoint_path, device):
    source = _torch_load(checkpoint_path, device)
    if isinstance(source, dict) and "state_dict" in source:
        source = source["state_dict"]
    current = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    current.update(matched)
    model.load_state_dict(current, strict=True)
    LOGGER.info(
        "Loaded %d matching pretrained tensors from %s",
        len(matched),
        checkpoint_path,
    )


def build_model(num_features, args, paths):
    model = StratsLOS(build_model_args(num_features, args)).to(args.device_object)
    if not args.no_pretrained:
        load_matching_pretrained(model, paths.pretrained_path, args.device_object)
    return model


def ordinal_rps_loss(probabilities, labels):
    observed = F.one_hot(labels, num_classes=NUM_LOS_CLASSES).to(probabilities.dtype)
    error = torch.cumsum(probabilities, dim=1)[:, :-1] - torch.cumsum(
        observed, dim=1
    )[:, :-1]
    return error.square().sum(dim=1).div(NUM_LOS_CLASSES - 1).mean()


def los_loss(logits, labels, ordinal_loss_weight=0.0):
    loss = F.cross_entropy(logits, labels)
    if ordinal_loss_weight:
        loss = loss + ordinal_loss_weight * ordinal_rps_loss(
            torch.softmax(logits, dim=1), labels
        )
    return loss


def _move_inputs(batch, device):
    values, times, variables, observation_mask, demo = (
        tensor.to(device) for tensor in batch[:5]
    )
    return values, times, variables, observation_mask, demo


def train_one_epoch(model, loader, optimizer, args):
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        if getattr(args, "max_train_steps", None) is not None and step > args.max_train_steps:
            break
        inputs = _move_inputs(batch, args.device_object)
        labels = batch[5].to(args.device_object).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = los_loss(logits, labels, args.ordinal_loss_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite iStraTS LOS training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def evaluate(model, loader, args, representative_days):
    model.eval()
    losses = []
    arrays = {
        "probabilities": [],
        "labels": [],
        "remaining_los_days": [],
        "ts_id": [],
        "icu_death": [],
        "in_hospital_mortality": [],
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate", leave=False):
            logits = model(*_move_inputs(batch, args.device_object))
            labels = batch[5].to(args.device_object).long()
            losses.append(
                float(los_loss(logits, labels, args.ordinal_loss_weight).item())
            )
            arrays["probabilities"].append(torch.softmax(logits, dim=1).cpu().numpy())
            arrays["labels"].append(batch[5].numpy())
            arrays["remaining_los_days"].append(batch[6].numpy())
            arrays["ts_id"].append(batch[7].numpy())
            arrays["icu_death"].append(batch[8].numpy())
            arrays["in_hospital_mortality"].append(batch[9].numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    arrays["labels"] = arrays["labels"].astype(np.int64)
    arrays["ts_id"] = arrays["ts_id"].astype(np.int64)
    metrics = compute_los_metrics(
        arrays["labels"],
        arrays["probabilities"],
        arrays["remaining_los_days"],
        representative_days,
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics, arrays


def is_better(metrics, best_metrics):
    if best_metrics is None:
        return True
    if metrics["linear_kappa"] != best_metrics["linear_kappa"]:
        return metrics["linear_kappa"] > best_metrics["linear_kappa"]
    return metrics["rps"] < best_metrics["rps"]


def prediction_frame(arrays, representative_days, dataset, fold):
    predictions = arrays["probabilities"].argmax(axis=1)
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "model": "istrats",
            "fold": fold,
            "ts_id": arrays["ts_id"],
            "true_class": arrays["labels"],
            "pred_class": predictions,
            "remaining_los_days": arrays["remaining_los_days"],
            "predicted_los_days": representative_days[predictions],
            "icu_death": arrays["icu_death"],
            "in_hospital_mortality": arrays["in_hospital_mortality"],
        }
    )
    for class_index in range(NUM_LOS_CLASSES):
        frame[f"prob_{class_index}"] = arrays["probabilities"][:, class_index]
    return frame


def write_summary(fold_frame, path):
    metrics = (
        "test_linear_kappa",
        "test_rps",
        "test_mae_days",
        "test_macro_f1",
        "test_accuracy",
        "test_within_1_bin_accuracy",
    )
    summary = pd.DataFrame(
        [
            {
                "dataset": fold_frame["dataset"].iloc[0],
                "model": "istrats",
                "metric": metric.removeprefix("test_"),
                "mean": fold_frame[metric].mean(),
                "std": fold_frame[metric].std(ddof=0),
                "n_folds": len(fold_frame),
            }
            for metric in metrics
        ]
    )
    summary.to_csv(path, index=False)


def main(args):
    paths = resolve_paths(args)
    validate_paths(paths, args.no_pretrained)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with paths.pkl_path.open("rb") as handle:
        events, outcomes, train_ids, val_ids, test_ids = pickle.load(handle)
    development_ids = np.concatenate([train_ids, val_ids]).astype(np.int64)
    test_ids = np.asarray(test_ids, dtype=np.int64)
    all_ids = np.concatenate([development_ids, test_ids])
    targets = load_los_label_frame(
        paths.los_labels_path,
        all_ids,
        dataset_name=args.dataset,
        landmark_hours=args.landmark_hours,
    )
    id_to_patient = dict(
        zip(targets["ts_id"].astype(int), targets["patient_id"])
    )
    assert_split_integrity(development_ids, (), test_ids, id_to_patient)
    folds = get_sliding_window_splits(
        development_ids,
        id_to_patient,
        n_splits=args.num_folds,
        window_size=args.window_size,
        seed=args.seed,
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    if getattr(args, "smoke_samples_per_class", None) is not None:
        indexed_targets = targets.set_index("ts_id")
        test_ids = subset_per_class(test_ids, indexed_targets, args.smoke_samples_per_class)
        folds = [
            (
                subset_per_class(train, indexed_targets, args.smoke_samples_per_class, require_all=True),
                subset_per_class(validation, indexed_targets, args.smoke_samples_per_class),
            )
            for train, validation in folds
        ]
        selected_ids = {int(stay) for stay in test_ids}
        for train, validation in folds:
            selected_ids.update(map(int, train))
            selected_ids.update(map(int, validation))
        all_ids = np.asarray([stay for stay in all_ids if int(stay) in selected_ids], dtype=np.int64)
        targets = targets.loc[targets["ts_id"].isin(selected_ids)].copy()

    base_dataset = StratsLOSDataset(
        paths.pkl_path,
        paths.statistics_path,
        all_ids,
        targets,
        total_hours=args.landmark_hours,
        max_seq_len=args.max_seq_len,
        events=events,
        training=False,
    )
    del events, outcomes
    pin_memory = args.device_object.type == "cuda"
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    test_loader = DataLoader(
        base_dataset.subset(test_ids, training=False),
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_options,
    )
    target_by_id = targets.set_index("ts_id")
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    prediction_frames = []

    for fold_index, (fold_train_ids, fold_val_ids) in enumerate(folds, start=1):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_ids, id_to_patient)
        train_targets = target_by_id.loc[list(map(int, fold_train_ids))]
        representative_days = class_representative_days(
            train_targets["los_class"], train_targets["remaining_los_days"]
        )
        train_loader = DataLoader(
            base_dataset.subset(fold_train_ids, training=True),
            batch_size=args.train_batch_size,
            shuffle=True,
            **loader_options,
        )
        val_loader = DataLoader(
            base_dataset.subset(fold_val_ids, training=False),
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_options,
        )
        model = build_model(base_dataset.num_features, args, paths)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_metrics = None
        checkpoint_path = paths.checkpoint_dir / f"best_model_fold{fold_index}.pt"
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, args)
            val_metrics, _ = evaluate(
                model, val_loader, args, representative_days
            )
            marker = ""
            if is_better(val_metrics, best_metrics):
                best_metrics = val_metrics.copy()
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "representative_days": torch.as_tensor(
                            representative_days, dtype=torch.float32
                        ),
                        "validation_metrics": best_metrics,
                        "num_features": base_dataset.num_features,
                        "los_class_names": LOS_CLASS_NAMES,
                    },
                    checkpoint_path,
                )
                marker = " *Saved*"
            LOGGER.info(
                "fold=%d epoch=%d train_loss=%.4f val_kappa=%.4f val_rps=%.4f%s",
                fold_index,
                epoch,
                train_loss,
                val_metrics["linear_kappa"],
                val_metrics["rps"],
                marker,
            )

        selected = _torch_load(checkpoint_path, args.device_object)
        model.load_state_dict(selected["model_state_dict"], strict=True)
        test_metrics, arrays = evaluate(
            model, test_loader, args, representative_days
        )
        fold_rows.append(
            {
                "dataset": args.dataset,
                "model": "istrats",
                "landmark_hours": args.landmark_hours,
                "fold": fold_index,
                **{f"val_{key}": value for key, value in best_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
        prediction_frames.append(
            prediction_frame(
                arrays, representative_days, args.dataset, fold_index
            )
        )
        LOGGER.info(
            "fold=%d test_kappa=%.4f test_rps=%.4f test_mae_days=%.4f",
            fold_index,
            test_metrics["linear_kappa"],
            test_metrics["rps"],
            test_metrics["mae_days"],
        )

    prefix = f"istrats_{args.landmark_hours}h"
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(paths.result_dir / f"{prefix}_fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        paths.result_dir / f"{prefix}_test_predictions.csv", index=False
    )
    targets.to_csv(
        paths.result_dir / f"istrats_los{args.landmark_hours}_labels.csv", index=False
    )
    write_summary(fold_frame, paths.result_dir / f"{prefix}_summary.csv")
    LOGGER.info("Results written to %s", paths.result_dir.resolve())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune iStraTS for ten-class remaining ICU LOS"
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="mimic3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=880)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--num-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--attention-dropout", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--pkl-path")
    parser.add_argument("--statistics-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--checkpoint-root", default="checkpoints/los")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--max-train-steps", type=int,
                        help="Smoke only: cap optimizer updates per epoch")
    parser.add_argument("--smoke-samples-per-class", type=int,
                        help="Smoke only: use at most this many stays per LOS class in each split")
    return parser


def parse_args():
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.max_seq_len <= 0:
        raise ValueError("epochs and max_seq_len must be positive")
    if args.max_folds is not None and args.max_folds <= 0:
        raise ValueError("max_folds must be positive")
    if args.ordinal_loss_weight < 0:
        raise ValueError("ordinal_loss_weight cannot be negative")
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
