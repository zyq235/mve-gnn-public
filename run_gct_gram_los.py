"""Train GCT or GRAM for ten-class remaining ICU LOS on six patient-level folds.

The mortality models, datasets, checkpoints and training scripts are unchanged.
This entry point reuses their encoders and cached-graph representations while
loading LOS targets from the precomputed label CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset.gct_dataset import GCTICUDataset, build_variable_type_index, gct_collate_fn
from dataset.gram_dataset import GRAMICUDataset, build_gram_hierarchy
from dataset.los_patient_dataset import load_los_label_frame
from ehr_baseline_utils import cpu_state_dict, move_batch, resolve_device, set_seed
from ML.los_metrics import NUM_LOS_CLASSES, class_representative_days, compute_los_metrics
from ML.los_smoke import subset_per_class
from models.gct import GCTBinaryClassifier
from models.gram import GRAMBinaryClassifier
from split_protocol import assert_split_integrity, get_sliding_window_splits
from run_los_main_table import load_split_ids


ROOT = Path(__file__).resolve().parent
DATASET_DIRS = {"mimic3": "mimic3", "mimic4": "mimic-iv", "eicu": "eicu"}
METRICS = (
    "linear_kappa", "rps", "mae_days", "macro_f1", "accuracy",
    "within_1_bin_accuracy",
)


class LOSLabelAdapter(Dataset):
    """Replace cached mortality labels at read time without changing the cache."""

    def __init__(self, base: Dataset, label_by_stay: dict[int, int]):
        self.base = base
        self.stay_ids = tuple(map(int, base.stay_ids))
        missing = set(self.stay_ids) - set(label_by_stay)
        if missing:
            raise KeyError(f"Missing LOS labels for stays: {sorted(missing)[:5]}")
        self.label_by_stay = label_by_stay

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = dict(self.base[index])
        stay_id = int(sample["stay_id"])
        sample["labels"] = torch.tensor(self.label_by_stay[stay_id], dtype=torch.long)
        return sample


def make_base_dataset(args, paths, stay_ids, variable_types, hierarchy):
    common = {"index_path": paths.index_path, "time_limit_hours": args.landmark_hours}
    if args.model == "gct":
        return GCTICUDataset(
            paths.processed_dir, stay_ids, variable_type_index=variable_types,
            max_codes=args.max_codes, **common,
        )
    return GRAMICUDataset(
        paths.processed_dir, stay_ids, num_variables=hierarchy.num_leaves,
        bin_minutes=args.bin_minutes, **common,
    )


def make_loader(args, dataset, batch_size, *, sampler=None):
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        collate_fn=gct_collate_fn if args.model == "gct" else None,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=args.device_object.type == "cuda",
    )


def make_model(args, demo_dim, variable_types, hierarchy):
    if args.model == "gct":
        model = GCTBinaryClassifier(
            num_variables=int(variable_types.numel()), demo_dim=demo_dim,
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            num_stacks=args.num_stacks, num_feedforward=args.num_feedforward,
            dropout=args.dropout if args.dropout is not None else 0.08,
            attention_normalizer="softmax", head_aggregation="concat",
            use_prior=True, use_guide=True,
        )
    else:
        model = GRAMBinaryClassifier(
            num_nodes=hierarchy.num_nodes,
            ancestor_index=hierarchy.ancestor_index,
            ancestor_mask=hierarchy.ancestor_mask,
            demo_dim=demo_dim, demo_hidden_dim=args.demo_hidden_dim,
            embedding_dim=args.embedding_dim, attention_dim=args.attention_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout if args.dropout is not None else 0.5,
            use_values=True,
        )
    old_head = model.classifier
    model.classifier = nn.Linear(old_head.in_features, NUM_LOS_CLASSES)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    return model.to(args.device_object)


def forward_logits(args, model, batch):
    if args.model == "gct":
        output = model(batch)
        return output["logits"], output["kl_loss"]
    logits = model(batch["presence"], batch["values"], batch["lengths"], batch["demo"])
    return logits, logits.new_zeros(())


def train_epoch(args, model, loader, optimizer):
    model.train()
    total, count = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        if getattr(args, "max_train_steps", None) is not None and step > args.max_train_steps:
            break
        batch = move_batch(batch, args.device_object)
        labels = batch["labels"].long().view(-1)
        optimizer.zero_grad(set_to_none=True)
        logits, kl_loss = forward_logits(args, model, batch)
        loss = F.cross_entropy(logits, labels)
        if args.model == "gct":
            loss = loss + args.regularization_coefficient * kl_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite GCT/GRAM LOS training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(args, model, loader, days_by_stay, representatives):
    model.eval()
    ids, labels, probabilities = [], [], []
    for batch in loader:
        batch = move_batch(batch, args.device_object)
        logits, _ = forward_logits(args, model, batch)
        if logits.ndim != 2 or logits.shape[1] != NUM_LOS_CLASSES:
            raise ValueError(f"Expected [batch, 10] logits, got {tuple(logits.shape)}")
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("Non-finite GCT/GRAM LOS evaluation logits")
        ids.append(batch["stay_id"].view(-1).cpu().numpy())
        labels.append(batch["labels"].view(-1).cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    stay_ids = np.concatenate(ids).astype(np.int64)
    true_labels = np.concatenate(labels).astype(np.int64)
    probs = np.concatenate(probabilities)
    days = np.asarray([days_by_stay[int(stay)] for stay in stay_ids], dtype=np.float64)
    metrics = compute_los_metrics(true_labels, probs, days, representatives)
    return metrics, {"ts_id": stay_ids, "true_class": true_labels,
                     "remaining_los_days": days, "probabilities": probs}


def prediction_frame(args, fold, arrays, targets, representatives):
    ids = arrays["ts_id"]
    probs = arrays["probabilities"]
    predicted = probs.argmax(axis=1)
    selected = targets.loc[ids]
    frame = pd.DataFrame({
        "dataset": args.dataset, "model": args.model, "fold": fold,
        "ts_id": ids, "true_class": arrays["true_class"],
        "pred_class": predicted,
        "remaining_los_days": arrays["remaining_los_days"],
        "predicted_los_days": representatives[predicted],
        "icu_death": selected["icu_death"].to_numpy(dtype=int),
        "in_hospital_mortality": selected["in_hospital_mortality"].to_numpy(dtype=int),
    })
    for index in range(NUM_LOS_CLASSES):
        frame[f"prob_{index}"] = probs[:, index]
    return frame


def write_results(args, fold_rows, prediction_frames):
    output_dir = Path(args.results_root) / args.dataset / "los"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.model}_{args.landmark_hours}h"
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / f"{prefix}_fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / f"{prefix}_test_predictions.csv", index=False
    )
    pd.DataFrame([
        {"dataset": args.dataset, "model": args.model, "metric": metric,
         "mean": folds[f"test_{metric}"].mean(),
         "std": folds[f"test_{metric}"].std(ddof=0), "n_folds": len(folds)}
        for metric in METRICS
    ]).to_csv(output_dir / f"{prefix}_summary.csv", index=False)


def run(args):
    if args.landmark_hours != 24:
        raise ValueError("GCT/GRAM LOS currently supports the 24h landmark")
    if args.batch_size is None:
        args.batch_size = 32 if args.model == "gct" else 64
    set_seed(args.seed)
    args.device_object = resolve_device(args.device)
    data_dir = Path(args.data_root) / DATASET_DIRS[args.dataset]
    processed_dir = data_dir / "processed"
    paths = SimpleNamespace(
        data_dir=data_dir,
        processed_dir=processed_dir,
        index_path=processed_dir / "processed_files.json",
        var2ind_path=data_dir / "var2ind.json",
        metadata_path=Path(args.statistics_path) if args.statistics_path else data_dir / "var_mean_std.json",
        pickle_path=Path(args.pkl_path) if args.pkl_path else data_dir / {
            "mimic3": "mimic_iii.pkl", "mimic4": "mimic_iv.pkl", "eicu": "eicu.pkl"
        }[args.dataset],
    )
    for required in (paths.pickle_path, paths.metadata_path, paths.var2ind_path, paths.index_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    label_path = Path(args.los_labels_path) if args.los_labels_path else (
        paths.data_dir / "los_24h_labels.csv"
    )
    train_ids, val_ids, test_ids = load_split_ids(
        args.dataset, paths.data_dir, paths.pickle_path
    )
    development_ids = np.concatenate((train_ids, val_ids))
    targets = load_los_label_frame(
        label_path, np.concatenate((development_ids, test_ids)),
        dataset_name=args.dataset, landmark_hours=args.landmark_hours,
    ).set_index("ts_id")
    label_by_stay = targets["los_class"].astype(int).to_dict()
    days_by_stay = targets["remaining_los_days"].astype(float).to_dict()
    patient_by_stay = targets["patient_id"].to_dict()
    assert_split_integrity(development_ids, (), test_ids, patient_by_stay)
    folds = get_sliding_window_splits(
        development_ids, patient_by_stay, n_splits=args.n_folds,
        window_size=args.window_size, seed=args.seed,
    )[: args.max_folds]
    if getattr(args, "smoke_samples_per_class", None) is not None:
        test_ids = subset_per_class(test_ids, targets, args.smoke_samples_per_class)
        folds = [
            (
                subset_per_class(train, targets, args.smoke_samples_per_class, require_all=True),
                subset_per_class(validation, targets, args.smoke_samples_per_class),
            )
            for train, validation in folds
        ]

    variable_types = hierarchy = None
    if args.model == "gct":
        variable_types, _ = build_variable_type_index(paths.var2ind_path, paths.metadata_path)
    else:
        hierarchy = build_gram_hierarchy(paths.var2ind_path, paths.metadata_path)
    test_base = make_base_dataset(args, paths, test_ids, variable_types, hierarchy)
    test_loader = make_loader(
        args, LOSLabelAdapter(test_base, label_by_stay), args.eval_batch_size
    )
    fold_rows, prediction_frames = [], []
    for fold, (fold_train_ids, fold_val_ids) in enumerate(folds, start=1):
        assert_split_integrity(fold_train_ids, fold_val_ids, test_ids, patient_by_stay)
        train_targets = targets.loc[fold_train_ids]
        representatives = class_representative_days(
            train_targets["los_class"], train_targets["remaining_los_days"]
        )
        train_base = make_base_dataset(args, paths, fold_train_ids, variable_types, hierarchy)
        val_base = make_base_dataset(args, paths, fold_val_ids, variable_types, hierarchy)
        train_labels = np.asarray([label_by_stay[int(stay)] for stay in fold_train_ids])
        counts = np.bincount(train_labels, minlength=NUM_LOS_CLASSES)
        weights = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(train_labels), replacement=True
        )
        train_loader = make_loader(
            args, LOSLabelAdapter(train_base, label_by_stay), args.batch_size,
            sampler=sampler,
        )
        val_loader = make_loader(
            args, LOSLabelAdapter(val_base, label_by_stay), args.eval_batch_size
        )
        best_state = best_val = None
        best_repeat = best_epoch = None
        for repeat in range(1, args.repeats + 1):
            set_seed(args.seed + fold * 1000 + repeat)
            model = make_model(args, train_base.demo_dim, variable_types, hierarchy)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            for epoch in range(1, args.epochs + 1):
                train_loss = train_epoch(args, model, train_loader, optimizer)
                val_metrics, _ = evaluate(
                    args, model, val_loader, days_by_stay, representatives
                )
                scheduler.step()
                if (best_val is None or val_metrics["linear_kappa"] > best_val["linear_kappa"]
                    or (val_metrics["linear_kappa"] == best_val["linear_kappa"]
                        and val_metrics["rps"] < best_val["rps"])):
                    best_state = cpu_state_dict(model)
                    best_val = val_metrics.copy()
                    best_repeat, best_epoch = repeat, epoch
                print(
                    f"model={args.model} fold={fold} repeat={repeat} epoch={epoch} "
                    f"train_loss={train_loss:.4f} val_kappa={val_metrics['linear_kappa']:.4f} "
                    f"val_rps={val_metrics['rps']:.4f}", flush=True,
                )
        if best_state is None:
            raise RuntimeError(f"Fold {fold} produced no validation checkpoint")
        checkpoint = Path(args.checkpoint_root) / args.dataset / args.model / f"fold_{fold}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "representative_days": torch.as_tensor(representatives),
            "best_repeat": best_repeat, "best_epoch": best_epoch,
            "validation_metrics": best_val,
            "dataset": args.dataset, "model": args.model, "fold": fold,
        }, checkpoint)
        selected_model = make_model(args, train_base.demo_dim, variable_types, hierarchy)
        selected_model.load_state_dict(best_state, strict=True)
        test_metrics, arrays = evaluate(
            args, selected_model, test_loader, days_by_stay, representatives
        )
        fold_rows.append({
            "dataset": args.dataset, "model": args.model,
            "landmark_hours": args.landmark_hours, "fold": fold,
            "best_repeat": best_repeat, "best_epoch": best_epoch,
            **{f"val_{key}": value for key, value in best_val.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        })
        prediction_frames.append(prediction_frame(args, fold, arrays, targets, representatives))
        write_results(args, fold_rows, prediction_frames)
        print(
            f"model={args.model} fold={fold} test_kappa={test_metrics['linear_kappa']:.4f} "
            f"test_rps={test_metrics['rps']:.4f} test_mae_days={test_metrics['mae_days']:.4f}",
            flush=True,
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("gct", "gram"))
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_DIRS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--pkl-path")
    parser.add_argument("--statistics-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--regularization-coefficient", type=float, default=0.1)
    parser.add_argument("--max-codes", type=int, default=256)
    parser.add_argument("--bin-minutes", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-stacks", type=int, default=3)
    parser.add_argument("--num-feedforward", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--demo-hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--checkpoint-root", default="checkpoints/los")
    parser.add_argument("--max-train-steps", type=int,
                        help="Smoke only: cap optimizer updates per epoch")
    parser.add_argument("--smoke-samples-per-class", type=int,
                        help="Smoke only: use at most this many stays per LOS class in each split")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_folds < 2 or not 1 <= args.window_size < args.n_folds:
        parser.error("Invalid n-folds/window-size combination")
    if args.max_folds is not None and not 1 <= args.max_folds <= args.n_folds:
        parser.error("--max-folds must be between 1 and --n-folds")
    for name in ("repeats", "epochs", "batch_size", "eval_batch_size", "hidden_dim"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.learning_rate <= 0:
        parser.error("Invalid worker count or learning rate")
    run(args)


if __name__ == "__main__":
    main()
