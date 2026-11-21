"""Train GCT with the same cyclic patient-level protocol as finetune_kfold.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.ehr_cached_graph_dataset import (
    SUPPORTED_DATASETS,
    load_protocol_splits,
    resolve_ehr_dataset_paths,
)
from dataset.gct_dataset import (
    GCTICUDataset,
    build_variable_type_index,
    gct_collate_fn,
)
from ehr_baseline_utils import (
    cpu_state_dict,
    make_weighted_sampler,
    resolve_device,
    save_json,
    selected_fold_indices,
    set_seed,
    summarize_folds,
)
from gct.common import build_model, evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="mimic3")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None, help="Run one 1-based fold")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--time-limit-hours", type=float, default=24.0)
    parser.add_argument("--max-codes", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-stacks", type=int, default=3)
    parser.add_argument("--num-feedforward", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument(
        "--attention-normalizer", choices=("softmax", "sigmoid"), default="softmax"
    )
    parser.add_argument(
        "--head-aggregation", choices=("concat", "sum"), default="concat"
    )
    parser.add_argument("--regularization-coefficient", type=float, default=0.1)
    parser.add_argument("--no-prior", action="store_true")
    parser.add_argument("--no-guide", action="store_true")
    return parser.parse_args()


def make_loader(
    dataset: GCTICUDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=gct_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def main(args: argparse.Namespace) -> None:
    if args.repeats <= 0 or args.epochs <= 0:
        raise ValueError("repeats and epochs must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    paths = resolve_ehr_dataset_paths(args.dataset, args.data_root)
    protocol = load_protocol_splits(
        paths,
        n_splits=args.n_folds,
        window_size=args.window_size,
        seed=args.seed,
    )
    variable_types, class_names = build_variable_type_index(
        paths.var2ind_path, paths.metadata_path
    )
    output_dir = args.output_dir or REPO_ROOT / "checkpoints" / f"gct_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {
        "variable_type_index": variable_types,
        "time_limit_hours": args.time_limit_hours,
        "max_codes": args.max_codes,
    }
    test_dataset = GCTICUDataset(
        paths.processed_dir,
        protocol.test_ids,
        index_path=paths.index_path,
        **dataset_kwargs,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model_kwargs = {
        "num_variables": int(variable_types.numel()),
        "demo_dim": test_dataset.demo_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_stacks": args.num_stacks,
        "num_feedforward": args.num_feedforward,
        "dropout": args.dropout,
        "attention_normalizer": args.attention_normalizer,
        "head_aggregation": args.head_aggregation,
        "use_prior": not args.no_prior,
        "use_guide": not args.no_guide,
        "output_attentions": False,
    }

    print(
        f"GCT | dataset={args.dataset} device={device} "
        f"development={len(protocol.development_ids)} test={len(protocol.test_ids)} "
        f"variables={variable_types.numel()} classes={len(class_names)}"
    )
    fold_records: list[dict] = []
    for fold_index in selected_fold_indices(len(protocol.folds), args.fold):
        fold_number = fold_index + 1
        train_ids, validation_ids = protocol.folds[fold_index]
        train_dataset = GCTICUDataset(
            paths.processed_dir,
            train_ids,
            index_path=paths.index_path,
            **dataset_kwargs,
        )
        validation_dataset = GCTICUDataset(
            paths.processed_dir,
            validation_ids,
            index_path=paths.index_path,
            **dataset_kwargs,
        )
        sampler = make_weighted_sampler(train_ids, protocol.label_by_stay)
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
        )
        validation_loader = make_loader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        fold_best_score = -float("inf")
        fold_best_state = None
        fold_best_validation = None
        fold_best_repeat = None
        for repeat_index in range(args.repeats):
            run_seed = args.seed + fold_index * 1000 + repeat_index
            set_seed(run_seed)
            model = build_model(model_kwargs, device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
            repeat_best_score = -float("inf")
            repeat_best_state = None
            repeat_best_metrics = None
            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    regularization_coefficient=args.regularization_coefficient,
                    gradient_clip=args.gradient_clip,
                )
                validation_metrics = evaluate(
                    model,
                    validation_loader,
                    device,
                    regularization_coefficient=args.regularization_coefficient,
                )
                scheduler.step()
                score = validation_metrics["auc"] + validation_metrics["prauc"]
                if score > repeat_best_score:
                    repeat_best_score = score
                    repeat_best_state = cpu_state_dict(model)
                    repeat_best_metrics = dict(validation_metrics)
                    repeat_best_metrics["train_loss"] = train_loss
                    repeat_best_metrics["epoch"] = epoch
                print(
                    f"fold={fold_number} repeat={repeat_index + 1} epoch={epoch} "
                    f"train_loss={train_loss:.4f} val_auc={validation_metrics['auc']:.4f} "
                    f"val_prauc={validation_metrics['prauc']:.4f}"
                )
            if repeat_best_score > fold_best_score:
                fold_best_score = repeat_best_score
                fold_best_state = repeat_best_state
                fold_best_validation = repeat_best_metrics
                fold_best_repeat = repeat_index + 1

        if fold_best_state is None or fold_best_validation is None:
            raise RuntimeError(f"Fold {fold_number} produced no validation checkpoint")
        checkpoint_path = output_dir / f"fold_{fold_number}_best.pt"
        checkpoint = {
            "model_state": fold_best_state,
            "model_kwargs": model_kwargs,
            "dataset_kwargs": dataset_kwargs,
            "dataset": args.dataset,
            "fold": fold_number,
            "repeat": fold_best_repeat,
            "validation_metrics": fold_best_validation,
            "regularization_coefficient": args.regularization_coefficient,
            "protocol": {
                "seed": args.seed,
                "n_folds": args.n_folds,
                "window_size": args.window_size,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        best_model = build_model(model_kwargs, device)
        best_model.load_state_dict(fold_best_state)
        test_metrics = evaluate(
            best_model,
            test_loader,
            device,
            regularization_coefficient=args.regularization_coefficient,
        )
        record = {
            "fold": fold_number,
            "repeat": fold_best_repeat,
            "checkpoint": str(checkpoint_path),
            "validation": fold_best_validation,
            "test": test_metrics,
        }
        fold_records.append(record)
        print(
            f"fold={fold_number} selected_repeat={fold_best_repeat} "
            f"test_auc={test_metrics['auc']:.4f} test_prauc={test_metrics['prauc']:.4f}"
        )

    summary = summarize_folds([record["test"] for record in fold_records])
    save_json(
        output_dir / "results.json",
        {"dataset": args.dataset, "folds": fold_records, "test_summary": summary},
    )
    print("Test summary:", summary)


if __name__ == "__main__":
    main(parse_args())

