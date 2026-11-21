"""Stage 1: train five MVE-GNN teachers on the complete original train split.

Each repeat sees every stay in the pickle's original train split once per
epoch.  No six-fold repartitioning is performed.  The untouched original
validation split selects the best epoch independently for each repeat.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from dataset.ehr_cached_graph_dataset import SUPPORTED_DATASETS
from ehr_baseline_utils import cpu_state_dict, resolve_device, save_json
from mve_bootstrap_common import (
    TEACHER_REPEATS,
    collect_mve_logits,
    logit_metrics,
    make_mve_loader,
    repeat_seed,
    set_seed,
)
from run_patient_bootstrap import MVEAdapter, load_original_split


ROOT = Path(__file__).resolve().parent


def default_output_dir(args: argparse.Namespace) -> Path:
    mode = "diff" if args.pool_mode == "diff_pool" else args.pool_mode
    return (
        ROOT
        / "results"
        / "patient_bootstrap"
        / args.dataset
        / f"mve_{mode}_repeat5_teachers"
        / f"seed_{args.seed}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats != TEACHER_REPEATS:
        raise ValueError(
            f"This protocol requires exactly {TEACHER_REPEATS} repeats"
        )
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("weight-decay must be nonnegative")
    if args.gradient_clip is not None and args.gradient_clip <= 0:
        raise ValueError("gradient-clip must be positive")


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    output_dir = (args.output_dir or default_output_dir(args)).resolve()
    manifest_path = output_dir / "teacher_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Teacher manifest already exists: {manifest_path}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, protocol, digest = load_original_split(
        args.dataset,
        args.data_root.resolve(),
        require_graph_cache=True,
        require_metadata=True,
    )
    device = resolve_device(args.device)
    adapter = MVEAdapter(paths, {}, args, output_dir, device)
    train_dataset = adapter.dataset(protocol.train_ids)
    validation_dataset = adapter.dataset(protocol.validation_ids)
    if getattr(train_dataset, "demo_dim", 2) != getattr(
        validation_dataset, "demo_dim", 2
    ):
        raise ValueError("Train and validation demographic dimensions differ")

    repeat_rows = []
    for repeat_number in range(1, args.repeats + 1):
        seed = repeat_seed(args.seed, repeat_number)
        set_seed(seed)
        repeat_dir = output_dir / f"repeat_{repeat_number}"
        checkpoint_path = repeat_dir / "best_validation.pt"
        if checkpoint_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Repeat checkpoint already exists: {checkpoint_path}; use --overwrite"
            )
        repeat_dir.mkdir(parents=True, exist_ok=True)
        train_loader = make_mve_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=True,
            seed=seed,
        )
        validation_loader = make_mve_loader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
        )
        model = adapter.build_model(
            device, getattr(train_dataset, "demo_dim", 2)
        )
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else float(adapter.model_args.lr)
        )
        weight_decay = (
            args.weight_decay
            if args.weight_decay is not None
            else float(adapter.default_weight_decay)
        )
        gradient_clip = (
            args.gradient_clip
            if args.gradient_clip is not None
            else float(adapter.default_gradient_clip)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        best_score = -float("inf")
        best_epoch = None
        best_metrics = None
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss = adapter.train_epoch(
                model, train_loader, optimizer, device, gradient_clip
            )
            ids, labels, logits = collect_mve_logits(
                adapter,
                model,
                validation_loader,
                device,
                protocol.validation_ids,
            )
            validation_frame_labels = dict(zip(ids.tolist(), labels.tolist()))
            for stay_id, label in validation_frame_labels.items():
                if int(label) != int(protocol.label_by_stay[int(stay_id)]):
                    raise ValueError(
                        f"Validation label mismatch for stay {stay_id}"
                    )
            metrics = logit_metrics(labels, logits)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **metrics,
                }
            )
            if metrics["selection_score"] > best_score:
                best_score = float(metrics["selection_score"])
                best_epoch = epoch
                best_metrics = dict(metrics)
                torch.save(
                    {
                        "model_state": cpu_state_dict(model),
                        "repeat": repeat_number,
                        "seed": seed,
                        "epoch": epoch,
                        "validation": best_metrics,
                        "dataset": args.dataset,
                        "pool_mode": adapter.model_args.pool_mode,
                        "split_digest": digest,
                    },
                    checkpoint_path,
                )
            scheduler.step()
            print(
                f"repeat={repeat_number}/{args.repeats} epoch={epoch}/{args.epochs} "
                f"loss={train_loss:.4f} val_auc={metrics['auc']:.4f} "
                f"val_prauc={metrics['prauc']:.4f} best={best_score:.4f}",
                flush=True,
            )

        if best_epoch is None or best_metrics is None:
            raise RuntimeError(f"Repeat {repeat_number} produced no checkpoint")
        save_json(repeat_dir / "history.json", history)
        repeat_rows.append(
            {
                "repeat": repeat_number,
                "seed": seed,
                "checkpoint": str(checkpoint_path.resolve()),
                "selected_epoch": best_epoch,
                "validation_metrics": best_metrics,
            }
        )
        del model, optimizer, scheduler, train_loader, validation_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "stage": "mve_full_train_repeat5_teachers",
        "dataset": args.dataset,
        "pool_mode": (
            "diff_pool" if args.pool_mode in {"diff", "diff_pool"} else args.pool_mode
        ),
        "split_digest": digest,
        "training_protocol": (
            "all original train stays once per epoch; original validation selects "
            "each repeat checkpoint; no folds"
        ),
        "repeats": args.repeats,
        "epochs": args.epochs,
        "train_stays": len(protocol.train_ids),
        "validation_stays": len(protocol.validation_ids),
        "test_stays": len(protocol.test_ids),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "pretrained_checkpoint": str(adapter.dataset_config.checkpoint_path),
        "pretrain_settings_file": str(adapter.dataset_config.pretrain_config_path),
        "finetune_settings_file": str(adapter.dataset_config.finetune_config_path),
        "teachers": repeat_rows,
    }
    save_json(manifest_path, manifest)
    print(f"Saved five teacher checkpoints and manifest to {output_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="mimic3")
    parser.add_argument(
        "--pool-mode",
        choices=("type", "time", "node_time", "diff", "diff_pool"),
        default="type",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--repeats", type=int, default=TEACHER_REPEATS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--settings-root", type=Path, default=ROOT / "settings")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
