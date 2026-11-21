"""Stage 2: distill five full-train MVE-GNN teachers into one model.

The five Stage-1 checkpoints are evaluated one at a time.  Their mean raw
logit is the soft target.  A fresh pretrained MVE-GNN student is optimized on
the original train split and selected exclusively on the original validation
split.  Only after selection are student and teacher ensemble evaluated on the
fixed test split and passed to the patient-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from dataset.ehr_cached_graph_dataset import SUPPORTED_DATASETS
from ehr_baseline_utils import cpu_state_dict, resolve_device, save_json
from mve_bootstrap_common import (
    TEACHER_REPEATS,
    SplitLogits,
    batch_stay_ids,
    binary_distillation_loss,
    checkpoint_signature,
    collect_mve_logits,
    load_checkpoint_state,
    load_logit_cache,
    logit_metrics,
    make_mve_loader,
    prediction_frame,
    save_logit_cache,
    set_seed,
    teacher_lookup,
)
from patient_bootstrap import bootstrap_predictions
from run_patient_bootstrap import (
    MVEAdapter,
    _check_labels,
    _check_prediction_file,
    load_original_split,
)


ROOT = Path(__file__).resolve().parent


def normalized_pool_mode(value: str) -> str:
    return "diff_pool" if value in {"diff", "diff_pool"} else value


def default_teacher_dir(args: argparse.Namespace) -> Path:
    mode = "diff" if args.pool_mode == "diff_pool" else args.pool_mode
    return (
        ROOT
        / "results"
        / "patient_bootstrap"
        / args.dataset
        / f"mve_{mode}_repeat5_teachers"
        / f"seed_{args.seed}"
    )


def default_output_dir(args: argparse.Namespace) -> Path:
    mode = "diff" if args.pool_mode == "diff_pool" else args.pool_mode
    weight = f"{args.distillation_weight:g}".replace(".", "p")
    return (
        ROOT
        / "results"
        / "patient_bootstrap"
        / args.dataset
        / f"mve_{mode}_distilled_w{weight}"
        / f"seed_{args.seed}"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.teacher_repeats != TEACHER_REPEATS:
        raise ValueError(
            f"This protocol requires exactly {TEACHER_REPEATS} teachers"
        )
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.num_workers < 0 or args.bootstrap_reps < 1:
        raise ValueError("num-workers must be nonnegative and bootstrap-reps positive")
    if not 0.0 <= args.distillation_weight <= 1.0:
        raise ValueError("distillation-weight must lie in [0, 1]")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.student_learning_rate is not None and args.student_learning_rate <= 0:
        raise ValueError("student-learning-rate must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("weight-decay must be nonnegative")
    if args.gradient_clip is not None and args.gradient_clip <= 0:
        raise ValueError("gradient-clip must be positive")


def load_teacher_manifest(
    teacher_dir: Path,
    args: argparse.Namespace,
    split_digest: str,
) -> tuple[dict, list[Path]]:
    manifest_path = teacher_dir / "teacher_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 teacher manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "stage": "mve_full_train_repeat5_teachers",
        "dataset": args.dataset,
        "pool_mode": normalized_pool_mode(args.pool_mode),
        "split_digest": split_digest,
        "repeats": TEACHER_REPEATS,
    }
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Teacher manifest does not match this run: {mismatched}")
    teachers = manifest.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != TEACHER_REPEATS:
        raise ValueError("Teacher manifest must contain five teacher entries")
    checkpoints = []
    for repeat_number, entry in enumerate(teachers, start=1):
        if int(entry.get("repeat", -1)) != repeat_number:
            raise ValueError("Teacher manifest repeat numbers must be 1 through 5")
        path = Path(entry["checkpoint"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append(path)
    return manifest, checkpoints


def expected_label_map(protocol, ids: Sequence[int]) -> dict[int, int]:
    return {
        int(stay_id): int(protocol.label_by_stay[int(stay_id)])
        for stay_id in ids
    }


def validate_split_logits(
    split: SplitLogits,
    expected_labels: Mapping[int, int],
    split_name: str,
) -> None:
    actual = {
        int(stay_id): int(label)
        for stay_id, label in zip(split.stay_ids, split.labels)
    }
    if actual != dict(expected_labels):
        raise ValueError(
            f"Teacher {split_name} logits do not match the fixed split IDs and labels"
        )


def compute_teacher_logits(
    adapter,
    checkpoint_paths: Sequence[Path],
    loaders: Mapping[str, object],
    expected_ids: Mapping[str, Sequence[int]],
    device: torch.device,
    demo_dim: int,
) -> dict[str, SplitLogits]:
    model = adapter.build_model(device, demo_dim)
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    members: dict[str, list[np.ndarray]] = {name: [] for name in loaders}
    try:
        for member_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            model.load_state_dict(load_checkpoint_state(checkpoint_path), strict=True)
            model.to(device)
            for split_name, loader in loaders.items():
                ids, labels, logits = collect_mve_logits(
                    adapter,
                    model,
                    loader,
                    device,
                    expected_ids[split_name],
                )
                if split_name not in references:
                    references[split_name] = (ids, labels)
                else:
                    ref_ids, ref_labels = references[split_name]
                    if not np.array_equal(ids, ref_ids) or not np.array_equal(
                        labels, ref_labels
                    ):
                        raise RuntimeError(
                            f"{split_name} order changed between teacher passes"
                        )
                members[split_name].append(logits)
            print(
                f"teacher logits {member_index}/{len(checkpoint_paths)}: "
                f"{checkpoint_path}",
                flush=True,
            )
        return {
            name: SplitLogits(
                stay_ids=references[name][0],
                labels=references[name][1],
                member_logits=np.stack(logits, axis=0),
            )
            for name, logits in members.items()
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def add_prediction_identity(
    frame,
    *,
    args: argparse.Namespace,
    protocol,
    split_digest: str,
    model_name: str,
):
    _check_labels(frame, protocol.label_by_stay)
    frame.insert(
        0,
        "patient_id",
        frame["stay_id"].map(
            lambda stay: str(protocol.patient_by_stay[int(stay)])
        ),
    )
    frame.insert(0, "split_digest", split_digest)
    frame.insert(0, "model", model_name)
    frame.insert(0, "dataset", args.dataset)
    return _check_prediction_file(frame, args, protocol, split_digest)


def train_student(
    adapter,
    train_dataset,
    validation_loader,
    teacher_train: SplitLogits,
    protocol,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    history_path: Path,
):
    student_seed = int(args.seed + 50_000)
    set_seed(student_seed)
    train_loader = make_mve_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=True,
        seed=student_seed,
    )
    model = adapter.build_model(
        device, getattr(train_dataset, "demo_dim", 2)
    )
    learning_rate = (
        args.student_learning_rate
        if args.student_learning_rate is not None
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
    confidence_weight = float(
        getattr(adapter.model_args, "lambda_confidence", 0.0)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    target_by_stay = teacher_lookup(teacher_train)
    best_score = -float("inf")
    best_metrics = None
    best_epoch = None
    history = []

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = {
                "loss": 0.0,
                "hard": 0.0,
                "soft": 0.0,
                "confidence": 0.0,
            }
            examples = 0
            for batch in train_loader:
                batch = adapter.ft.normalize_collated_group_ids(
                    batch.to(device), adapter.model_args
                )
                labels = batch.y.float().view(-1, 1)
                ids = batch_stay_ids(batch)
                try:
                    teacher_values = [target_by_stay[stay_id] for stay_id in ids]
                except KeyError as error:
                    raise KeyError(
                        f"No mean teacher logit for train stay {error.args[0]}"
                    ) from error
                teacher_logits = torch.as_tensor(
                    teacher_values, dtype=labels.dtype, device=device
                ).view(-1, 1)
                optimizer.zero_grad(set_to_none=True)
                use_confidence = (
                    adapter.model_args.pool_mode == "diff_pool"
                    and confidence_weight > 0
                )
                if use_confidence:
                    student_logits, confidence_loss = (
                        adapter.ft.forward_diffpool_with_confidence(
                            model, batch, adapter.model_args.pool_mode
                        )
                    )
                else:
                    student_logits = model(
                        batch, mode=adapter.model_args.pool_mode
                    )
                    confidence_loss = student_logits.new_zeros(())
                distilled_loss, hard_loss, soft_loss = binary_distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    weight=args.distillation_weight,
                    temperature=args.temperature,
                )
                loss = distilled_loss + confidence_weight * confidence_loss
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"Non-finite student loss for train stays {ids}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                count = int(labels.numel())
                examples += count
                totals["loss"] += float(loss.detach()) * count
                totals["hard"] += float(hard_loss.detach()) * count
                totals["soft"] += float(soft_loss.detach()) * count
                totals["confidence"] += float(confidence_loss.detach()) * count

            ids, labels, logits = collect_mve_logits(
                adapter,
                model,
                validation_loader,
                device,
                protocol.validation_ids,
            )
            metrics = logit_metrics(labels, logits)
            row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": totals["loss"] / max(examples, 1),
                "train_hard_loss": totals["hard"] / max(examples, 1),
                "train_distillation_loss": totals["soft"] / max(examples, 1),
                "train_confidence_loss": totals["confidence"] / max(examples, 1),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
            history.append(row)
            if metrics["selection_score"] > best_score:
                best_score = float(metrics["selection_score"])
                best_metrics = dict(metrics)
                best_epoch = epoch
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": cpu_state_dict(model),
                        "epoch": epoch,
                        "validation": best_metrics,
                        "student_seed": student_seed,
                        "dataset": args.dataset,
                        "pool_mode": adapter.model_args.pool_mode,
                        "distillation_weight": args.distillation_weight,
                        "temperature": args.temperature,
                    },
                    checkpoint_path,
                )
            scheduler.step()
            print(
                f"student epoch={epoch}/{args.epochs} loss={row['train_loss']:.4f} "
                f"val_auc={metrics['auc']:.4f} val_prauc={metrics['prauc']:.4f} "
                f"best={best_score:.4f}",
                flush=True,
            )
        if best_metrics is None or best_epoch is None:
            raise RuntimeError("The distilled student produced no checkpoint")
        save_json(history_path, history)
        return model, {
            "selected_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "student_seed": student_seed,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "confidence_weight": confidence_weight,
        }
    finally:
        del optimizer, scheduler, train_loader


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    teacher_dir = (args.teacher_dir or default_teacher_dir(args)).resolve()
    output_dir = (args.output_dir or default_output_dir(args)).resolve()
    prediction_path = output_dir / "predictions.csv"
    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Distilled predictions already exist: {prediction_path}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, protocol, digest = load_original_split(
        args.dataset,
        args.data_root.resolve(),
        require_graph_cache=True,
        require_metadata=True,
    )
    manifest, checkpoint_paths = load_teacher_manifest(
        teacher_dir, args, digest
    )
    signature = checkpoint_signature(checkpoint_paths)
    device = resolve_device(args.device)
    adapter = MVEAdapter(paths, {}, args, output_dir, device)
    configured_pretrain = adapter.dataset_config.checkpoint_path.resolve()
    manifest_pretrain = Path(manifest["pretrained_checkpoint"]).resolve()
    if configured_pretrain != manifest_pretrain:
        raise ValueError(
            "Stage 1 and Stage 2 use different pretrained checkpoints: "
            f"{manifest_pretrain} != {configured_pretrain}"
        )

    train_dataset = adapter.dataset(protocol.train_ids)
    validation_dataset = adapter.dataset(protocol.validation_ids)
    validation_loader = make_mve_loader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    teacher_eval_loaders = {
        "train": make_mve_loader(
            train_dataset,
            batch_size=args.teacher_batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
        ),
        "validation": make_mve_loader(
            validation_dataset,
            batch_size=args.teacher_batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
        ),
    }
    train_validation_cache = output_dir / "teacher_train_validation_logits.npz"
    expected_train_validation = {
        "train": expected_label_map(protocol, protocol.train_ids),
        "validation": expected_label_map(protocol, protocol.validation_ids),
    }
    teacher_splits = None
    if args.reuse_teacher_cache:
        teacher_splits = load_logit_cache(
            train_validation_cache,
            signature,
            expected_train_validation,
            args.teacher_repeats,
        )
    if teacher_splits is None:
        teacher_splits = compute_teacher_logits(
            adapter,
            checkpoint_paths,
            teacher_eval_loaders,
            {
                "train": protocol.train_ids,
                "validation": protocol.validation_ids,
            },
            device,
            getattr(train_dataset, "demo_dim", 2),
        )
        save_logit_cache(train_validation_cache, signature, teacher_splits)
    else:
        print(f"Reused teacher cache: {train_validation_cache}")
    for split_name, expected_labels in expected_train_validation.items():
        validate_split_logits(
            teacher_splits[split_name], expected_labels, split_name
        )

    checkpoint_path = output_dir / "best_distilled_student.pt"
    student, student_training = train_student(
        adapter,
        train_dataset,
        validation_loader,
        teacher_splits["train"],
        protocol,
        args,
        device,
        checkpoint_path,
        output_dir / "student_history.json",
    )
    student.load_state_dict(load_checkpoint_state(checkpoint_path), strict=True)
    student.to(device)

    # The test split is first materialized only after validation has selected
    # the student checkpoint.
    test_dataset = adapter.dataset(protocol.test_ids)
    test_loader = make_mve_loader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    student_test = collect_mve_logits(
        adapter, student, test_loader, device, protocol.test_ids
    )

    test_cache = output_dir / "teacher_test_logits.npz"
    expected_test = {"test": expected_label_map(protocol, protocol.test_ids)}
    teacher_test_splits = None
    if args.reuse_teacher_cache:
        teacher_test_splits = load_logit_cache(
            test_cache,
            signature,
            expected_test,
            args.teacher_repeats,
        )
    if teacher_test_splits is None:
        teacher_test_splits = compute_teacher_logits(
            adapter,
            checkpoint_paths,
            {"test": test_loader},
            {"test": protocol.test_ids},
            device,
            getattr(test_dataset, "demo_dim", 2),
        )
        save_logit_cache(test_cache, signature, teacher_test_splits)
    validate_split_logits(
        teacher_test_splits["test"], expected_test["test"], "test"
    )
    teacher_test = teacher_test_splits["test"]

    mode = "diff" if args.pool_mode == "diff_pool" else args.pool_mode
    student_name = args.run_name or f"mve_{mode}_distilled"
    teacher_name = f"mve_{mode}_teacher_avg5"
    student_frame = add_prediction_identity(
        prediction_frame(student_test),
        args=args,
        protocol=protocol,
        split_digest=digest,
        model_name=student_name,
    )
    teacher_frame = add_prediction_identity(
        prediction_frame(teacher_test, use_mean=True),
        args=args,
        protocol=protocol,
        split_digest=digest,
        model_name=teacher_name,
    )
    student_frame.to_csv(prediction_path, index=False, encoding="utf-8")
    teacher_frame.to_csv(
        output_dir / "teacher_ensemble_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    bootstrap_summary, repetitions = bootstrap_predictions(
        {student_name: student_frame, teacher_name: teacher_frame},
        reference_model=student_name,
        repetitions=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    save_json(output_dir / "bootstrap_summary.json", bootstrap_summary)
    repetitions.to_csv(
        output_dir / "bootstrap_repetitions.csv", index=False
    )
    metadata = {
        "stage": "mve_repeat5_mean_logit_distillation",
        "dataset": args.dataset,
        "pool_mode": normalized_pool_mode(args.pool_mode),
        "split_digest": digest,
        "teacher_manifest": str((teacher_dir / "teacher_manifest.json").resolve()),
        "teacher_checkpoint_signature": signature,
        "teacher_repeats": args.teacher_repeats,
        "teacher_aggregation": "mean raw logits",
        "student_checkpoint": str(checkpoint_path.resolve()),
        "epochs": args.epochs,
        "distillation_weight": args.distillation_weight,
        "hard_label_weight": 1.0 - args.distillation_weight,
        "temperature": args.temperature,
        "train_stays": len(protocol.train_ids),
        "validation_stays": len(protocol.validation_ids),
        "test_stays": len(protocol.test_ids),
        "selection_rule": "maximize original validation AUROC + AUPRC",
        "training_protocol": "original train split; no folds",
        "student_training": student_training,
        "teacher_validation_metrics": logit_metrics(
            teacher_splits["validation"].labels,
            teacher_splits["validation"].mean_logits,
        ),
        "teacher_test_metrics": logit_metrics(
            teacher_test.labels, teacher_test.mean_logits
        ),
        "student_test_metrics": logit_metrics(
            student_test[1], student_test[2]
        ),
        "bootstrap_repetitions": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
    }
    save_json(output_dir / "run_metadata.json", metadata)
    print(f"Saved distilled predictions and bootstrap results to {output_dir}")
    return metadata


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
    parser.add_argument("--teacher-repeats", type=int, default=TEACHER_REPEATS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--student-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--distillation-weight", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--run-name")
    parser.add_argument("--reuse-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--settings-root", type=Path, default=ROOT / "settings")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
