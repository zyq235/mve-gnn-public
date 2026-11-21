"""Distill LOS repeat checkpoints into one MVE-GNN student per fold.

Teachers are the ``best.pt`` files produced by ``finetune_los_kfold.py``.
Their mean *logits* provide a ten-class soft target.  The student starts from
the existing pretrained graph encoder, not from a mortality/LOS teacher, and
is selected using validation linear Kappa (RPS breaks ties).  The held-out
test set is scored only after all student epochs have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import finetune_los_kfold as los_base
from dataset.los_patient_dataset import LOS_CLASS_NAMES, load_los_label_frame
from ML.los_metrics import (
    NUM_LOS_CLASSES,
    class_representative_days,
    compute_los_metrics,
)
from models.graphmae.utils import set_random_seed
from split_protocol import assert_split_integrity, get_sliding_window_splits


LOGGER = logging.getLogger("los_ensemble_distillation")


@dataclass(frozen=True)
class TeacherSplit:
    stay_ids: np.ndarray
    labels: np.ndarray
    member_logits: np.ndarray  # [repeat, stay, class]

    @property
    def mean_logits(self) -> np.ndarray:
        return self.member_logits.mean(axis=0)


def teacher_checkpoint_paths(args, fold: int) -> list[Path]:
    paths = [
        los_base._resolve_path(args.teacher_checkpoint_root, "checkpoints/los")
        / args.dataset
        / args.pool_mode
        / f"fold_{fold}"
        / f"repeat_{repeat}"
        / "best.pt"
        for repeat in range(1, args.teacher_repeats + 1)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing LOS repeat checkpoints:\n" + "\n".join(missing)
        )
    return paths


def checkpoint_signature(paths: list[Path]) -> str:
    metadata = [
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    ]
    return hashlib.sha256(json.dumps(metadata).encode("utf-8")).hexdigest()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    weight: float = 0.75,
    temperature: float = 2.0,
    ordinal_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine hard LOS labels with multiclass temperature-scaled KL."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have identical shapes")
    if student_logits.ndim != 2 or student_logits.shape[1] != NUM_LOS_CLASSES:
        raise ValueError("LOS logits must have shape [batch, 10]")
    if not 0 <= weight <= 1 or temperature <= 0:
        raise ValueError("Invalid distillation weight or temperature")
    hard = los_base._loss(student_logits, labels, ordinal_loss_weight)
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    return (1 - weight) * hard + weight * soft, hard, soft


@torch.no_grad()
def collect_logits(model, loader, device, pool_mode) -> tuple[np.ndarray, ...]:
    model.eval()
    ids, labels, logits = [], [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch, mode=pool_mode)
        if output.ndim != 2 or output.shape[1] != NUM_LOS_CLASSES:
            raise ValueError(f"Expected ten teacher logits, got {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError("Teacher produced non-finite LOS logits")
        ids.append(batch.ts_id.view(-1).cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())
        logits.append(output.cpu().numpy())
    stay_ids = np.concatenate(ids).astype(np.int64)
    if len(np.unique(stay_ids)) != len(stay_ids):
        raise ValueError("A teacher split contains duplicate ICU stays")
    return (
        stay_ids,
        np.concatenate(labels).astype(np.int64),
        np.concatenate(logits),
    )


def collect_teachers(paths, loaders, args, model_paths, pretrain_config, finetune_config) -> dict[str, TeacherSplit]:
    # The complete teacher state replaces random initialization.  Skipping
    # pretraining here avoids an unnecessary encoder load for each fold.
    teacher_args = argparse.Namespace(**vars(args))
    teacher_args.no_pretrained = True
    model = los_base.build_los_model(
        teacher_args, model_paths, pretrain_config, finetune_config
    )
    reference = {}
    members = {split: [] for split in loaders}
    try:
        for member, path in enumerate(paths, start=1):
            payload = los_base._torch_load(path, args.device_object)
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=True)
            for split_name, loader in loaders.items():
                ids, labels, logits = collect_logits(
                    model, loader, args.device_object, args.pool_mode
                )
                if split_name in reference:
                    old_ids, old_labels = reference[split_name]
                    if not np.array_equal(ids, old_ids) or not np.array_equal(
                        labels, old_labels
                    ):
                        raise RuntimeError(
                            f"{split_name} order changed between teacher repeats"
                        )
                else:
                    reference[split_name] = (ids, labels)
                members[split_name].append(logits)
            LOGGER.info("Teacher %d/%d: %s", member, len(paths), path)
        return {
            split_name: TeacherSplit(
                reference[split_name][0],
                reference[split_name][1],
                np.stack(logits, axis=0),
            )
            for split_name, logits in members.items()
        }
    finally:
        del model
        if args.device_object.type == "cuda":
            torch.cuda.empty_cache()


def save_teacher_cache(path: Path, signature: str, splits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"signature": np.asarray(signature)}
    for name, split in splits.items():
        arrays[f"{name}_ids"] = split.stay_ids
        arrays[f"{name}_labels"] = split.labels
        arrays[f"{name}_logits"] = split.member_logits
    np.savez_compressed(path, **arrays)


def load_teacher_cache(
    path: Path, signature: str, expected_labels: dict[str, dict[int, int]],
    teacher_repeats: int,
) -> dict[str, TeacherSplit] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) != signature:
                return None
            result = {}
            for name, expected in expected_labels.items():
                ids = cache[f"{name}_ids"].astype(np.int64)
                labels = cache[f"{name}_labels"].astype(np.int64)
                logits = cache[f"{name}_logits"]
                if (
                    len(ids) != len(expected)
                    or len(np.unique(ids)) != len(ids)
                    or set(ids.tolist()) != set(expected)
                    or not np.array_equal(
                        labels,
                        np.asarray([expected[int(stay)] for stay in ids]),
                    )
                    or logits.shape != (teacher_repeats, len(ids), NUM_LOS_CLASSES)
                    or not np.isfinite(logits).all()
                ):
                    return None
                result[name] = TeacherSplit(ids, labels, logits)
            return result
    except (KeyError, OSError, ValueError, TypeError):
        return None


def validate_teacher_splits(splits, expected_labels) -> None:
    """Require exact split coverage before using or caching teacher outputs."""

    for name, expected in expected_labels.items():
        split = splits[name]
        ids = split.stay_ids.astype(np.int64)
        if len(ids) != len(expected) or set(ids.tolist()) != set(expected):
            raise ValueError(f"Teacher {name} predictions do not cover the cohort")
        if not np.array_equal(
            split.labels,
            np.asarray([expected[int(stay)] for stay in ids], dtype=np.int64),
        ):
            raise ValueError(f"Teacher {name} labels disagree with the LOS label file")


def teacher_metrics(split: TeacherSplit, exact_days, representatives):
    by_stay = exact_days
    days = np.asarray([by_stay[int(stay)] for stay in split.stay_ids])
    rows = []
    for repeat, logits in enumerate(split.member_logits, start=1):
        rows.append(
            {"teacher": f"repeat_{repeat}", **compute_los_metrics(
                split.labels, torch.softmax(torch.as_tensor(logits), dim=1).numpy(),
                days, representatives,
            )}
        )
    rows.append(
        {"teacher": "mean_logits", **compute_los_metrics(
            split.labels,
            torch.softmax(torch.as_tensor(split.mean_logits), dim=1).numpy(),
            days, representatives,
        )}
    )
    return rows


def train_student(
    model, train_loader, val_loader, teacher_train, args, representatives,
    checkpoint_path: Path,
):
    teacher_by_stay = {
        int(stay): logits
        for stay, logits in zip(teacher_train.stay_ids, teacher_train.mean_logits)
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch
    )
    best = None
    epoch_rows = []
    use_confidence = args.pool_mode == "diff_pool" and args.lambda_confidence > 0
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        totals = np.zeros(5, dtype=np.float64)
        for batch in tqdm(train_loader, desc=f"Distill epoch {epoch}/{args.max_epoch}", leave=False):
            batch = batch.to(args.device_object)
            labels = batch.y.view(-1).long()
            ids = batch.ts_id.view(-1).cpu().tolist()
            try:
                soft_logits = torch.as_tensor(
                    np.stack([teacher_by_stay[int(stay)] for stay in ids]),
                    dtype=torch.float32,
                    device=args.device_object,
                )
            except KeyError as error:
                raise KeyError(f"No teacher logits for stay {error.args[0]}") from error
            optimizer.zero_grad(set_to_none=True)
            if use_confidence:
                logits, confidence = los_base.forward_diffpool_with_confidence(
                    model, batch, args.pool_mode
                )
            else:
                logits = model(batch, mode=args.pool_mode)
                confidence = logits.new_zeros(())
            distill_objective, hard, soft = distillation_loss(
                logits, soft_logits, labels, args.distillation_weight,
                args.temperature, args.ordinal_loss_weight,
            )
            loss = distill_objective + args.lambda_confidence * confidence
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite LOS distillation loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            count = len(ids)
            totals += np.asarray([
                loss.item() * count,
                hard.item() * count,
                soft.item() * count,
                confidence.item() * count,
                count,
            ])
        scheduler.step()
        validation, _ = los_base._evaluate(
            model, val_loader, args.device_object, args.pool_mode,
            representatives, args.ordinal_loss_weight,
        )
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": totals[0] / totals[4],
            "train_hard_loss": totals[1] / totals[4],
            "train_distillation_loss": totals[2] / totals[4],
            "train_confidence_loss": totals[3] / totals[4],
            **{f"val_{key}": value for key, value in validation.items()},
        })
        LOGGER.info(
            "epoch=%d train_confidence_loss=%.4f val_kappa=%.4f val_rps=%.4f",
            epoch, totals[3] / totals[4], validation["linear_kappa"],
            validation["rps"],
        )
        if los_base._is_better(validation, best):
            best = validation.copy()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_epoch": epoch,
                "validation_metrics": best,
                "representative_days": torch.as_tensor(representatives),
                "los_class_names": LOS_CLASS_NAMES,
                "distillation_weight": args.distillation_weight,
                "temperature": args.temperature,
                "lambda_confidence": (
                    args.lambda_confidence if use_confidence else 0.0
                ),
            }, checkpoint_path)
    if best is None:
        raise RuntimeError("Student produced no validation checkpoint")
    return best, epoch_rows


def _run_tag(args) -> str:
    if args.experiment_name:
        return args.experiment_name

    def number(value):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    effective_lambda = (
        args.lambda_confidence if args.pool_mode == "diff_pool" else 0.0
    )
    return (
        f"w{number(args.distillation_weight)}_t{number(args.temperature)}_"
        f"lr{number(args.student_lr)}_lc{number(effective_lambda)}"
    )


def result_directory(args) -> Path:
    """Keep concurrently running views from overwriting one another's CSVs."""

    return (
        los_base._resolve_path(args.results_root, "results")
        / args.dataset / "los" / "distillation" / args.pool_mode / _run_tag(args)
    )


def checkpoint_directory(args) -> Path:
    return (
        los_base._resolve_path(args.checkpoint_root, "checkpoints/los_distilled")
        / args.dataset / args.pool_mode / _run_tag(args)
    )


METRIC_NAMES = (
    "linear_kappa", "rps", "mae_days", "macro_f1", "accuracy",
    "within_1_bin_accuracy",
)


def write_student_results(result_dir, fold_rows, prediction_frames, args) -> None:
    """Write the same student test reports for training and checkpoint-only runs."""

    result_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(result_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        result_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame([
        {
            "dataset": args.dataset, "pool_mode": args.pool_mode,
            "metric": metric, "mean": fold_frame[f"test_{metric}"].mean(),
            "std": fold_frame[f"test_{metric}"].std(ddof=0),
            "n_folds": len(fold_frame),
        }
        for metric in METRIC_NAMES
    ]).to_csv(result_dir / "summary.csv", index=False)


def evaluate_saved_students(args) -> None:
    """Re-score all saved fold students without loading teachers or training."""

    paths = los_base.resolve_los_paths(args)
    los_base.validate_los_paths(paths, no_pretrained=True)
    _, _, test_ids = los_base._load_fold_metadata(
        args.dataset, paths["data_root"], paths["pkl_path"]
    )
    labels = load_los_label_frame(
        paths["los_labels"], test_ids,
        dataset_name=args.dataset, landmark_hours=args.landmark_hours,
    )
    pretrain_config, finetune_config = los_base.load_los_configs(
        paths, args.dataset, args.pool_mode
    )
    result_dir = result_directory(args)
    checkpoint_dir = checkpoint_directory(args)
    fold_count = args.max_folds or args.n_folds
    checkpoint_paths = [
        checkpoint_dir / f"fold_{fold}" / "best.pt"
        for fold in range(1, fold_count + 1)
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing distilled LOS students:\n" + "\n".join(missing)
        )

    test_dataset = los_base._make_dataset(
        test_ids, labels, paths, args.landmark_hours
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device_object.type == "cuda",
    )
    model_args = argparse.Namespace(**vars(args))
    model_args.no_pretrained = True
    student = los_base.build_los_model(
        model_args, paths, pretrain_config, finetune_config
    )
    fold_rows, prediction_frames = [], []
    for fold, checkpoint_path in enumerate(checkpoint_paths, start=1):
        payload = los_base._torch_load(checkpoint_path, args.device_object)
        if "model_state_dict" not in payload or "representative_days" not in payload:
            raise ValueError(
                f"Not a complete LOS student checkpoint: {checkpoint_path}"
            )
        representatives = np.asarray(
            payload["representative_days"].cpu(), dtype=np.float64
        )
        if representatives.shape != (NUM_LOS_CLASSES,) or not np.isfinite(representatives).all():
            raise ValueError(f"Invalid LOS representative days: {checkpoint_path}")
        student.load_state_dict(payload["model_state_dict"], strict=True)
        metrics, arrays = los_base._evaluate(
            student, test_loader, args.device_object, args.pool_mode,
            representatives, args.ordinal_loss_weight,
        )
        fold_rows.append({
            "dataset": args.dataset, "pool_mode": args.pool_mode,
            "landmark_hours": args.landmark_hours, "fold": fold,
            "best_epoch": payload.get("best_epoch"),
            "student_checkpoint": str(checkpoint_path.resolve()),
            **{
                f"val_{key}": value
                for key, value in payload.get("validation_metrics", {}).items()
            },
            **{f"test_{key}": value for key, value in metrics.items()},
        })
        prediction_frames.append(
            los_base._prediction_frame(
                arrays, representatives, args.dataset, args.pool_mode, fold, 0
            ).rename(columns={"repeat": "student_repeat"})
        )
        LOGGER.info(
            "checkpoint-only fold=%d test_kappa=%.4f test_rps=%.4f test_mae_days=%.4f",
            fold, metrics["linear_kappa"], metrics["rps"], metrics["mae_days"],
        )
    write_student_results(result_dir, fold_rows, prediction_frames, args)
    LOGGER.info("Checkpoint-only LOS results written to %s", result_dir.resolve())


def run(args) -> None:
    if args.evaluate_only:
        evaluate_saved_students(args)
        return
    paths = los_base.resolve_los_paths(args)
    los_base.validate_los_paths(paths, no_pretrained=args.no_pretrained)
    data_root = paths["data_root"]
    set_random_seed(args.seed)
    train_ids, val_ids, test_ids = los_base._load_fold_metadata(
        args.dataset, data_root, paths["pkl_path"]
    )
    development_ids = np.concatenate((train_ids, val_ids))
    labels = load_los_label_frame(
        paths["los_labels"], np.concatenate((development_ids, test_ids)),
        dataset_name=args.dataset, landmark_hours=args.landmark_hours,
    )
    label_by_stay = labels.set_index("ts_id")
    patient_by_stay = label_by_stay["patient_id"].to_dict()
    class_by_stay = label_by_stay["los_class"].astype(int).to_dict()
    days_by_stay = label_by_stay["remaining_los_days"].astype(float).to_dict()
    folds = get_sliding_window_splits(
        development_ids, id_to_patient=patient_by_stay,
        n_splits=args.n_folds, window_size=args.window_size, seed=args.seed,
    )
    if args.max_folds is not None:
        folds = folds[:args.max_folds]
    teacher_paths_by_fold = [
        teacher_checkpoint_paths(args, fold)
        for fold in range(1, len(folds) + 1)
    ]
    pretrain_config, finetune_config = los_base.load_los_configs(
        paths, args.dataset, args.pool_mode
    )
    result_dir = result_directory(args)
    checkpoint_dir = checkpoint_directory(args)
    result_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = los_base._make_dataset(
        test_ids, labels, paths, args.landmark_hours
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": args.device_object.type == "cuda",
    }
    test_loader = DataLoader(
        test_dataset, batch_size=args.teacher_batch_size, shuffle=False,
        **loader_options,
    )
    fold_rows, teacher_rows, epoch_rows, prediction_frames = [], [], [], []
    for fold, (fold_train_ids, fold_val_ids) in enumerate(folds, start=1):
        assert_split_integrity(
            fold_train_ids, fold_val_ids, test_ids, patient_by_stay
        )
        train_frame = label_by_stay.loc[list(map(int, fold_train_ids))]
        representatives = class_representative_days(
            train_frame["los_class"], train_frame["remaining_los_days"]
        )
        train_dataset = los_base._make_dataset(
            fold_train_ids, labels, paths, args.landmark_hours
        )
        val_dataset = los_base._make_dataset(
            fold_val_ids, labels, paths, args.landmark_hours
        )
        teacher_loaders = {
            "train": DataLoader(train_dataset, batch_size=args.teacher_batch_size,
                                shuffle=False, **loader_options),
            "validation": DataLoader(val_dataset, batch_size=args.teacher_batch_size,
                                     shuffle=False, **loader_options),
            "test": test_loader,
        }
        expected = {
            name: {int(stay): class_by_stay[int(stay)] for stay in ids}
            for name, ids in (
                ("train", fold_train_ids),
                ("validation", fold_val_ids),
                ("test", test_ids),
            )
        }
        signature = checkpoint_signature(teacher_paths_by_fold[fold - 1])
        cache_path = (
            los_base._resolve_path(args.teacher_cache_root, "checkpoints/los_teacher_cache")
            / args.dataset / args.pool_mode / f"fold_{fold}.npz"
        )
        teachers = None
        if not args.no_teacher_cache:
            teachers = load_teacher_cache(
                cache_path, signature, expected, args.teacher_repeats
            )
        if teachers is None:
            teachers = collect_teachers(
                teacher_paths_by_fold[fold - 1], teacher_loaders, args, paths,
                pretrain_config, finetune_config,
            )
            validate_teacher_splits(teachers, expected)
            if not args.no_teacher_cache:
                save_teacher_cache(cache_path, signature, teachers)
        else:
            validate_teacher_splits(teachers, expected)
        teacher_test_summary = None
        for split in ("validation", "test"):
            split_metrics = teacher_metrics(
                teachers[split], days_by_stay, representatives
            )
            if split == "test":
                teacher_test_summary = {
                    row["teacher"]: row for row in split_metrics
                }
            for row in split_metrics:
                teacher_rows.append({
                    "dataset": args.dataset, "pool_mode": args.pool_mode,
                    "fold": fold, "split": split, **row,
                })

        set_random_seed(args.seed + fold * 1000 + 1)
        student = los_base.build_los_model(
            args, paths, pretrain_config, finetune_config
        )
        student_train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            **loader_options,
        )
        student_val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            **loader_options,
        )
        checkpoint_path = checkpoint_dir / f"fold_{fold}" / "best.pt"
        validation, history = train_student(
            student, student_train_loader, student_val_loader, teachers["train"],
            args, representatives, checkpoint_path,
        )
        epoch_rows.extend({"fold": fold, **row} for row in history)
        # Test is evaluated only after epoch selection.  Teacher test logits
        # above are for a reported comparison and never enter the loss/selection.
        payload = los_base._torch_load(checkpoint_path, args.device_object)
        student.load_state_dict(payload["model_state_dict"], strict=True)
        student_test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            **loader_options,
        )
        test_metrics, arrays = los_base._evaluate(
            student, student_test_loader, args.device_object, args.pool_mode,
            representatives, args.ordinal_loss_weight,
        )
        fold_rows.append({
            "dataset": args.dataset, "pool_mode": args.pool_mode,
            "landmark_hours": args.landmark_hours, "fold": fold,
            "teacher_repeats": args.teacher_repeats,
            "distillation_weight": args.distillation_weight,
            "temperature": args.temperature,
            "lambda_confidence": (
                args.lambda_confidence if args.pool_mode == "diff_pool" else 0.0
            ),
            "best_epoch": payload["best_epoch"],
            "student_checkpoint": str(checkpoint_path.resolve()),
            "teacher_checkpoint_signature": signature,
            **{f"val_{key}": value for key, value in validation.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{
                f"teacher_repeat1_test_{key}": value
                for key, value in teacher_test_summary["repeat_1"].items()
                if key != "teacher"
            },
            **{
                f"teacher_mean_test_{key}": value
                for key, value in teacher_test_summary["mean_logits"].items()
                if key != "teacher"
            },
        })
        predictions = los_base._prediction_frame(
            arrays, representatives, args.dataset, args.pool_mode, fold, 0
        ).rename(columns={"repeat": "student_repeat"})
        teacher_test = teachers["test"]
        teacher_probs = torch.softmax(
            torch.as_tensor(teacher_test.mean_logits), dim=1
        ).numpy()
        teacher_by_stay = dict(zip(teacher_test.stay_ids, teacher_probs))
        for class_index in range(NUM_LOS_CLASSES):
            predictions[f"teacher_prob_{class_index}"] = [
                teacher_by_stay[int(stay)][class_index]
                for stay in predictions["ts_id"]
            ]
        prediction_frames.append(predictions)
        LOGGER.info(
            "fold=%d selected_epoch=%d test_kappa=%.4f test_rps=%.4f test_mae_days=%.4f",
            fold, payload["best_epoch"], test_metrics["linear_kappa"],
            test_metrics["rps"], test_metrics["mae_days"],
        )
        pd.DataFrame(teacher_rows).to_csv(result_dir / "teacher_metrics.csv", index=False)
        pd.DataFrame(epoch_rows).to_csv(result_dir / "epoch_metrics.csv", index=False)
        write_student_results(result_dir, fold_rows, prediction_frames, args)
    LOGGER.info("LOS distillation results written to %s", result_dir.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill MVE-GNN LOS repeat ensembles")
    parser.add_argument("--dataset", type=los_base._normalise_dataset_name,
                        default="mimic3")
    parser.add_argument("--pool-mode", choices=("type", "time", "node_time", "diff_pool"),
                        default="type")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--teacher-repeats", type=int, default=5)
    parser.add_argument("--max-epoch", type=int, default=15)
    parser.add_argument("--student-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--distillation-weight", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    parser.add_argument("--lambda-confidence", type=float, default=0.0044)
    parser.add_argument("--landmark-hours", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn-drop", type=float, default=0.2)
    parser.add_argument("--gat-heads", type=int, default=8)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-teacher-cache", action="store_true")
    parser.add_argument("--experiment-name")
    parser.add_argument("--data-root")
    parser.add_argument("--pkl-path")
    parser.add_argument("--los-labels-path")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--settings-root", "--settings_root", default="settings")
    parser.add_argument("--model-var2ind-path")
    parser.add_argument("--pretrained-var2ind-path")
    parser.add_argument("--teacher-checkpoint-root", default="checkpoints/los")
    parser.add_argument("--teacher-cache-root", default="checkpoints/los_teacher_cache")
    parser.add_argument("--checkpoint-root", default="checkpoints/los_distilled")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Re-evaluate existing distilled fold checkpoints on test without training")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_epoch < 1 or args.teacher_repeats < 1:
        raise ValueError("max-epoch and teacher-repeats must be positive")
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("max-folds must be positive")
    if args.batch_size < 1 or args.teacher_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if not 0 <= args.distillation_weight <= 1:
        raise ValueError("distillation-weight must be in [0, 1]")
    if args.temperature <= 0 or args.ordinal_loss_weight < 0:
        raise ValueError("temperature must be positive and ordinal weight nonnegative")
    if args.lambda_confidence < 0:
        raise ValueError("lambda-confidence must be nonnegative")
    if args.experiment_name and ("/" in args.experiment_name or "\\" in args.experiment_name):
        raise ValueError("experiment-name must be a single directory name")
    args.device_object = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    run(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main()
