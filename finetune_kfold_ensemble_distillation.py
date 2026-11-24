"""Distill historical repeat-5 logit ensembles into one fine-tuned model.

The data protocol and student architecture follow ``finetune_kfold.py``.  For
each fold, five existing checkpoints are loaded one at a time and evaluated on
the fold's train, validation, and hold-out test sets.  Their mean logits form
the teacher target.  A fresh single-model student is then trained with

    (1 - alpha) * BCE(student_logit, label)
    + alpha * temperature**2
      * BCE(student_logit / temperature,
            sigmoid(mean_teacher_logit / temperature)).

Only validation AUROC + AUPRC selects the student checkpoint.  The test set is
evaluated after selection.  Teacher logits are cached per fold so reruns with
different distillation hyperparameters do not repeat the five teacher passes.

Historical repeat checkpoints use dataset-specific prefixes such as
``mm3_gat_ft_pool_*``, ``mm4_gat_ft_pool_*``, and ``eicu_gat_ft_pool_*``.
MIMIC-IV and eICU keep their 124 graph-node IDs and 129 embedding rows, as
in ``finetune_kfold.py``.

The five teachers are never resident on the GPU together.  AMP is disabled.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
)
from evaluator import trapezoidal_pr_auc
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import finetune_kfold as fold_base


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("ensemble_distillation_kfold")
DISTILLATION_POOL_MODES = ("type", "time", "node_time", "diff", "diff_pool")


@dataclass(frozen=True)
class SplitTeacherLogits:
    stay_ids: np.ndarray
    labels: np.ndarray
    member_logits: np.ndarray

    @property
    def mean_logits(self) -> np.ndarray:
        return self.member_logits.mean(axis=0)


class _FoldLoggerAdapter:
    @staticmethod
    def log(message: Any) -> None:
        LOGGER.info("%s", message)


def relative_to_project(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_pool_mode(pool_mode: str) -> str:
    return "diff" if pool_mode == "diff_pool" else pool_mode


def expected_forward_pool_mode(pool_mode: str) -> str:
    """Map a CLI/checkpoint alias to the mode passed to model.forward()."""
    return "diff_pool" if pool_mode in ("diff", "diff_pool") else pool_mode


def structural_confidence_weight(args: argparse.Namespace) -> float:
    if expected_forward_pool_mode(args.pool_mode) == "diff_pool":
        return args.lambda_confidence
    return 0.0


def experiment_tag(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return str(args.experiment_name)

    def number(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    tag = (
        f"w{number(args.distillation_weight)}_"
        f"t{number(args.temperature)}_lr{number(args.student_lr)}"
    )
    confidence_weight = structural_confidence_weight(args)
    if confidence_weight > 0:
        tag += f"_conf{number(confidence_weight)}"
    return tag


def configure_logging(
    output_root: Path, dataset: str, pool_mode: str, run_tag: str
) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_dir / f"{dataset}_{pool_mode}_ensemble_distillation_{run_tag}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    logits = np.asarray(logits).reshape(-1).astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    try:
        auc = float(roc_auc_score(labels, probabilities))
        prauc = float(trapezoidal_pr_auc(labels, probabilities))
        prauc_trapezoid = prauc
    except ValueError:
        auc = 0.5
        prauc = 0.0
        prauc_trapezoid = 0.0
    return {
        "auc": auc,
        # Training, checkpoint selection, and evaluation share this definition.
        "prauc": prauc,
        "prauc_trapezoid": prauc_trapezoid,
        "objective": auc + prauc,
    }


def batch_stay_ids(batch: Any) -> list[int]:
    value = getattr(batch, "ts_id", None)
    if value is None:
        raise AttributeError("Distillation requires batch.ts_id")
    if isinstance(value, torch.Tensor):
        items = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    return [int(item) for item in items]


def forward_distillation_model(
    model: nn.Module,
    batch: Any,
    model_args: SimpleNamespace,
) -> torch.Tensor:
    """Run one distillation forward pass and verify node-level tokenization.

    ``node_time`` must send every encoded graph node to the Transformer.  The
    model implements this with ``to_dense_batch(enc, batch)``; on the first
    batch for each model instance, a lightweight pre-hook verifies that the
    Transformer sequence length equals the largest graph-node count in that
    batch.  Later batches use the same forward path without hook overhead.
    """

    mode = str(model_args.pool_mode)
    if mode != "node_time":
        return model(batch, mode=mode)

    if not hasattr(batch, "x_time"):
        raise AttributeError("node_time distillation requires batch.x_time")
    if not hasattr(batch, "batch"):
        raise AttributeError("node_time distillation requires batch.batch")
    node_count = int(batch.batch.numel())
    if int(batch.x_time.size(0)) != node_count:
        raise RuntimeError(
            "node_time requires one timestamp per graph node: "
            f"nodes={node_count}, timestamps={int(batch.x_time.size(0))}"
        )

    if bool(getattr(model, "_node_time_tokenization_verified", False)):
        return model(batch, mode=mode)

    observed: dict[str, tuple[int, ...]] = {}

    def capture_transformer_input(_module, inputs):
        if inputs:
            observed["shape"] = tuple(int(value) for value in inputs[0].shape)

    handle = model.transformer.register_forward_pre_hook(capture_transformer_input)
    try:
        output = model(batch, mode=mode)
    finally:
        handle.remove()

    num_graphs = int(batch.num_graphs)
    counts = torch.bincount(batch.batch.long(), minlength=num_graphs)
    max_nodes = int(counts.max().item()) if counts.numel() else 0
    shape = observed.get("shape")
    if shape is None or len(shape) != 3:
        raise RuntimeError(
            "Unable to verify node_time Transformer input; expected [B, N, D], "
            f"observed={shape}"
        )
    if shape[0] != num_graphs or shape[1] != max_nodes:
        raise RuntimeError(
            "node_time did not pass all graph nodes as Transformer tokens: "
            f"observed={shape[:2]}, expected=({num_graphs}, {max_nodes})"
        )
    model._node_time_tokenization_verified = True
    LOGGER.info(
        "Verified node_time tokenization: batch=%d max_nodes=%d hidden=%d",
        shape[0],
        shape[1],
        shape[2],
    )
    return output


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(payload)}")
    return dict(payload)


def teacher_checkpoint_paths(
    args: argparse.Namespace,
    fold_number: int,
) -> list[Path]:
    mode = checkpoint_pool_mode(args.pool_mode)
    output_prefix = fold_base.DATASET_CONFIGS[args.dataset]["output_prefix"]
    paths = []
    for turn in range(1, args.teacher_repeats + 1):
        directory = args.teacher_directory_template.format(
            dataset=args.dataset,
            output_prefix=output_prefix,
            pool_mode=mode,
            fold=fold_number,
            turn=turn,
        )
        filename = args.teacher_filename_template.format(
            dataset=args.dataset,
            output_prefix=output_prefix,
            pool_mode=mode,
            fold=fold_number,
            turn=turn,
        )
        path = args.checkpoint_root / directory / filename
        paths.append(path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing repeat teacher checkpoints:\n" + "\n".join(missing)
        )
    return paths


def checkpoint_signature(paths: Sequence[Path]) -> str:
    entries = []
    for path in paths:
        stat = path.stat()
        entries.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    encoded = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def infer_num_clusters(paths: Sequence[Path]) -> int:
    inferred: list[int] = []
    for path in paths:
        state = load_checkpoint_state(path)
        key = "diffpool_proj1.weight"
        if key not in state:
            raise KeyError(f"Teacher checkpoint has no {key!r}: {path}")
        inferred.append(int(state[key].shape[0]))
        del state
    unique = sorted(set(inferred))
    if len(unique) != 1:
        raise RuntimeError(
            f"Teacher checkpoints disagree on DiffPool capacity: {inferred}"
        )
    return unique[0]


def prepare_model_context(
    pretrain_params: Mapping[str, Any],
    finetune_params: Mapping[str, Any],
    args: argparse.Namespace,
    dataset_config: SimpleNamespace,
) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    """Obtain the current dataset-aware pretrained encoder and model args."""

    args.task_name = (
        f"{dataset_config.output_prefix}_prepare_distillation_"
        f"{checkpoint_pool_mode(args.pool_mode)}"
    )
    set_seed(args.seed)
    temporary_model, model_args = fold_base.build_finetune_models(
        pretrain_params,
        finetune_params,
        args,
        dataset_config,
    )
    pretrained_backbone = {
        key: value.detach().cpu().clone()
        for key, value in temporary_model.pre_model.state_dict().items()
    }
    if isinstance(getattr(model_args, "variable_to_class", None), torch.Tensor):
        model_args.variable_to_class = model_args.variable_to_class.detach().cpu()
    del temporary_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_args, pretrained_backbone


def build_single_model(
    model_args: SimpleNamespace,
    pretrained_backbone: Mapping[str, torch.Tensor] | None,
    num_clusters: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    args_copy = SimpleNamespace(**vars(model_args))
    model = fold_base.HierarchicalEHRModel(
        input_dim=1,
        hidden_dim=int(args_copy.num_hidden),
        args=args_copy,
        num_classes_type=int(args_copy.num_classes_type),
        num_clusters_k=int(num_clusters),
        num_demo=2,
        num_heads=int(args_copy.num_heads),
        num_layers=int(args_copy.num_layers),
        dropout=float(dropout),
        aggr_type=str(args_copy.aggr_type),
    )
    if pretrained_backbone is not None:
        result = model.pre_model.load_state_dict(pretrained_backbone, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Pretrained encoder incompatibility: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
    return model.to(device)


@torch.no_grad()
def collect_model_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_args: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    stay_ids: list[int] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        batch = fold_base.normalize_collated_group_ids(batch, model_args)
        output = forward_distillation_model(model, batch, model_args)
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError(
                f"Non-finite teacher logits for stay IDs {batch_stay_ids(batch)}"
            )
        stay_ids.extend(batch_stay_ids(batch))
        labels.append(batch.y.detach().view(-1).cpu())
        logits.append(output.detach().float().view(-1).cpu())
    id_array = np.asarray(stay_ids, dtype=np.int64)
    if np.unique(id_array).size != id_array.size:
        raise RuntimeError("A teacher prediction split contains duplicate stay IDs")
    return (
        id_array,
        torch.cat(labels).numpy().astype(np.int64),
        torch.cat(logits).numpy().astype(np.float32),
    )


def save_teacher_cache(
    path: Path,
    signature: str,
    predictions: Mapping[str, SplitTeacherLogits],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"checkpoint_signature": np.asarray(signature)}
    for split_name, split in predictions.items():
        arrays[f"{split_name}_ids"] = split.stay_ids
        arrays[f"{split_name}_labels"] = split.labels
        arrays[f"{split_name}_member_logits"] = split.member_logits
    np.savez_compressed(path, **arrays)


def load_teacher_cache(
    path: Path,
    signature: str,
    expected_ids: Mapping[str, Sequence[int]],
    expected_members: int,
) -> dict[str, SplitTeacherLogits] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            cached_signature = str(payload["checkpoint_signature"].item())
            if cached_signature != signature:
                return None
            result: dict[str, SplitTeacherLogits] = {}
            for split_name, split_ids in expected_ids.items():
                split = SplitTeacherLogits(
                    stay_ids=payload[f"{split_name}_ids"].copy(),
                    labels=payload[f"{split_name}_labels"].copy(),
                    member_logits=payload[
                        f"{split_name}_member_logits"
                    ].copy(),
                )
                if split.member_logits.shape != (
                    expected_members,
                    split.stay_ids.size,
                ):
                    return None
                if not np.array_equal(
                    np.sort(split.stay_ids),
                    np.sort(np.asarray(split_ids, dtype=np.int64)),
                ):
                    return None
                result[split_name] = split
            return result
    except (KeyError, OSError, ValueError):
        return None


def compute_teacher_logits(
    checkpoint_paths: Sequence[Path],
    loaders: Mapping[str, DataLoader],
    model_args: SimpleNamespace,
    num_clusters: int,
    dropout: float,
    device: torch.device,
) -> dict[str, SplitTeacherLogits]:
    """Evaluate one teacher at a time and retain only CPU logits."""

    model = build_single_model(
        model_args,
        pretrained_backbone=None,
        num_clusters=num_clusters,
        dropout=dropout,
        device=device,
    )
    reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    member_logits: dict[str, list[np.ndarray]] = {
        split_name: [] for split_name in loaders
    }
    try:
        for member_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            state = load_checkpoint_state(checkpoint_path)
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Teacher checkpoint is incompatible with the inferred "
                    f"{num_clusters}-cluster model: {checkpoint_path}\n{error}"
                ) from error
            del state
            for split_name, loader in loaders.items():
                ids, labels, logits = collect_model_logits(
                    model,
                    loader,
                    device,
                    model_args,
                )
                if split_name not in reference:
                    reference[split_name] = (ids, labels)
                else:
                    reference_ids, reference_labels = reference[split_name]
                    if not np.array_equal(ids, reference_ids) or not np.array_equal(
                        labels,
                        reference_labels,
                    ):
                        raise RuntimeError(
                            f"{split_name} order changed between teacher passes"
                        )
                member_logits[split_name].append(logits)
            LOGGER.info(
                "teacher %d/%d evaluated: %s",
                member_index,
                len(checkpoint_paths),
                checkpoint_path,
            )
        return {
            split_name: SplitTeacherLogits(
                stay_ids=reference[split_name][0],
                labels=reference[split_name][1],
                member_logits=np.stack(logit_list, axis=0),
            )
            for split_name, logit_list in member_logits.items()
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_teacher_split(split: SplitTeacherLogits) -> dict[str, Any]:
    member_metrics = [
        classification_metrics(split.labels, split.member_logits[index])
        for index in range(split.member_logits.shape[0])
    ]
    summary: dict[str, Any] = {
        "repeat1": member_metrics[0],
        "avg5": classification_metrics(split.labels, split.mean_logits),
        "members": member_metrics,
    }
    return summary


def teacher_lookup(split: SplitTeacherLogits) -> dict[int, float]:
    return {
        int(stay_id): float(logit)
        for stay_id, logit in zip(split.stay_ids, split.mean_logits)
    }


@torch.no_grad()
def evaluate_student(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_args: SimpleNamespace,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    ids: list[int] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        batch = fold_base.normalize_collated_group_ids(batch, model_args)
        output = forward_distillation_model(model, batch, model_args)
        targets = batch.y.float().view(-1, 1)
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError(
                f"Non-finite student logits for stay IDs {batch_stay_ids(batch)}"
            )
        loss = criterion(output, targets)
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        ids.extend(batch_stay_ids(batch))
        labels.append(targets.view(-1).cpu())
        logits.append(output.float().view(-1).cpu())
    label_array = torch.cat(labels).numpy().astype(np.int64)
    logit_array = torch.cat(logits).numpy().astype(np.float32)
    metrics = classification_metrics(label_array, logit_array)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics, {
        "stay_ids": np.asarray(ids, dtype=np.int64),
        "labels": label_array,
        "logits": logit_array,
    }


def append_numerical_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def train_student(
    train_dataset: Any,
    val_loader: DataLoader,
    teacher_train: SplitTeacherLogits,
    model_args: SimpleNamespace,
    pretrained_backbone: Mapping[str, torch.Tensor],
    num_clusters: int,
    args: argparse.Namespace,
    device: torch.device,
    fold_number: int,
    seed: int,
    checkpoint_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, float], Path]:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = build_single_model(
        model_args,
        pretrained_backbone,
        num_clusters,
        args.dropout,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch,
    )
    hard_criterion = nn.BCEWithLogitsLoss()
    confidence_weight = structural_confidence_weight(args)
    target_by_stay = teacher_lookup(teacher_train)
    best_objective = -math.inf
    best_metrics: dict[str, float] = {}
    epoch_rows: list[dict[str, Any]] = []
    skipped_nonfinite_gradients = 0
    numerical_path = artifact_dir / "numerical_events.jsonl"

    try:
        for epoch in range(args.max_epoch):
            model.train()
            total_loss = 0.0
            total_hard_loss = 0.0
            total_distillation_loss = 0.0
            total_confidence_loss = 0.0
            total_examples = 0
            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=(
                    f"Fold {fold_number} distill "
                    f"epoch {epoch + 1}/{args.max_epoch}"
                ),
                leave=False,
            )
            for batch_index, batch in progress:
                batch = batch.to(device)
                batch = fold_base.normalize_collated_group_ids(batch, model_args)
                targets = batch.y.float().view(-1, 1)
                ids = batch_stay_ids(batch)
                try:
                    teacher_values = [target_by_stay[stay_id] for stay_id in ids]
                except KeyError as error:
                    raise KeyError(
                        f"No cached teacher logit for training stay {error.args[0]}"
                    ) from error
                teacher_logits = torch.as_tensor(
                    teacher_values,
                    dtype=targets.dtype,
                    device=device,
                ).view(-1, 1)

                optimizer.zero_grad(set_to_none=True)
                if confidence_weight > 0:
                    student_logits, confidence_loss = (
                        fold_base.forward_diffpool_with_confidence(
                            model,
                            batch,
                            model_args.pool_mode,
                        )
                    )
                else:
                    student_logits = forward_distillation_model(
                        model, batch, model_args
                    )
                    confidence_loss = student_logits.new_zeros(())

                hard_loss = hard_criterion(student_logits, targets)
                soft_targets = torch.sigmoid(
                    teacher_logits / args.temperature
                )
                distillation_loss = F.binary_cross_entropy_with_logits(
                    student_logits / args.temperature,
                    soft_targets,
                )
                loss = (
                    (1.0 - args.distillation_weight) * hard_loss
                    + args.distillation_weight
                    * (args.temperature**2)
                    * distillation_loss
                    + confidence_weight * confidence_loss
                )
                if not bool(torch.isfinite(loss).item()):
                    append_numerical_event(
                        numerical_path,
                        {
                            "fold": fold_number,
                            "seed": seed,
                            "epoch": epoch + 1,
                            "batch_index": batch_index,
                            "stay_ids": ids,
                            "student_logits_finite": bool(
                                torch.isfinite(student_logits).all().item()
                            ),
                            "teacher_logits_finite": bool(
                                torch.isfinite(teacher_logits).all().item()
                            ),
                            "hard_loss": float(hard_loss.detach().item()),
                            "distillation_loss": float(
                                distillation_loss.detach().item()
                            ),
                        },
                    )
                    raise FloatingPointError(
                        f"Non-finite distillation loss; see {numerical_path}"
                    )
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=0.5,
                        error_if_nonfinite=True,
                    )
                except RuntimeError:
                    skipped_nonfinite_gradients += 1
                    optimizer.zero_grad(set_to_none=True)
                    append_numerical_event(
                        numerical_path,
                        {
                            "fold": fold_number,
                            "seed": seed,
                            "epoch": epoch + 1,
                            "batch_index": batch_index,
                            "stay_ids": ids,
                            "phase": "backward",
                            "skipped_nonfinite_gradient_batches": (
                                skipped_nonfinite_gradients
                            ),
                        },
                    )
                    if (
                        skipped_nonfinite_gradients
                        > args.max_nonfinite_gradient_batches
                    ):
                        raise FloatingPointError(
                            "Exceeded --max_nonfinite_gradient_batches; see "
                            f"{numerical_path}"
                        )
                    continue
                optimizer.step()
                batch_size = int(targets.size(0))
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_hard_loss += float(hard_loss.detach().item()) * batch_size
                total_distillation_loss += (
                    float(distillation_loss.detach().item()) * batch_size
                )
                total_confidence_loss += (
                    float(confidence_loss.detach().item()) * batch_size
                )
            scheduler.step()

            validation, predictions = evaluate_student(
                model,
                val_loader,
                device,
                model_args,
            )
            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": total_loss / max(total_examples, 1),
                "train_hard_loss": total_hard_loss / max(total_examples, 1),
                "train_distillation_loss": total_distillation_loss
                / max(total_examples, 1),
                "train_confidence_loss": total_confidence_loss
                / max(total_examples, 1),
                "skipped_nonfinite_gradient_batches": (
                    skipped_nonfinite_gradients
                ),
            }
            row.update(flatten_metrics("validation", validation))
            epoch_rows.append(row)
            if validation["objective"] > best_objective:
                best_objective = float(validation["objective"])
                best_metrics = dict(validation)
                best_metrics["best_epoch"] = float(epoch + 1)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                write_json(
                    artifact_dir / "best_validation_metrics.json",
                    best_metrics,
                )
                np.savez_compressed(
                    artifact_dir / "best_validation_logits.npz",
                    **predictions,
                )
            LOGGER.info(
                "fold=%d epoch=%02d validation AUC/PRAUC=%.4f/%.4f",
                fold_number,
                epoch + 1,
                validation["auc"],
                validation["prauc"],
            )
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Student produced no checkpoint: {checkpoint_path}")
        write_rows(artifact_dir / "epoch_metrics.csv", epoch_rows)
        return best_metrics, checkpoint_path
    finally:
        del model, optimizer, scheduler, train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_selected_student(
    checkpoint_path: Path,
    loader: DataLoader,
    model_args: SimpleNamespace,
    num_clusters: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model = build_single_model(
        model_args,
        pretrained_backbone=None,
        num_clusters=num_clusters,
        dropout=args.dropout,
        device=device,
    )
    state = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    del state
    try:
        return evaluate_student(model, loader, device, model_args)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"fold_count": len(rows), "metrics": {}}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and key not in {"fold", "student_seed"}
        }
    )
    for key in numeric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            summary["metrics"][key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def load_completed_fold_rows(
    results_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    mode: str,
    run_tag: str,
) -> list[dict[str, Any]]:
    """Verify and retain completed folds preceding ``--start_fold``."""

    if args.start_fold == 1:
        return []
    metrics_path = results_root / "fold_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Cannot start at fold {args.start_fold}: missing prior results {metrics_path}"
        )
    with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "fold" not in reader.fieldnames:
            raise ValueError(f"Prior fold metrics have no fold column: {metrics_path}")
        prior_rows: dict[int, dict[str, Any]] = {}
        later_folds: list[int] = []
        for raw_row in reader:
            fold_number = int(raw_row["fold"])
            if fold_number >= args.start_fold:
                later_folds.append(fold_number)
                continue
            if fold_number in prior_rows:
                raise ValueError(f"Duplicate fold {fold_number} in {metrics_path}")
            if not 1 <= fold_number <= args.n_folds:
                raise ValueError(f"Invalid fold {fold_number} in {metrics_path}")
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if not key or value is None or value == "":
                    raise ValueError(
                        f"Incomplete metrics for fold {fold_number} in {metrics_path}"
                    )
                if key in {"fold", "student_seed", "num_clusters"}:
                    row[key] = int(value)
                elif key in {"student_checkpoint", "teacher_checkpoint_signature"}:
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            prior_rows[fold_number] = row

    required_folds = set(range(1, args.start_fold))
    if set(prior_rows) != required_folds:
        missing = sorted(required_folds - set(prior_rows))
        raise ValueError(
            f"Cannot start at fold {args.start_fold}: missing completed folds "
            f"{missing} in {metrics_path}"
        )

    summary_path = results_root / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)
        expected_settings = {
            "dataset": args.dataset,
            "pool_mode": mode,
            "experiment_name": run_tag,
            "n_folds_requested": args.n_folds,
            "distillation_weight": args.distillation_weight,
            "temperature": args.temperature,
            "student_lr": args.student_lr,
            "max_epoch": args.max_epoch,
            "lambda_confidence": structural_confidence_weight(args),
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
        }
        mismatched = {
            key: (previous_summary[key], expected)
            for key, expected in expected_settings.items()
            if key in previous_summary and previous_summary[key] != expected
        }
        if mismatched:
            raise ValueError(
                f"Existing summary uses different settings: {mismatched}. "
                "Use the original settings or a new --experiment_name."
            )
    else:
        LOGGER.warning(
            "No completed summary at %s; confirm skipped folds used the "
            "same training hyperparameters before combining their metrics.",
            summary_path,
        )

    for fold_number in sorted(prior_rows):
        row = prior_rows[fold_number]
        expected_seed = int(args.seed + (fold_number - 1) * 10_000)
        if row.get("student_seed") != expected_seed:
            raise ValueError(
                f"Fold {fold_number} seed is {row.get('student_seed')}, "
                f"expected {expected_seed}"
            )
        expected_checkpoint = (
            output_root
            / "checkpoints"
            / (
                f"{fold_base.DATASET_CONFIGS[args.dataset]['output_prefix']}_"
                f"gat_ft_pool_{mode}_distilled_{run_tag}_fold{fold_number}"
            )
            / f"checkpoint_fold{fold_number}_distilled_best.pt"
        )
        if (
            "student_checkpoint" not in row
            or Path(row["student_checkpoint"]).resolve()
            != expected_checkpoint.resolve()
            or not expected_checkpoint.is_file()
        ):
            raise FileNotFoundError(
                f"Fold {fold_number} student checkpoint is missing or mismatched: "
                f"{expected_checkpoint}"
            )
        signature = checkpoint_signature(teacher_checkpoint_paths(args, fold_number))
        if row.get("teacher_checkpoint_signature") != signature:
            raise ValueError(
                f"Fold {fold_number} teacher checkpoints differ from its saved metrics"
            )
        fold_dir = results_root / f"fold_{fold_number}"
        for artifact in (
            fold_dir / "teacher_metrics.json",
            fold_dir / "student" / "best_validation_logits.npz",
            fold_dir / "student" / "test_metrics.json",
            fold_dir / "student" / "test_logits.npz",
        ):
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"Fold {fold_number} is not complete; missing {artifact}"
                )
        for metric in ("student_test_auc", "student_test_prauc"):
            if metric not in row:
                raise ValueError(
                    f"Fold {fold_number} is missing {metric} in {metrics_path}"
                )
    if later_folds:
        LOGGER.warning(
            "Existing metrics for folds %s will be replaced as training "
            "restarts from fold %d.",
            sorted(set(later_folds)),
            args.start_fold,
        )
    LOGGER.info(
        "Verified and retained completed folds %s from %s",
        sorted(prior_rows),
        metrics_path,
    )
    return [prior_rows[fold_number] for fold_number in sorted(prior_rows)]


def run(args: argparse.Namespace) -> None:
    dataset_config = fold_base.resolve_dataset_config(args)
    fold_base.validate_required_paths(dataset_config)
    output_root = dataset_config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_tag = experiment_tag(args)
    configure_logging(
        output_root,
        args.dataset,
        checkpoint_pool_mode(args.pool_mode),
        run_tag,
    )
    fold_base.logger = _FoldLoggerAdapter()

    pretrain_params = fold_base.load_pretrain_args(
        dataset_config.pretrain_config_path,
        dataset_config.pretrain_profile,
    )
    finetune_params = fold_base.load_finetune_args(
        dataset_config.finetune_config_path
    )
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    model_args, pretrained_backbone = prepare_model_context(
        pretrain_params,
        finetune_params,
        args,
        dataset_config,
    )
    expected_mode = expected_forward_pool_mode(args.pool_mode)
    if str(model_args.pool_mode) != expected_mode:
        raise RuntimeError(
            "Fine-tuning settings resolved to the wrong model forward mode: "
            f"requested={args.pool_mode!r}, expected={expected_mode!r}, "
            f"configured={model_args.pool_mode!r}, "
            f"settings={dataset_config.finetune_config_path}"
        )
    # The internal forward mode comes from ft_param_*.json.  For the CLI alias
    # ``diff``, this is intentionally ``diff_pool``.
    LOGGER.info(
        "dataset=%s requested_pool=%s model_pool=%s device=%s AMP=False",
        args.dataset,
        args.pool_mode,
        model_args.pool_mode,
        device,
    )
    with dataset_config.var2ind_path.open("r", encoding="utf-8") as handle:
        data_vocabulary = json.load(handle)
    with dataset_config.model_var2ind_path.open("r", encoding="utf-8") as handle:
        model_vocabulary = json.load(handle)
    embedding_rows = int(pretrained_backbone["variable_emb.weight"].shape[0])
    if embedding_rows != len(model_vocabulary):
        raise RuntimeError(
            "Pretrained embedding capacity does not match the model vocabulary: "
            f"{embedding_rows} rows vs {len(model_vocabulary)} entries"
        )
    if max(map(int, data_vocabulary.values()), default=-1) >= embedding_rows:
        raise RuntimeError(
            "Dataset variable IDs exceed pretrained embedding capacity: "
            f"{dataset_config.var2ind_path} vs {embedding_rows} rows"
        )
    LOGGER.info(
        "pretrained_checkpoint=%s data_variable_ids=%d "
        "model_embedding_rows=%d data_vocabulary=%s model_vocabulary=%s",
        dataset_config.checkpoint_path,
        len(data_vocabulary),
        embedding_rows,
        dataset_config.var2ind_path,
        dataset_config.model_var2ind_path,
    )

    development_ids, test_ids, folds = fold_base.load_protocol_metadata(
        dataset_config,
        n_folds=args.n_folds,
        window_size=args.window_size,
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    del development_ids
    loader_options = {
        "batch_size": args.teacher_batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    test_dataset = fold_base.make_dataset(dataset_config, test_ids)
    teacher_test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )
    student_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    mode = checkpoint_pool_mode(args.pool_mode)
    experiment_root = output_root / "results" / (
        f"{args.dataset}_{mode}_ensemble_distillation_{args.n_folds}fold"
    )
    results_root = experiment_root / run_tag
    teacher_cache_root = experiment_root / "teacher_cache"
    results_root.mkdir(parents=True, exist_ok=True)
    fold_rows = load_completed_fold_rows(
        results_root, output_root, args, mode, run_tag
    )

    for fold_index, (train_ids, val_ids) in enumerate(folds):
        fold_number = fold_index + 1
        if fold_number < args.start_fold:
            LOGGER.info("Skipping completed fold %d/%d", fold_number, len(folds))
            continue
        student_seed = int(args.seed + fold_index * 10_000)
        fold_dir = results_root / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths = teacher_checkpoint_paths(args, fold_number)
        signature = checkpoint_signature(checkpoint_paths)
        num_clusters = infer_num_clusters(checkpoint_paths)
        LOGGER.info(
            "========== fold=%d/%d seed=%d clusters=%d train=%d val=%d test=%d ==========",
            fold_number,
            len(folds),
            student_seed,
            num_clusters,
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

        train_dataset = fold_base.make_dataset(dataset_config, train_ids)
        val_dataset = fold_base.make_dataset(dataset_config, val_ids)
        teacher_train_loader = DataLoader(
            train_dataset,
            shuffle=False,
            **loader_options,
        )
        teacher_val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_options,
        )
        student_val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        expected_ids = {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids,
        }
        cache_path = (
            teacher_cache_root
            / f"fold_{fold_number}"
            / "teacher_repeat5_logits.npz"
        )
        teacher_predictions = None
        if args.reuse_teacher_cache:
            teacher_predictions = load_teacher_cache(
                cache_path,
                signature,
                expected_ids,
                args.teacher_repeats,
            )
        if teacher_predictions is None:
            teacher_predictions = compute_teacher_logits(
                checkpoint_paths,
                {
                    "train": teacher_train_loader,
                    "validation": teacher_val_loader,
                    "test": teacher_test_loader,
                },
                model_args,
                num_clusters,
                args.dropout,
                device,
            )
            save_teacher_cache(
                cache_path,
                signature,
                teacher_predictions,
            )
        else:
            LOGGER.info("Reused teacher-logit cache: %s", cache_path)

        teacher_validation = summarize_teacher_split(
            teacher_predictions["validation"]
        )

        checkpoint_dir = output_root / "checkpoints" / (
            f"{dataset_config.output_prefix}_gat_ft_pool_{mode}_"
            f"distilled_{run_tag}_fold{fold_number}"
        )
        student_checkpoint = checkpoint_dir / (
            f"checkpoint_fold{fold_number}_distilled_best.pt"
        )
        student_validation, student_checkpoint = train_student(
            train_dataset,
            student_val_loader,
            teacher_predictions["train"],
            model_args,
            pretrained_backbone,
            num_clusters,
            args,
            device,
            fold_number,
            student_seed,
            student_checkpoint,
            fold_dir / "student",
        )
        student_test, student_test_predictions = evaluate_selected_student(
            student_checkpoint,
            student_test_loader,
            model_args,
            num_clusters,
            args,
            device,
        )
        # Test metrics are deliberately materialized only after validation has
        # selected the student checkpoint.  Cached test logits never enter the
        # loss, early stopping, or hyperparameter selection.
        teacher_test = summarize_teacher_split(teacher_predictions["test"])
        write_json(
            fold_dir / "teacher_metrics.json",
            {"validation": teacher_validation, "test": teacher_test},
        )
        write_json(fold_dir / "student" / "test_metrics.json", student_test)
        np.savez_compressed(
            fold_dir / "student" / "test_logits.npz",
            **student_test_predictions,
        )

        row: dict[str, Any] = {
            "fold": fold_number,
            "student_seed": student_seed,
            "num_clusters": num_clusters,
            "student_checkpoint": str(student_checkpoint.resolve()),
            "teacher_checkpoint_signature": signature,
        }
        row.update(
            flatten_metrics(
                "repeat1_validation",
                teacher_validation["repeat1"],
            )
        )
        row.update(
            flatten_metrics("repeat1_test", teacher_test["repeat1"])
        )
        row.update(
            flatten_metrics("teacher_avg5_validation", teacher_validation["avg5"])
        )
        row.update(flatten_metrics("teacher_avg5_test", teacher_test["avg5"]))
        row.update(flatten_metrics("student_validation", student_validation))
        row.update(flatten_metrics("student_test", student_test))
        for metric_name in ("auc", "prauc", "prauc_trapezoid", "objective"):
            row[f"student_gain_over_repeat1_{metric_name}"] = float(
                student_test[metric_name]
                - teacher_test["repeat1"][metric_name]
            )
            row[f"student_gap_to_teacher_avg5_{metric_name}"] = float(
                student_test[metric_name]
                - teacher_test["avg5"][metric_name]
            )
        fold_rows.append(row)
        write_rows(results_root / "fold_metrics.csv", fold_rows)
        LOGGER.info(
            "fold=%d test repeat1=%.4f/%.4f teacher_avg5=%.4f/%.4f "
            "student=%.4f/%.4f",
            fold_number,
            teacher_test["repeat1"]["auc"],
            teacher_test["repeat1"]["prauc"],
            teacher_test["avg5"]["auc"],
            teacher_test["avg5"]["prauc"],
            student_test["auc"],
            student_test["prauc"],
        )

        del (
            train_dataset,
            val_dataset,
            teacher_train_loader,
            teacher_val_loader,
            student_val_loader,
            teacher_predictions,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate_rows(fold_rows)
    summary.update(
        {
            "dataset": args.dataset,
            "pool_mode": mode,
            "experiment_name": run_tag,
            "n_folds_requested": int(args.n_folds),
            "n_folds_completed": len(fold_rows),
            "start_fold": int(args.start_fold),
            "n_folds_reused": int(args.start_fold - 1),
            "n_folds_trained_this_run": int(len(folds) - args.start_fold + 1),
            "teacher_repeats": int(args.teacher_repeats),
            "teacher_aggregation": "mean_logits",
            "student_inference_models": 1,
            "distillation_weight": float(args.distillation_weight),
            "lambda_confidence": structural_confidence_weight(args),
            "temperature": float(args.temperature),
            "student_lr": float(args.student_lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "max_epoch": int(args.max_epoch),
            "amp": False,
        }
    )
    write_json(results_root / "summary.json", summary)
    for metric_name in (
        "repeat1_test_auc",
        "repeat1_test_prauc",
        "teacher_avg5_test_auc",
        "teacher_avg5_test_prauc",
        "teacher_avg5_test_prauc_trapezoid",
        "student_test_auc",
        "student_test_prauc",
        "student_test_prauc_trapezoid",
        "student_gain_over_repeat1_auc",
        "student_gain_over_repeat1_prauc",
    ):
        statistics = summary["metrics"].get(metric_name)
        if statistics:
            LOGGER.info(
                "%s: %.4f +/- %.4f",
                metric_name,
                statistics["mean"],
                statistics["std"],
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Distill five historical repeat checkpoints into one k-fold student."
        ),
    )
    parser.add_argument(
        "--dataset", choices=("mimic3", "mimic4", "eicu"), default="mimic3"
    )
    parser.add_argument(
        "--pool_mode",
        choices=DISTILLATION_POOL_MODES,
        default="type",
    )
    parser.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--teacher_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n_folds", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--max_folds", type=int)
    parser.add_argument(
        "--start_fold",
        type=int,
        default=1,
        help="Start training at this 1-based fold; verify and retain earlier results.",
    )
    parser.add_argument("--teacher_repeats", type=int, default=5)
    parser.add_argument("--max_epoch", type=int, default=15)
    parser.add_argument("--student_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--distillation_weight", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--lambda_confidence", type=float, default=fold_base.DIFFPOOL_LAMBDA_CONFIDENCE,
        help="Assignment entropy weight for the structural view.",
    )
    parser.add_argument(
        "--experiment_name",
        help=(
            "Optional output/checkpoint tag. By default it is derived from "
            "distillation weight, temperature, learning rate, and structural "
            "confidence weight when applicable."
        ),
    )
    parser.add_argument(
        "--max_nonfinite_gradient_batches",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--reuse_teacher_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher_directory_template",
        default="{output_prefix}_gat_ft_pool_{pool_mode}_fold{fold}_turn{turn}_auc",
        help="Teacher directory pattern; output_prefix is mm3, mm4, or eicu.",
    )
    parser.add_argument(
        "--teacher_filename_template",
        default="checkpoint_fold{fold}_turn{turn}_best.pt",
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--settings_root", "--settings-root", type=Path, default=Path("settings"))
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument("--output_root", type=Path, default=Path("."))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset not in ("mimic3", "mimic4", "eicu"):
        raise ValueError("Distillation supports mimic3, mimic4, and eicu")
    if args.pool_mode not in DISTILLATION_POOL_MODES:
        raise ValueError(
            "Unsupported distillation pool mode: "
            f"{args.pool_mode!r}; choices={DISTILLATION_POOL_MODES}"
        )
    if args.n_folds < 2:
        raise ValueError("--n_folds must be at least 2")
    if not 1 <= args.window_size < args.n_folds:
        raise ValueError("--window_size must lie in [1, n_folds)")
    if args.max_folds is not None and not 1 <= args.max_folds <= args.n_folds:
        raise ValueError("--max_folds must lie in [1, n_folds]")
    final_fold = args.max_folds or args.n_folds
    if not 1 <= args.start_fold <= final_fold:
        raise ValueError("--start_fold must lie in [1, max_folds or n_folds]")
    if args.teacher_repeats != 5:
        raise ValueError("This repeat5 distillation protocol requires 5 teachers")
    if args.batch_size < 1 or args.teacher_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")
    if args.max_epoch < 1:
        raise ValueError("--max_epoch must be positive")
    if args.student_lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer parameters")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must lie in [0, 1)")
    if not 0 <= args.distillation_weight <= 1:
        raise ValueError("--distillation_weight must lie in [0, 1]")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.lambda_confidence < 0:
        raise ValueError("--lambda_confidence cannot be negative")
    if args.experiment_name:
        name = str(args.experiment_name)
        if Path(name).name != name or any(character in name for character in "/\\"):
            raise ValueError("--experiment_name must be a single directory-safe name")
    if args.max_nonfinite_gradient_batches < 0:
        raise ValueError("--max_nonfinite_gradient_batches cannot be negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.data_root = relative_to_project(args.data_root)
    args.settings_root = relative_to_project(args.settings_root)
    args.checkpoint_root = relative_to_project(args.checkpoint_root)
    args.output_root = relative_to_project(args.output_root)
    print(args)
    run(args)


if __name__ == "__main__":
    main()
