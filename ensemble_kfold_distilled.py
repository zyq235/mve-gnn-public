"""Evaluate distilled single-view models and their multi-view ensemble.

This script is the distilled-model counterpart of ``ensemble_kfold_avg.py``.
The original script first averages five repeat logits within each view and
then averages probabilities across type, time, and DiffPool views.  A
distilled checkpoint already compresses one repeat-5 teacher into a single
student, so this evaluator removes only the within-view repeat loop while
preserving the cross-view soft/hard ensemble and its reported metrics.

Each view is loaded, evaluated, and released sequentially.  Thus at most one
full fine-tuned model is resident on the GPU.  The script never changes the
original evaluator or any checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    auc as sklearn_auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import finetune_kfold as fold_base
import finetune_kfold_ensemble_distillation as distill_base


PROJECT_ROOT = Path(__file__).resolve().parent
VIEW_CHOICES = ("type", "time", "diff", "node_time")
FORWARD_VIEW = {
    "type": "type",
    "time": "time",
    "diff": "diff_pool",
    "node_time": "node_time",
}
LOGGER = logging.getLogger("distilled_kfold_ensemble")


class _FoldLoggerAdapter:
    @staticmethod
    def log(message: Any) -> None:
        LOGGER.info("%s", message)


def relative_to_project(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_logging(output_root: Path, dataset: str, run_name: str) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    log_name = (
        f"ensemble_kfold_distilled_{run_name}.log"
        if dataset == "mimic3"
        else f"ensemble_kfold_distilled_{dataset}_{run_name}.log"
    )
    file_handler = logging.FileHandler(log_dir / log_name, encoding="utf-8")
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
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }


def precision_at_recall(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )
    eligible = np.where(recall >= target_recall)[0]
    if eligible.size == 0:
        return 0.0, 1.0
    index = int(eligible[-1])
    threshold = float(thresholds[index]) if index < thresholds.size else 1.0
    return float(precision[index]), threshold


def evaluation_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    probabilities = np.asarray(probabilities).reshape(-1).astype(np.float64)
    if predictions is None:
        predictions = (probabilities > 0.5).astype(np.int64)
    else:
        predictions = np.asarray(predictions).reshape(-1).astype(np.int64)
    precision_curve, recall_curve, _ = precision_recall_curve(
        labels,
        probabilities,
    )
    if np.unique(labels).size < 2:
        roc_auc = 0.5
    else:
        roc_auc = float(roc_auc_score(labels, probabilities))
    pr_auc = float(sklearn_auc(recall_curve, precision_curve))
    min_rp = float(np.minimum(precision_curve, recall_curve).max())
    p_at_r08, threshold_r08 = precision_at_recall(
        labels,
        probabilities,
        0.8,
    )
    p_at_r09, threshold_r09 = precision_at_recall(
        labels,
        probabilities,
        0.9,
    )
    return {
        "auc": roc_auc,
        "prauc": pr_auc,
        "prauc_trapezoid": pr_auc,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "p_at_recall_0.8": p_at_r08,
        "p_at_recall_0.9": p_at_r09,
        "threshold_at_recall_0.8": threshold_r08,
        "threshold_at_recall_0.9": threshold_r09,
        "min_rp": min_rp,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def batch_stay_ids(batch: Any) -> list[int]:
    value = getattr(batch, "ts_id", None)
    if value is None:
        raise AttributeError("Evaluation requires batch.ts_id")
    if isinstance(value, torch.Tensor):
        items = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    return [int(item) for item in items]


def make_size_aware_batches(dataset: Any, max_stays: int, max_nodes: int) -> list[list[int]]:
    """Keep test order while isolating exceptionally large patient graphs."""
    if max_stays < 1 or max_nodes < 1:
        raise ValueError("max_stays and max_nodes must be positive")
    batches: list[list[int]] = []
    current: list[int] = []
    current_nodes = 0
    for index in range(len(dataset)):
        nodes = int(dataset[index].num_nodes)
        if nodes < 0:
            raise ValueError(f"Graph {index} has a negative node count")
        if current and (len(current) >= max_stays or current_nodes + nodes > max_nodes):
            batches.append(current)
            current = []
            current_nodes = 0
        current.append(index)
        current_nodes += nodes
    if current:
        batches.append(current)
    if [index for batch in batches for index in batch] != list(range(len(dataset))):
        raise RuntimeError("Size-aware batching changed the test sample order")
    return batches


def view_experiment_name(args: argparse.Namespace, view: str) -> str:
    override = getattr(args, f"{view}_experiment_name", None)
    if override or args.experiment_name:
        return str(override or args.experiment_name)
    student_args = distill_base.build_parser().parse_args(["--pool_mode", view])
    return distill_base.experiment_tag(student_args)


def distilled_checkpoint_path(
    args: argparse.Namespace,
    view: str,
    fold_number: int,
) -> Path:
    experiment = view_experiment_name(args, view)
    dataset_prefix = fold_base.DATASET_CONFIGS[args.dataset]["output_prefix"]
    relative = args.checkpoint_template.format(
        dataset=args.dataset,
        dataset_prefix=dataset_prefix,
        mode=view,
        experiment=experiment,
        fold=fold_number,
    )
    return args.checkpoint_root / relative


def prepare_view_context(
    args: argparse.Namespace,
    view: str,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build the same dataset-aware model arguments as distillation training."""

    view_args = argparse.Namespace(**vars(args))
    view_args.pool_mode = view
    pretrained_root = getattr(args, "pretrained_checkpoint_root", None)
    if pretrained_root is not None:
        view_args.checkpoint_root = pretrained_root
    dataset_config = fold_base.resolve_dataset_config(view_args)
    fold_base.validate_required_paths(dataset_config)
    pretrain_params = fold_base.load_pretrain_args(
        dataset_config.pretrain_config_path,
        dataset_config.pretrain_profile,
    )
    finetune_params = fold_base.load_finetune_args(
        dataset_config.finetune_config_path
    )
    model_args, pretrained_backbone = distill_base.prepare_model_context(
        pretrain_params,
        finetune_params,
        view_args,
        dataset_config,
    )
    del pretrained_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    expected_forward = FORWARD_VIEW[view]
    if model_args.pool_mode != expected_forward:
        raise RuntimeError(
            f"View {view!r} resolved to model mode {model_args.pool_mode!r}; "
            f"expected {expected_forward!r}"
        )
    return model_args, dataset_config


def load_distilled_model(
    checkpoint_path: Path,
    model_args: SimpleNamespace,
    device: torch.device,
    dropout: float,
) -> tuple[nn.Module, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    num_clusters = distill_base.infer_num_clusters([checkpoint_path])
    state = distill_base.load_checkpoint_state(checkpoint_path)
    build_args = model_args
    if model_args.pool_mode == "time":
        group_weight = state.get("group_embeddings.weight")
        if group_weight is None or group_weight.ndim != 2:
            raise RuntimeError(
                f"Time checkpoint has no 2-D group_embeddings.weight: "
                f"{checkpoint_path}"
            )
        checkpoint_groups = int(group_weight.shape[0])
        if checkpoint_groups < num_clusters:
            raise RuntimeError(
                f"Time checkpoint has {checkpoint_groups} group embeddings, "
                f"fewer than its {num_clusters} DiffPool clusters: {checkpoint_path}"
            )
        current_groups = max(int(model_args.num_classes_type), num_clusters)
        if checkpoint_groups != current_groups:
            # Time pooling uses time_proj, not group_embeddings. Recreate the
            # checkpoint's unused table exactly so strict loading stays intact.
            build_args = SimpleNamespace(**vars(model_args))
            build_args.num_classes_type = checkpoint_groups
            LOGGER.info(
                "Using time checkpoint group-embedding capacity %d instead "
                "of current metadata capacity %d: %s",
                checkpoint_groups,
                current_groups,
                checkpoint_path,
            )
    model = distill_base.build_single_model(
        build_args,
        pretrained_backbone=None,
        num_clusters=num_clusters,
        dropout=dropout,
        device=device,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Model checkpoint is incompatible with the inferred "
            f"{num_clusters}-cluster {model_args.pool_mode} model: "
            f"{checkpoint_path}\n{error}"
        ) from error
    del state
    model.eval()
    return model, num_clusters


@torch.no_grad()
def predict_view(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_args: SimpleNamespace,
    view: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    stay_ids: list[int] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    for batch in tqdm(loader, desc=f"Testing distilled {view}", leave=False):
        batch = batch.to(device)
        batch = fold_base.normalize_collated_group_ids(batch, model_args)
        output = model(batch, mode=model_args.pool_mode)
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError(
                f"Non-finite {view} logits for stay IDs {batch_stay_ids(batch)}"
            )
        stay_ids.extend(batch_stay_ids(batch))
        labels.append(batch.y.detach().view(-1).cpu())
        logits.append(output.detach().float().view(-1).cpu())
    id_array = np.asarray(stay_ids, dtype=np.int64)
    label_array = torch.cat(labels).numpy().astype(np.int64)
    logit_array = torch.cat(logits).numpy().astype(np.float32)
    probability_array = torch.sigmoid(torch.from_numpy(logit_array)).numpy()
    if np.unique(id_array).size != id_array.size:
        raise RuntimeError(f"View {view} produced duplicate test stay IDs")
    return id_array, label_array, logit_array, probability_array


def normalized_view_weights(
    views: Sequence[str],
    values: Sequence[float] | None,
) -> np.ndarray:
    if values is None:
        return np.full(len(views), 1.0 / len(views), dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (len(views),):
        raise ValueError(
            "--view_weights must contain exactly one value for each --views item"
        )
    if not np.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("--view_weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("At least one view weight must be positive")
    return weights / total


def combine_view_probabilities(
    view_probabilities: Mapping[str, np.ndarray],
    views: Sequence[str],
    ensemble_mode: str,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([view_probabilities[view] for view in views], axis=0)
    if ensemble_mode == "soft":
        probabilities = np.sum(stacked * weights[:, None], axis=0)
        predictions = (probabilities > 0.5).astype(np.int64)
    elif ensemble_mode == "hard":
        votes = (stacked > 0.5).astype(np.float64)
        # This reproduces the original evaluator's unweighted majority vote.
        probabilities = votes.mean(axis=0)
        predictions = (probabilities > 0.5).astype(np.int64)
    else:
        raise ValueError(f"Unknown ensemble mode: {ensemble_mode}")
    return probabilities, predictions


def correctness_subsets(
    labels: np.ndarray,
    stay_ids: np.ndarray,
    view_predictions: Mapping[str, np.ndarray],
) -> dict[str, list[int]]:
    required = {"type", "time", "diff"}
    if not required.issubset(view_predictions):
        return {"type_only_ids": [], "all_correct_ids": []}
    correct_type = view_predictions["type"] == labels
    correct_time = view_predictions["time"] == labels
    correct_diff = view_predictions["diff"] == labels
    type_only = correct_type & ~correct_time & ~correct_diff
    all_correct = correct_type & correct_time & correct_diff
    return {
        "type_only_ids": stay_ids[type_only].astype(np.int64).tolist(),
        "all_correct_ids": stay_ids[all_correct].astype(np.int64).tolist(),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"fold_count": len(rows), "metrics": {}}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and key != "fold"
        }
    )
    for key in numeric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows if key in row],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            result["metrics"][key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return result


def run(args: argparse.Namespace) -> None:
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    configure_logging(args.output_root, args.dataset, args.run_name)
    fold_base.logger = _FoldLoggerAdapter()
    contexts: dict[str, tuple[SimpleNamespace, SimpleNamespace]] = {}
    for view in args.views:
        contexts[view] = prepare_view_context(args, view)

    # Every view uses the selected dataset's processed test IDs.  Load them
    # from one resolved context and keep the original train/valid/test protocol.
    reference_config = contexts[args.views[0]][1]
    for view in args.views[1:]:
        candidate_config = contexts[view][1]
        for field in (
            "pkl_path",
            "var2ind_path",
            "var_mean_std_path",
            "model_var2ind_path",
        ):
            if getattr(candidate_config, field) != getattr(reference_config, field):
                raise RuntimeError(
                    f"View {view!r} uses a different {field}; "
                    "views cannot share one test split"
                )
    _, test_ids, _ = fold_base.load_protocol_metadata(
        reference_config,
        n_folds=args.n_folds,
        window_size=args.window_size,
    )
    test_dataset = fold_base.make_dataset(reference_config, test_ids)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.max_batch_nodes is None:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            **loader_options,
        )
    else:
        batches = make_size_aware_batches(
            test_dataset, args.batch_size, args.max_batch_nodes
        )
        LOGGER.info(
            "Size-aware batching: %d stays in %d contiguous batches "
            "(up to %d stays and %d nodes; oversized stays run alone)",
            len(test_dataset), len(batches), args.batch_size, args.max_batch_nodes,
        )
        test_loader = DataLoader(
            test_dataset, batch_sampler=batches, **loader_options,
        )
    weights = normalized_view_weights(args.views, args.view_weights)
    LOGGER.info(
        "dataset=%s views=%s weights=%s ensemble_mode=%s device=%s "
        "data_vocabulary=%s model_embedding_vocabulary=%s",
        args.dataset,
        list(args.views),
        weights.tolist(),
        args.ensemble_mode,
        device,
        reference_config.var2ind_path,
        reference_config.model_var2ind_path,
    )

    results_directory = (
        "ensemble_kfold_distilled"
        if args.dataset == "mimic3"
        else f"ensemble_kfold_distilled_{args.dataset}"
    )
    results_root = (
        args.output_root
        / "results"
        / results_directory
        / args.run_name
    )
    results_root.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    fold_count = args.max_folds or args.n_folds

    for fold_number in range(1, fold_count + 1):
        LOGGER.info(
            "==================== fold %d/%d ====================",
            fold_number,
            fold_count,
        )
        fold_dir = results_root / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        reference_ids: np.ndarray | None = None
        reference_labels: np.ndarray | None = None
        view_probabilities: dict[str, np.ndarray] = {}
        view_predictions: dict[str, np.ndarray] = {}
        view_metrics: dict[str, dict[str, Any]] = {}
        view_logits: dict[str, np.ndarray] = {}
        checkpoint_manifest: dict[str, Any] = {}

        for view in args.views:
            checkpoint_path = distilled_checkpoint_path(
                args,
                view,
                fold_number,
            )
            model_args = contexts[view][0]
            model, num_clusters = load_distilled_model(
                checkpoint_path,
                model_args,
                device,
                args.dropout,
            )
            try:
                ids, labels, logits, probabilities = predict_view(
                    model,
                    test_loader,
                    device,
                    model_args,
                    view,
                )
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if reference_ids is None:
                reference_ids = ids
                reference_labels = labels
            elif not np.array_equal(reference_ids, ids) or not np.array_equal(
                reference_labels,
                labels,
            ):
                raise RuntimeError(
                    f"Test sample order changed while evaluating view {view}"
                )
            predictions = (probabilities > 0.5).astype(np.int64)
            metrics = evaluation_metrics(labels, probabilities, predictions)
            view_probabilities[view] = probabilities
            view_predictions[view] = predictions
            view_metrics[view] = metrics
            view_logits[view] = logits
            checkpoint_manifest[view] = {
                "path": str(checkpoint_path.resolve()),
                "experiment_name": (
                    view_experiment_name(args, view)
                    if "{experiment}" in args.checkpoint_template else None
                ),
                "forward_mode": model_args.pool_mode,
                "num_clusters": num_clusters,
            }
            LOGGER.info(
                "fold=%d view=%s AUC/PRAUC=%.4f/%.4f checkpoint=%s",
                fold_number,
                view,
                metrics["auc"],
                metrics["prauc"],
                checkpoint_path,
            )

        if reference_ids is None or reference_labels is None:
            raise RuntimeError(f"Fold {fold_number} evaluated no views")
        ensemble_probabilities, ensemble_predictions = combine_view_probabilities(
            view_probabilities,
            args.views,
            args.ensemble_mode,
            weights,
        )
        ensemble_metrics = evaluation_metrics(
            reference_labels,
            ensemble_probabilities,
            ensemble_predictions,
        )
        subsets = correctness_subsets(
            reference_labels,
            reference_ids,
            view_predictions,
        )
        LOGGER.info(
            "fold=%d ensemble AUC/PRAUC=%.4f/%.4f "
            "P@R0.8=%.4f P@R0.9=%.4f minRP=%.4f",
            fold_number,
            ensemble_metrics["auc"],
            ensemble_metrics["prauc"],
            ensemble_metrics["p_at_recall_0.8"],
            ensemble_metrics["p_at_recall_0.9"],
            ensemble_metrics["min_rp"],
        )

        prediction_arrays: dict[str, Any] = {
            "stay_ids": reference_ids,
            "labels": reference_labels,
            "ensemble_probabilities": ensemble_probabilities,
            "ensemble_predictions": ensemble_predictions,
            "view_names": np.asarray(args.views),
            "view_weights": weights,
        }
        for view in args.views:
            prediction_arrays[f"{view}_logits"] = view_logits[view]
            prediction_arrays[f"{view}_probabilities"] = view_probabilities[view]
            prediction_arrays[f"{view}_predictions"] = view_predictions[view]
        np.savez_compressed(fold_dir / "test_predictions.npz", **prediction_arrays)
        write_json(fold_dir / "view_metrics.json", view_metrics)
        write_json(fold_dir / "ensemble_metrics.json", ensemble_metrics)
        write_json(fold_dir / "correctness_subsets.json", subsets)
        write_json(fold_dir / "checkpoint_manifest.json", checkpoint_manifest)

        row: dict[str, Any] = {"fold": fold_number}
        for view, metrics in view_metrics.items():
            row.update(flatten_metrics(view, metrics))
        row.update(flatten_metrics("ensemble", ensemble_metrics))
        row["type_only_count"] = len(subsets["type_only_ids"])
        row["all_correct_count"] = len(subsets["all_correct_ids"])
        fold_rows.append(row)
        write_rows(results_root / "fold_metrics.csv", fold_rows)

    summary = aggregate_rows(fold_rows)
    summary.update(
        {
            "dataset": args.dataset,
            "views": list(args.views),
            "view_weights": weights.tolist(),
            "ensemble_mode": args.ensemble_mode,
            "n_folds_requested": int(args.n_folds),
            "n_folds_completed": len(fold_rows),
            "checkpoint_template": args.checkpoint_template,
            "max_batch_nodes": args.max_batch_nodes,
            "view_experiments": {
                view: (
                    view_experiment_name(args, view)
                    if "{experiment}" in args.checkpoint_template else None
                ) for view in args.views
            },
        }
    )
    write_json(results_root / "summary.json", summary)
    for prefix in (*args.views, "ensemble"):
        LOGGER.info("-------------------- %s --------------------", prefix)
        for metric_name in (
            "auc",
            "prauc",
            "p_at_recall_0.8",
            "p_at_recall_0.9",
            "min_rp",
        ):
            statistics = summary["metrics"].get(f"{prefix}_{metric_name}")
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
            "Evaluate one distilled model per view and ensemble the views "
            "across six folds."
        ),
    )
    parser.add_argument(
        "--dataset", choices=("mimic3", "mimic4", "eicu"), default="mimic3"
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEW_CHOICES,
        default=["type", "time", "diff"],
    )
    parser.add_argument(
        "--ensemble_mode",
        choices=("soft", "hard"),
        default="soft",
    )
    parser.add_argument(
        "--view_weights",
        nargs="+",
        type=float,
        help=(
            "Soft-ensemble weights in the same order as --views. Values are "
            "normalized to sum to one."
        ),
    )
    parser.add_argument(
        "--experiment_name",
        help="Override the experiment tag generated by default student training.",
    )
    parser.add_argument("--type_experiment_name")
    parser.add_argument("--time_experiment_name")
    parser.add_argument("--diff_experiment_name")
    parser.add_argument("--node_time_experiment_name")
    parser.add_argument(
        "--checkpoint_template",
        default=(
            "{dataset_prefix}_gat_ft_pool_{mode}_distilled_{experiment}_"
            "fold{fold}/checkpoint_fold{fold}_distilled_best.pt"
        ),
        help=(
            "Path relative to --checkpoint_root. Available placeholders: "
            "dataset, dataset_prefix, mode, experiment, fold."
        ),
    )
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_batch_nodes", type=int,
        help="Optional total-node cap per test batch; large stays run alone.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--n_folds", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--max_folds", type=int)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint", type=Path)
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
        raise ValueError("Distilled checkpoints support mimic3, mimic4, and eicu")
    if not args.views or len(set(args.views)) != len(args.views):
        raise ValueError("--views must contain one or more unique views")
    if args.n_folds < 2:
        raise ValueError("--n_folds must be at least 2")
    if not 1 <= args.window_size < args.n_folds:
        raise ValueError("--window_size must lie in [1, n_folds)")
    if args.max_folds is not None and not 1 <= args.max_folds <= args.n_folds:
        raise ValueError("--max_folds must lie in [1, n_folds]")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if args.max_batch_nodes is not None and args.max_batch_nodes < 1:
        raise ValueError("--max_batch_nodes must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must lie in [0, 1)")
    if args.ensemble_mode == "hard" and args.view_weights is not None:
        raise ValueError("--view_weights applies only to soft ensemble mode")
    normalized_view_weights(args.views, args.view_weights)
    for name in (
        args.run_name,
        args.experiment_name,
        args.type_experiment_name,
        args.time_experiment_name,
        args.diff_experiment_name,
        args.node_time_experiment_name,
    ):
        if name is None:
            continue
        if not str(name) or Path(str(name)).name != str(name):
            raise ValueError(f"Unsafe output/experiment name: {name!r}")


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
