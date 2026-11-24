"""Small shared utilities for the GCT and GRAM experiment entry points."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from evaluator import trapezoidal_pr_auc
from torch import Tensor
from torch.utils.data import WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but is unavailable")
    return device


def move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def binary_metrics(labels: Iterable[float], probabilities: Iterable[float]) -> dict[str, float]:
    labels_array = np.asarray(list(labels), dtype=np.int64).reshape(-1)
    probability_array = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if labels_array.size == 0 or labels_array.size != probability_array.size:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    predictions = (probability_array >= 0.5).astype(np.int64)
    unique_labels = np.unique(labels_array)
    auc = (
        float(roc_auc_score(labels_array, probability_array))
        if unique_labels.size == 2
        else 0.5
    )
    prauc = (
        float(trapezoidal_pr_auc(labels_array, probability_array))
        if np.any(labels_array == 1)
        else 0.0
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions,
        average="binary",
        zero_division=0,
    )
    curve_precision, curve_recall, _ = precision_recall_curve(
        labels_array, probability_array
    )

    def precision_at_recall(target_recall: float) -> float:
        valid = np.where(curve_recall >= target_recall)[0]
        return float(curve_precision[valid[-1]]) if valid.size else 0.0

    return {
        "auc": auc,
        "prauc": prauc,
        "pr09": precision_at_recall(0.9),
        "pr08": precision_at_recall(0.8),
        "acc": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "minrp": float(np.minimum(curve_precision, curve_recall).max()),
    }


def make_weighted_sampler(
    stay_ids: Iterable[int], label_by_stay: Mapping[int, int]
) -> WeightedRandomSampler:
    labels = np.asarray(
        [int(label_by_stay[int(stay_id)]) for stay_id in stay_ids],
        dtype=np.int64,
    )
    if labels.size == 0 or np.any((labels < 0) | (labels > 1)):
        raise ValueError("Binary fold labels must contain only 0 and 1")
    counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def cpu_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def summarize_folds(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not results:
        return {}
    return {
        key: {
            "mean": float(np.mean([result[key] for result in results])),
            "std": float(np.std([result[key] for result in results])),
        }
        for key in results[0]
        if isinstance(results[0][key], (int, float))
    }


def selected_fold_indices(total: int, requested: int | None) -> list[int]:
    if requested is None:
        return list(range(total))
    if not 1 <= requested <= total:
        raise ValueError(f"fold must be in [1, {total}]")
    return [requested - 1]


RESULT_METRIC_COLUMNS = (
    "auc",
    "prauc",
    "pr09",
    "pr08",
    "f1",
    "acc",
    "minrp",
    "recall",
    "precision",
)


def write_fold_results_csv(
    fold_results: list[dict[str, float]],
    output_path: str | Path,
    *,
    dataset: str,
    model: str,
    seed: int,
    num_folds: int,
    window_size: int,
    regularization_c: float,
    train_sizes: Mapping[int, int],
    test_size: int,
    feature_csv_path: str | Path,
    processed_pkl_path: str | Path,
) -> pd.DataFrame:
    """Write the project-standard fold/mean/std result table.

    The column contract intentionally matches
    ``results/*/sapsii_logistic_regression.csv`` so model results can be
    concatenated without schema-specific handling.
    """

    if not fold_results:
        raise ValueError("Cannot write an empty result set")
    missing_metrics = {
        name
        for result in fold_results
        for name in RESULT_METRIC_COLUMNS
        if name not in result
    }
    if missing_metrics:
        raise KeyError(f"Result rows are missing metrics: {sorted(missing_metrics)}")

    common = {
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "num_folds": int(num_folds),
        "window_size": int(window_size),
        "regularization_c": float(regularization_c),
        "test_samples": int(test_size),
        "feature_csv": str(Path(feature_csv_path)),
        "processed_pkl": str(Path(processed_pkl_path)),
    }
    rows: list[dict[str, object]] = []
    for result in sorted(fold_results, key=lambda item: int(item["fold"])):
        fold = int(result["fold"])
        if fold not in train_sizes:
            raise KeyError(f"No training-set size is available for fold {fold}")
        rows.append(
            {
                **common,
                "result_type": "fold",
                "fold": fold,
                "train_samples": int(train_sizes[fold]),
                **{name: float(result[name]) for name in RESULT_METRIC_COLUMNS},
            }
        )
    for result_type, reducer in (("mean", np.mean), ("std", np.std)):
        rows.append(
            {
                **common,
                "result_type": result_type,
                "fold": pd.NA,
                "train_samples": pd.NA,
                **{
                    name: float(reducer([row[name] for row in fold_results]))
                    for name in RESULT_METRIC_COLUMNS
                },
            }
        )

    columns = [
        "dataset",
        "model",
        "result_type",
        "fold",
        "seed",
        "num_folds",
        "window_size",
        "regularization_c",
        "train_samples",
        "test_samples",
        "feature_csv",
        "processed_pkl",
        *RESULT_METRIC_COLUMNS,
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["fold"] = frame["fold"].astype("Int64")
    frame["train_samples"] = frame["train_samples"].astype("Int64")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved fold and summary metrics to {output_path.resolve()}")
    return frame
