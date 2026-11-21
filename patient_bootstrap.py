"""Patient-cluster bootstrap for fixed, aligned ICU test predictions.

This module never fits a model or changes a train/validation/test split.  One
bootstrap draw samples *patients* with replacement at the original test-patient
count, and includes every ICU stay belonging to each selected patient.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


METRICS = ("auc", "prauc", "pr09")
PREDICTION_COLUMNS = ("dataset", "model", "split_digest", "stay_id", "patient_id", "label", "probability")


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Match the project's AUROC, average precision and precision@recall 0.9."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.size != probabilities.size or labels.size == 0:
        raise ValueError("Labels and probabilities must be nonempty and aligned")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUROC requires both outcome classes")
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    qualifying = np.flatnonzero(recall >= 0.9)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "prauc": float(average_precision_score(labels, probabilities)),
        "pr09": float(precision[qualifying[-1]]) if qualifying.size else 0.0,
    }


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    frame = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    if frame.empty or frame["stay_id"].isna().any() or frame["patient_id"].isna().any():
        raise ValueError("Prediction rows need nonempty ICU-stay and patient IDs")
    for column in ("dataset", "model", "split_digest"):
        if (frame[column].isna().any() or frame[column].astype(str).nunique() != 1
                or not str(frame[column].iloc[0]).strip()):
            raise ValueError(f"All rows must have one nonempty {column}")
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="raise").astype("int64")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int64")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise").astype("float64")
    if frame["stay_id"].duplicated().any():
        raise ValueError("A prediction file contains repeated ICU-stay IDs")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Mortality labels must be binary")
    if not np.isfinite(frame["probability"]).all() or not frame["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be finite and between zero and one")
    return frame.sort_values("stay_id").reset_index(drop=True)


def _aligned_predictions(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if not frames:
        raise ValueError("At least one model prediction file is required")
    normalized = {name: validate_predictions(frame) for name, frame in frames.items()}
    if any(name != str(frame["model"].iloc[0]) for name, frame in normalized.items()):
        raise ValueError("Prediction-file model names must match their mapping keys")
    reference = next(iter(normalized.values()))
    probabilities: dict[str, np.ndarray] = {}
    for name, frame in normalized.items():
        for column in ("stay_id", "patient_id", "label", "dataset", "split_digest"):
            if not np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"{name} does not match the reference {column}; paired bootstrap is invalid")
        probabilities[name] = frame["probability"].to_numpy(dtype=np.float64)
    return reference, probabilities


def bootstrap_predictions(
    frames: Mapping[str, pd.DataFrame],
    *,
    reference_model: str,
    repetitions: int = 100,
    seed: int = 2026,
) -> tuple[dict, pd.DataFrame]:
    """Return observed/percentile CIs and null-centered paired-bootstrap tests.

    All model differences use identical patient draws.  A draw with only one
    outcome class is discarded and replaced; the number of discards is reported.
    The Monte Carlo p-value is exploratory when repetitions is only 100.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    base, probabilities = _aligned_predictions(frames)
    if reference_model not in probabilities:
        raise ValueError(f"Reference model {reference_model!r} is missing")
    labels = base["label"].to_numpy(dtype=np.int64)
    observed = {name: metric_values(labels, scores) for name, scores in probabilities.items()}
    patient_rows: dict[str, list[int]] = defaultdict(list)
    for row, patient_id in enumerate(base["patient_id"].tolist()):
        patient_rows[patient_id].append(row)
    clusters = [np.asarray(rows, dtype=np.int64) for rows in patient_rows.values()]
    rng = np.random.default_rng(seed)
    replicate_rows = []
    completed = 0
    discarded = 0
    while completed < repetitions:
        if discarded > max(1000, 20 * repetitions):
            raise RuntimeError("Too many one-class bootstrap draws; inspect test-set class balance")
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[index] for index in chosen])
        sampled_labels = labels[rows]
        if np.unique(sampled_labels).size != 2:
            discarded += 1
            continue
        completed += 1
        repetition = completed
        for name, scores in probabilities.items():
            replicate_rows.append({"repetition": repetition, "model": name,
                                   **metric_values(sampled_labels, scores[rows])})
    replicate_frame = pd.DataFrame(replicate_rows)
    summary = {
        "dataset": str(base["dataset"].iloc[0]),
        "split_digest": str(base["split_digest"].iloc[0]),
        "test_stays": int(len(base)),
        "test_patients": int(len(clusters)),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "discarded_one_class_draws": int(discarded),
        "bootstrap_unit": "patient (all ICU stays retained)",
        "ci_method": "2.5/97.5 percentile of paired patient-cluster draws",
        "p_method": "two-sided null-centered paired bootstrap, Monte Carlo +1 correction",
        "reference_model": reference_model,
        "models": {},
        "paired_differences": {},
    }
    for name in probabilities:
        subset = replicate_frame.loc[replicate_frame["model"].eq(name)]
        summary["models"][name] = {
            metric: {
                "point": observed[name][metric],
                "ci95": [float(value) for value in np.quantile(subset[metric], [0.025, 0.975])],
            }
            for metric in METRICS
        }
        if name == reference_model:
            continue
        delta_summary = {}
        reference_subset = replicate_frame.loc[
            replicate_frame["model"].eq(reference_model)
        ].sort_values("repetition")
        subset = subset.sort_values("repetition")
        for metric in METRICS:
            point = observed[reference_model][metric] - observed[name][metric]
            draws = reference_subset[metric].to_numpy() - subset[metric].to_numpy()
            centered = draws - point
            p_value = (1 + int(np.count_nonzero(np.abs(centered) >= abs(point)))) / (repetitions + 1)
            delta_summary[metric] = {
                "point": float(point),
                "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
                "p_uncorrected": float(p_value),
            }
        summary["paired_differences"][f"{reference_model}_minus_{name}"] = delta_summary
    # One family per dataset/run: every reported reference-vs-comparator metric.
    tests = [value for comparison in summary["paired_differences"].values()
             for value in comparison.values()]
    for rank, test in enumerate(sorted(tests, key=lambda item: item["p_uncorrected"])):
        test["p_holm"] = min(1.0, (len(tests) - rank) * test["p_uncorrected"])
    # Holm-adjusted p-values must be monotone in the unadjusted ordering.
    ordered = sorted(tests, key=lambda item: item["p_uncorrected"])
    running = 0.0
    for test in ordered:
        running = max(running, test["p_holm"])
        test["p_holm"] = running
    return summary, replicate_frame
