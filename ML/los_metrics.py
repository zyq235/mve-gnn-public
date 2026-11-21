"""Metrics for the ordinal ten-class remaining-ICU-LOS task."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


NUM_LOS_CLASSES = 10


def ranked_probability_score(y_true, probabilities) -> float:
    """Return the multiclass RPS (lower is better).

    Unlike AUROC, RPS respects the order of the LOS bins and penalises a
    prediction more strongly when its probability mass is farther away from
    the true bin.
    """

    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape != (len(labels), NUM_LOS_CLASSES):
        raise ValueError(
            f"probabilities must have shape (n, {NUM_LOS_CLASSES})"
        )
    if len(labels) == 0:
        raise ValueError("Cannot score an empty target array")
    if ((labels < 0) | (labels >= NUM_LOS_CLASSES)).any():
        raise ValueError("LOS class labels must be in [0, 9]")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("Every probability row must sum to one")

    observed = np.eye(NUM_LOS_CLASSES, dtype=np.float64)[labels]
    cumulative_error = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(
        observed, axis=1
    )[:, :-1]
    return float(np.mean(np.sum(cumulative_error**2, axis=1) / (NUM_LOS_CLASSES - 1)))


def class_representative_days(y_class, remaining_los_days) -> np.ndarray:
    """Estimate one day-valued representative per bin from training data only."""

    labels = np.asarray(y_class, dtype=np.int64).reshape(-1)
    days = np.asarray(remaining_los_days, dtype=np.float64).reshape(-1)
    if len(labels) != len(days) or len(labels) == 0:
        raise ValueError("Labels and LOS days must be non-empty and aligned")
    if not np.isfinite(days).all() or (days < 0).any():
        raise ValueError("remaining_los_days must be finite and non-negative")

    representatives = np.empty(NUM_LOS_CLASSES, dtype=np.float64)
    for class_index in range(NUM_LOS_CLASSES):
        class_days = days[labels == class_index]
        if len(class_days) == 0:
            raise ValueError(
                f"Training data contain no samples for LOS class {class_index}"
            )
        representatives[class_index] = float(class_days.mean())
    return representatives


def compute_los_metrics(
    y_true,
    probabilities,
    remaining_los_days,
    representative_days,
) -> dict[str, float]:
    """Compute ordinal classification and day-scale LOS metrics."""

    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    days = np.asarray(remaining_los_days, dtype=np.float64).reshape(-1)
    representatives = np.asarray(representative_days, dtype=np.float64).reshape(-1)
    if probs.shape != (len(labels), NUM_LOS_CLASSES):
        raise ValueError("Probability and label arrays are not aligned")
    if len(days) != len(labels):
        raise ValueError("Exact LOS and label arrays are not aligned")
    if representatives.shape != (NUM_LOS_CLASSES,):
        raise ValueError("representative_days must contain ten values")

    predictions = probs.argmax(axis=1)
    predicted_days = representatives[predictions]
    return {
        "linear_kappa": float(
            cohen_kappa_score(
                labels,
                predictions,
                labels=np.arange(NUM_LOS_CLASSES),
                weights="linear",
            )
        ),
        "rps": ranked_probability_score(labels, probs),
        "mae_days": float(np.mean(np.abs(predicted_days - days))),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(NUM_LOS_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "within_1_bin_accuracy": float(
            np.mean(np.abs(predictions - labels) <= 1)
        ),
    }
