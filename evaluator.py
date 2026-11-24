import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    auc,
)


PRAUC_DEFINITION = "trapezoidal_auc_recall_precision"


def trapezoidal_pr_auc(labels, probabilities):
    """Integrate precision against recall, averaging columns for one-hot targets."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim == 2 and labels.shape[1] > 1:
        if probabilities.shape != labels.shape:
            raise ValueError("One-hot labels and prediction scores must have matching shapes")
        scores_by_class = [
            trapezoidal_pr_auc(labels[:, column], probabilities[:, column])
            for column in range(labels.shape[1])
        ]
        return float(np.mean(scores_by_class))
    precision, recall, _ = precision_recall_curve(
        labels.reshape(-1), probabilities.reshape(-1)
    )
    return float(auc(recall, precision))


def get_precision_at_target_recall(labels, probs, target_recall=0.9):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    idx = np.where(recalls >= target_recall)[0][-1]

    p_at_r = precisions[idx]
    corresponding_threshold = thresholds[idx] if idx < len(thresholds) else 1.0

    return p_at_r, corresponding_threshold