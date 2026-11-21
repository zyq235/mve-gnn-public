import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, average_precision_score, \
    precision_recall_curve

def get_precision_at_target_recall(labels, probs, target_recall=0.9):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    idx = np.where(recalls >= target_recall)[0][-1]

    p_at_r = precisions[idx]
    corresponding_threshold = thresholds[idx] if idx < len(thresholds) else 1.0

    return p_at_r, corresponding_threshold