"""Small, class-stratified subsets for isolated LOS smoke runs only."""

from __future__ import annotations

import numpy as np

from ML.los_metrics import NUM_LOS_CLASSES


def subset_per_class(stay_ids, targets, per_class: int, *, require_all: bool = False):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    counts = np.zeros(NUM_LOS_CLASSES, dtype=np.int64)
    selected = []
    for stay_id in stay_ids:
        label = int(targets.loc[int(stay_id), "los_class"])
        if counts[label] < per_class:
            selected.append(int(stay_id))
            counts[label] += 1
    if not selected or (require_all and (counts == 0).any()):
        raise ValueError("Smoke subset is empty or lacks a required LOS class")
    return np.asarray(selected, dtype=np.int64)
