"""Shared model, metric, and checkpoint helpers for baseline scripts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from evaluator import get_precision_at_target_recall
from models.modeling_gru import GRU_TS
from models.modeling_grud import GRUD_TS
from models.modeling_sand import SAND
from models.modeling_tcn import TCN_TS


MODEL_CLASSES = {
    "gru": GRU_TS,
    "grud": GRUD_TS,
    "sand": SAND,
    "tcn": TCN_TS,
}


def build_model(model_type, data_store, config):
    """Build one baseline with dimensions inferred from the loaded dataset."""

    args = SimpleNamespace(
        model_type=model_type,
        V=data_store.num_features,
        D=data_store.num_demo,
        T=int(np.ceil(data_store.max_minute / config.time_step_minutes)),
        hid_dim=config.hid_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        kernel_size=config.kernel_size,
        r=config.r,
        M=config.M,
        pretrain=0,
        # These baselines are trained directly for classification.  Keeping
        # this None makes the reference base class construct a one-stage head.
        load_ckpt_path=None,
        pos_class_weight=config.pos_class_weight,
    )
    return MODEL_CLASSES[model_type](args)


def calculate_metrics(y_true, y_probs, threshold=0.5):
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_preds = (y_probs > threshold).astype(np.int64)
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc = 0.0
    accuracy = float(accuracy_score(y_true, y_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_preds, average="binary", zero_division=0
    )
    try:
        prauc = float(average_precision_score(y_true, y_probs))
        pr_values, recall_values, _ = precision_recall_curve(y_true, y_probs)
        minrp = float(np.minimum(pr_values, recall_values).max())
    except ValueError:
        prauc, minrp = 0.0, 0.0
    try:
        pr08, _ = get_precision_at_target_recall(
            y_true, y_probs, target_recall=0.8
        )
        pr09, _ = get_precision_at_target_recall(
            y_true, y_probs, target_recall=0.9
        )
    except (IndexError, ValueError, ZeroDivisionError):
        pr08, pr09 = 0.0, 0.0
    return {
        "auc": auc,
        "acc": accuracy,
        "f1": float(f1),
        "prauc": prauc,
        "minrp": minrp,
        "recall": float(recall),
        "precision": float(precision),
        "pr08": float(pr08),
        "pr09": float(pr09),
    }


def move_batch(batch, device, include_labels=True):
    moved = {name: tensor.to(device) for name, tensor in batch.items()}
    if not include_labels:
        moved.pop("labels")
    return moved


def load_state_dict_file(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {path} does not contain a state dict")
    return checkpoint


def load_matching_weights(model, path, device):
    """Load shape-compatible tensors for optional initialization."""

    incoming = load_state_dict_file(path, device)
    current = model.state_dict()
    matched = {
        name: value
        for name, value in incoming.items()
        if name in current and current[name].shape == value.shape
    }
    current.update(matched)
    model.load_state_dict(current)
    return len(matched), len(incoming)

