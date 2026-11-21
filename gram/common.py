"""Shared GRAM training/evaluation operations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ehr_baseline_utils import binary_metrics, move_batch
from models.gram import GRAMBinaryClassifier


def build_model(model_kwargs: dict[str, Any], device: torch.device) -> GRAMBinaryClassifier:
    return GRAMBinaryClassifier(**model_kwargs).to(device)


def _forward(model: GRAMBinaryClassifier, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        batch["presence"],
        batch["values"],
        batch["lengths"],
        batch["demo"],
    )
    assert isinstance(logits, torch.Tensor)
    return logits


def train_one_epoch(
    model: GRAMBinaryClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: float,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch)
        loss = criterion(logits, batch["labels"].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach()) * batch_size
        examples += batch_size
    return total_loss / max(examples, 1)


def evaluate(
    model: GRAMBinaryClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    examples = 0
    probabilities: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = _forward(model, batch)
            loss = criterion(logits, batch["labels"].float())
            batch_size = int(batch["labels"].shape[0])
            total_loss += float(loss) * batch_size
            examples += batch_size
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / max(examples, 1)
    return metrics

