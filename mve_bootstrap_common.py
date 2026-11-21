"""Shared contracts for full-train repeat-5 MVE-GNN distillation.

The training cohort is always the original pickle ``train`` split.  The
original validation split selects checkpoints, and the fixed test split is
reserved for final evaluation and patient-level bootstrap analysis.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from patient_bootstrap import metric_values


TEACHER_REPEATS = 5


@dataclass(frozen=True)
class SplitLogits:
    stay_ids: np.ndarray
    labels: np.ndarray
    member_logits: np.ndarray

    @property
    def mean_logits(self) -> np.ndarray:
        return self.member_logits.mean(axis=0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def repeat_seed(base_seed: int, repeat_number: int) -> int:
    if repeat_number < 1:
        raise ValueError("repeat_number is 1-based and must be positive")
    return int(base_seed + (repeat_number - 1) * 10_000)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def logit_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    metrics = metric_values(
        np.asarray(labels, dtype=np.int64), sigmoid_numpy(logits)
    )
    metrics["selection_score"] = float(metrics["auc"] + metrics["prauc"])
    return metrics


def binary_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: float = 0.75,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, hard-label, and soft-teacher binary losses."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
    soft_targets = torch.sigmoid(teacher_logits / temperature)
    soft_loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature, soft_targets
    )
    total_loss = (
        (1.0 - weight) * hard_loss
        + weight * (temperature**2) * soft_loss
    )
    return total_loss, hard_loss, soft_loss


def batch_stay_ids(batch: Any) -> list[int]:
    value = getattr(batch, "ts_id", None)
    if value is None:
        raise AttributeError("MVE distillation requires batch.ts_id")
    if isinstance(value, torch.Tensor):
        items = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    return [int(item) for item in items]


def make_mve_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int | None = None,
):
    from torch_geometric.loader import DataLoader

    kwargs: dict[str, Any] = {}
    if shuffle:
        if seed is None:
            raise ValueError("A deterministic seed is required for a shuffled loader")
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        **kwargs,
    )


@torch.no_grad()
def collect_mve_logits(
    adapter,
    model: nn.Module,
    loader,
    device: torch.device,
    expected_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    stay_ids: list[int] = []
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    for batch in loader:
        batch = adapter.ft.normalize_collated_group_ids(
            batch.to(device), adapter.model_args
        )
        output = model(batch, mode=adapter.model_args.pool_mode)
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError(
                f"Non-finite MVE logits for stays {batch_stay_ids(batch)}"
            )
        stay_ids.extend(batch_stay_ids(batch))
        labels.append(batch.y.detach().view(-1).cpu())
        logits.append(output.detach().float().view(-1).cpu())

    id_array = np.asarray(stay_ids, dtype=np.int64)
    if id_array.size != len(expected_ids):
        raise RuntimeError("MVE output count does not match the requested split")
    if np.unique(id_array).size != id_array.size:
        raise RuntimeError("MVE produced duplicate stay IDs")
    if set(id_array.tolist()) != set(map(int, expected_ids)):
        raise RuntimeError("MVE output stay IDs do not match the requested split")
    label_array = torch.cat(labels).numpy().astype(np.int64)
    logit_array = torch.cat(logits).numpy().astype(np.float32)
    order = np.argsort(id_array, kind="stable")
    return id_array[order], label_array[order], logit_array[order]


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
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    return dict(payload)


def checkpoint_signature(paths: Sequence[Path]) -> str:
    entries = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        stat = resolved.stat()
        entries.append(
            {
                "path": str(resolved),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode("utf-8")
    ).hexdigest()


def save_logit_cache(
    path: Path,
    signature: str,
    splits: Mapping[str, SplitLogits],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"checkpoint_signature": np.asarray(signature)}
    for name, split in splits.items():
        arrays[f"{name}_ids"] = split.stay_ids
        arrays[f"{name}_labels"] = split.labels
        arrays[f"{name}_member_logits"] = split.member_logits
    np.savez_compressed(path, **arrays)


def load_logit_cache(
    path: Path,
    signature: str,
    expected: Mapping[str, Mapping[int, int]],
    expected_members: int = TEACHER_REPEATS,
) -> dict[str, SplitLogits] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["checkpoint_signature"].item()) != signature:
                return None
            result: dict[str, SplitLogits] = {}
            for name, expected_labels in expected.items():
                split = SplitLogits(
                    stay_ids=payload[f"{name}_ids"].copy(),
                    labels=payload[f"{name}_labels"].copy(),
                    member_logits=payload[f"{name}_member_logits"].copy(),
                )
                if split.member_logits.shape != (
                    expected_members,
                    split.stay_ids.size,
                ):
                    return None
                actual = {
                    int(stay_id): int(label)
                    for stay_id, label in zip(split.stay_ids, split.labels)
                }
                normalized = {
                    int(stay_id): int(label)
                    for stay_id, label in expected_labels.items()
                }
                if actual != normalized:
                    return None
                if not bool(np.isfinite(split.member_logits).all()):
                    return None
                result[name] = split
            return result
    except (KeyError, OSError, ValueError):
        return None


def teacher_lookup(split: SplitLogits) -> dict[int, float]:
    return {
        int(stay_id): float(logit)
        for stay_id, logit in zip(split.stay_ids, split.mean_logits)
    }


def prediction_frame(
    split: SplitLogits | tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    use_mean: bool = False,
) -> pd.DataFrame:
    if isinstance(split, SplitLogits):
        if not use_mean:
            raise ValueError("SplitLogits requires use_mean=True")
        stay_ids, labels, logits = split.stay_ids, split.labels, split.mean_logits
    else:
        stay_ids, labels, logits = split
    return pd.DataFrame(
        {
            "stay_id": np.asarray(stay_ids, dtype=np.int64),
            "label": np.asarray(labels, dtype=np.int64),
            "probability": sigmoid_numpy(np.asarray(logits)),
        }
    ).sort_values("stay_id").reset_index(drop=True)
