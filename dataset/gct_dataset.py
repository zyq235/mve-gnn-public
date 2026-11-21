"""GCT event-code adapter for MIMIC-III, MIMIC-IV and eICU caches."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from dataset.ehr_cached_graph_dataset import CachedICUStayDataset, read_json


def build_variable_type_index(
    var2ind_path: str | Path, metadata_path: str | Path
) -> tuple[Tensor, tuple[str, ...]]:
    """Map every variable ID to its sorted top-level clinical class."""

    var2ind = read_json(var2ind_path)
    metadata = read_json(metadata_path)
    if sorted(map(int, var2ind.values())) != list(range(len(var2ind))):
        raise ValueError("var2ind values must be contiguous from zero")
    class_names = sorted(
        {str(metadata[variable]["class"]) for variable in var2ind}
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    variable_types = torch.empty(len(var2ind), dtype=torch.long)
    for variable, variable_index in var2ind.items():
        if variable not in metadata or "class" not in metadata[variable]:
            raise KeyError(f"{variable!r} has no class metadata")
        variable_types[int(variable_index)] = class_to_index[
            str(metadata[variable]["class"])
        ]
    return variable_types, tuple(class_names)


class GCTICUDataset(CachedICUStayDataset):
    """Represent one ICU stay as a set of observed clinical variable codes.

    Repeated measurements of a variable are collapsed to their mean normalized
    value.  Its final observation time is retained as a normalized temporal
    feature.  This matches GCT's code-set formulation while making it usable on
    the continuous bedside variables shared by all three project cohorts.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        stay_ids: Sequence[int],
        *,
        variable_type_index: Tensor,
        time_limit_hours: float = 24.0,
        max_codes: int | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        super().__init__(processed_dir, stay_ids, index_path=index_path)
        if variable_type_index.ndim != 1 or variable_type_index.numel() == 0:
            raise ValueError("variable_type_index must be a non-empty vector")
        if time_limit_hours <= 0:
            raise ValueError("time_limit_hours must be positive")
        if max_codes is not None and max_codes <= 0:
            raise ValueError("max_codes must be positive when supplied")
        self.variable_type_index = variable_type_index.long().clone()
        self.num_variables = int(variable_type_index.numel())
        self.num_types = int(variable_type_index.max().item()) + 1
        self.time_limit_minutes = float(time_limit_hours) * 60.0
        self.max_codes = max_codes

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        stay_id = self.stay_ids[index]
        stay = self._load_stay(stay_id)
        variables = stay["variables"]
        values = stay["values"]
        times = stay["times"]
        valid = (
            torch.isfinite(values)
            & torch.isfinite(times)
            & (times >= 0)
            & (times < self.time_limit_minutes)
            & (variables >= 0)
            & (variables < self.num_variables)
        )
        variables, values, times = variables[valid], values[valid], times[valid]
        if variables.numel() == 0:
            raise ValueError(f"ICU stay {stay_id} has no event inside the time horizon")

        unique_variables, inverse = torch.unique(
            variables, sorted=True, return_inverse=True
        )
        sums = torch.zeros(unique_variables.numel(), dtype=torch.float32)
        counts = torch.zeros_like(sums)
        sums.index_add_(0, inverse, values)
        counts.index_add_(0, inverse, torch.ones_like(values))
        mean_values = sums / counts.clamp_min(1.0)
        last_times = torch.full_like(sums, -torch.inf)
        for event_index, group_index in enumerate(inverse.tolist()):
            last_times[group_index] = torch.maximum(
                last_times[group_index], times[event_index]
            )
        order = torch.argsort(last_times)
        if self.max_codes is not None and order.numel() > self.max_codes:
            order = order[-self.max_codes :]
        unique_variables = unique_variables[order]
        mean_values = mean_values[order]
        last_times = last_times[order] / self.time_limit_minutes
        return {
            # Zero is reserved for padding in the model embedding table.
            "code_ids": unique_variables + 1,
            "values": mean_values,
            "times": last_times.clamp(0.0, 1.0),
            "type_ids": self.variable_type_index[unique_variables],
            "mask": torch.ones(unique_variables.numel(), dtype=torch.bool),
            "demo": stay["demo"],
            "labels": stay["label"].float(),
            "stay_id": torch.tensor(stay_id, dtype=torch.long),
        }


def gct_collate_fn(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    max_codes = max(int(sample["code_ids"].numel()) for sample in samples)
    code_ids = torch.zeros(batch_size, max_codes, dtype=torch.long)
    values = torch.zeros(batch_size, max_codes, dtype=torch.float32)
    times = torch.zeros(batch_size, max_codes, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_codes), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_codes, dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = int(sample["code_ids"].numel())
        code_ids[row, :length] = sample["code_ids"]
        values[row, :length] = sample["values"]
        times[row, :length] = sample["times"]
        type_ids[row, :length] = sample["type_ids"]
        mask[row, :length] = True
    return {
        "code_ids": code_ids,
        "values": values,
        "times": times,
        "type_ids": type_ids,
        "mask": mask,
        "demo": torch.stack([sample["demo"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
        "stay_id": torch.stack([sample["stay_id"] for sample in samples]),
    }

