"""GRAM dataset adapter for MIMIC-III, MIMIC-IV and eICU graph caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from dataset.ehr_cached_graph_dataset import CachedICUStayDataset, read_json


@dataclass(frozen=True)
class GRAMHierarchy:
    ancestor_index: Tensor
    ancestor_mask: Tensor
    node_names: tuple[str, ...]
    leaf_names: tuple[str, ...]

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_names)

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)


def build_gram_hierarchy(
    var2ind_path: str | Path, metadata_path: str | Path
) -> GRAMHierarchy:
    """Build leaf -> subclass -> class -> root paths from project metadata."""

    var2ind = read_json(var2ind_path)
    metadata = read_json(metadata_path)
    if sorted(map(int, var2ind.values())) != list(range(len(var2ind))):
        raise ValueError("var2ind values must be contiguous from zero")
    leaf_names = [""] * len(var2ind)
    for variable, index in var2ind.items():
        leaf_names[int(index)] = str(variable)

    node_names = list(leaf_names)
    node_index = {f"leaf:{name}": index for index, name in enumerate(leaf_names)}

    def get_node(namespace: str, name: str) -> int:
        key = f"{namespace}:{name}"
        if key not in node_index:
            node_index[key] = len(node_names)
            node_names.append(key)
        return node_index[key]

    root_index = get_node("root", "EHR")
    paths: list[list[int]] = []
    for leaf_index, variable in enumerate(leaf_names):
        if variable not in metadata:
            raise KeyError(f"{variable!r} is missing from {metadata_path}")
        item = metadata[variable]
        path = [leaf_index]
        subclass = item.get("sub_class")
        if subclass not in (None, "", "None"):
            path.append(get_node("subclass", str(subclass)))
        class_name = item.get("class")
        if class_name in (None, ""):
            raise KeyError(f"{variable!r} has no class metadata")
        path.append(get_node("class", str(class_name)))
        path.append(root_index)
        # Preserve order but remove duplicate hierarchy nodes.
        paths.append(list(dict.fromkeys(path)))

    max_ancestors = max(map(len, paths))
    ancestor_index = torch.zeros(len(paths), max_ancestors, dtype=torch.long)
    ancestor_mask = torch.zeros(len(paths), max_ancestors, dtype=torch.bool)
    for row, path in enumerate(paths):
        ancestor_index[row, : len(path)] = torch.tensor(path, dtype=torch.long)
        ancestor_mask[row, : len(path)] = True
    return GRAMHierarchy(
        ancestor_index=ancestor_index,
        ancestor_mask=ancestor_mask,
        node_names=tuple(node_names),
        leaf_names=tuple(leaf_names),
    )


class GRAMICUDataset(CachedICUStayDataset):
    """Convert cached events to dense time-bin visits expected by GRAM."""

    def __init__(
        self,
        processed_dir: str | Path,
        stay_ids: Sequence[int],
        *,
        num_variables: int,
        time_limit_hours: float = 24.0,
        bin_minutes: int = 60,
        index_path: str | Path | None = None,
    ) -> None:
        super().__init__(processed_dir, stay_ids, index_path=index_path)
        if num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if time_limit_hours <= 0 or bin_minutes <= 0:
            raise ValueError("time_limit_hours and bin_minutes must be positive")
        self.num_variables = int(num_variables)
        self.time_limit_minutes = float(time_limit_hours) * 60.0
        self.bin_minutes = int(bin_minutes)
        self.num_time_bins = int(
            (self.time_limit_minutes + self.bin_minutes - 1) // self.bin_minutes
        )

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

        bins = torch.div(times, self.bin_minutes, rounding_mode="floor").long()
        flat_index = bins * self.num_variables + variables
        flat_size = self.num_time_bins * self.num_variables
        sums = torch.zeros(flat_size, dtype=torch.float32)
        counts = torch.zeros(flat_size, dtype=torch.float32)
        sums.index_add_(0, flat_index, values)
        counts.index_add_(0, flat_index, torch.ones_like(values))
        observed = counts > 0
        dense_values = torch.zeros_like(sums)
        dense_values[observed] = sums[observed] / counts[observed]
        presence = observed.float().view(self.num_time_bins, self.num_variables)
        dense_values = dense_values.view(self.num_time_bins, self.num_variables)
        length = int(bins.max().item()) + 1
        return {
            "presence": presence,
            "values": dense_values,
            "lengths": torch.tensor(length, dtype=torch.long),
            "demo": stay["demo"],
            "labels": stay["label"].float(),
            "stay_id": torch.tensor(stay_id, dtype=torch.long),
        }

