"""Utilities for safe semantic-type grouping in batched PyG graphs.

``torch_geometric.data.Data`` treats every attribute containing ``"batch"``
as an index.  Its default increment is the largest value observed in the
preceding graph plus one.  That is not a valid stride for ``batch_type`` when a
patient does not contain the highest semantic type: the type-view model always
allocates the same number of type slots to every patient.

The dataset now supplies the correct fixed increment.  The normalizer below is
also kept at the model boundary so legacy cached ``Data`` objects and external
callers cannot silently mix semantic groups between patients.
"""

from __future__ import annotations

from typing import Any

import torch


def _batch_type_increments(batch: Any, device: torch.device) -> torch.Tensor | None:
    increments_by_key = getattr(batch, "_inc_dict", None)
    if not isinstance(increments_by_key, dict):
        return None
    increments = increments_by_key.get("batch_type")
    if increments is None:
        return None
    return torch.as_tensor(increments, device=device).view(-1).long()


def normalize_batched_type_indices(
    batch: Any,
    num_type_groups: int,
    variable_to_class: Any | None = None,
) -> Any:
    """Return ``batch`` with fixed-stride global semantic-type indices.

    The canonical index for a node is

    ``graph_index * num_type_groups + patient_local_type_index``.

    When ``variable_to_class`` is supplied, the patient-local type is rebuilt
    deterministically from each node's variable ID.  This is the authoritative
    path for multi-dataset fine-tuning because cached ``batch_type`` tensors may
    have been created with an older dataset taxonomy.  Otherwise, correctly
    batched inputs are returned unchanged and legacy PyG batches are recovered
    from their collation metadata.
    """

    if not isinstance(num_type_groups, int) or num_type_groups < 1:
        raise ValueError("num_type_groups must be a positive integer")
    if not hasattr(batch, "batch_type") or not hasattr(batch, "batch"):
        raise AttributeError("type pooling requires both batch_type and batch")

    stored = batch.batch_type.view(-1).long()
    graph_index = batch.batch.view(-1).long()
    if stored.numel() != graph_index.numel():
        raise ValueError(
            "batch_type and batch must contain one entry per node: "
            f"got {stored.numel()} and {graph_index.numel()}"
        )

    num_graphs = int(batch.num_graphs)
    if num_graphs < 1:
        raise ValueError("type pooling requires at least one graph")
    if graph_index.numel() == 0:
        raise ValueError("type pooling does not support empty patient graphs")
    if int(graph_index.min()) < 0 or int(graph_index.max()) >= num_graphs:
        raise ValueError(
            f"batch contains graph IDs outside [0, {num_graphs}): "
            f"range=({int(graph_index.min())}, {int(graph_index.max())})"
        )

    node_counts = torch.bincount(graph_index, minlength=num_graphs)
    if bool((node_counts == 0).any().item()):
        empty_graphs = torch.nonzero(node_counts == 0, as_tuple=False).view(-1)
        raise ValueError(
            "type pooling does not support empty patient graphs; graph IDs: "
            f"{empty_graphs.detach().cpu().tolist()}"
        )

    if variable_to_class is not None:
        if not hasattr(batch, "x_var"):
            raise AttributeError(
                "variable_to_class reconstruction requires batch.x_var"
            )
        variable_ids = batch.x_var.view(-1).long()
        if variable_ids.numel() != graph_index.numel():
            raise ValueError(
                "x_var and batch must contain one entry per node: "
                f"got {variable_ids.numel()} and {graph_index.numel()}"
            )
        lookup = torch.as_tensor(
            variable_to_class, device=stored.device, dtype=torch.long
        ).view(-1)
        invalid_variables = (variable_ids < 0) | (variable_ids >= lookup.numel())
        if bool(invalid_variables.any().item()):
            invalid = torch.unique(
                variable_ids[invalid_variables]
            ).detach().cpu().tolist()
            raise ValueError(
                "Found variable IDs outside the current variable-to-class "
                f"lookup [0, {lookup.numel()}): {invalid}"
            )
        local_type = lookup[variable_ids]
        valid_local = (local_type >= 0) & (local_type < num_type_groups)
        if not bool(valid_local.all().item()):
            invalid = torch.unique(
                local_type[~valid_local]
            ).detach().cpu().tolist()
            raise ValueError(
                "The current variable-to-class lookup produced type IDs "
                f"outside [0, {num_type_groups}): {invalid}"
            )
        batch.batch_type = graph_index * num_type_groups + local_type
        batch._fixed_stride_type_indices = True
        batch._type_indices_rebuilt_from_variables = True
        return batch

    canonical_owner = torch.div(stored, num_type_groups, rounding_mode="floor")
    already_canonical = bool(
        (
            (stored >= 0)
            & (stored < num_graphs * num_type_groups)
            & (canonical_owner == graph_index)
        )
        .all()
        .item()
    )
    if already_canonical:
        batch._fixed_stride_type_indices = True
        return batch

    increments = _batch_type_increments(batch, stored.device)
    if increments is None:
        raise ValueError(
            "batch_type is not fixed-stride and PyG collation increments are "
            "unavailable, so patient-local type IDs cannot be recovered safely"
        )
    if increments.numel() != num_graphs:
        raise ValueError(
            "Unexpected batch_type increment metadata: "
            f"got {increments.numel()} values for {num_graphs} graphs"
        )

    local_type = stored - increments[graph_index]
    valid_local = (local_type >= 0) & (local_type < num_type_groups)
    if not bool(valid_local.all().item()):
        invalid = torch.unique(local_type[~valid_local]).detach().cpu().tolist()
        raise ValueError(
            f"Found patient-local type IDs outside [0, {num_type_groups}): "
            f"{invalid}"
        )

    fixed = graph_index * num_type_groups + local_type
    if not bool(
        (
            torch.div(fixed, num_type_groups, rounding_mode="floor")
            == graph_index
        )
        .all()
        .item()
    ):
        raise RuntimeError("Failed to build patient-aligned semantic-type indices")

    batch.batch_type = fixed
    batch._fixed_stride_type_indices = True
    return batch
