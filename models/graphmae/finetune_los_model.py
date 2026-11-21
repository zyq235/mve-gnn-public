"""Length-of-stay prediction head for the existing MVE-GNN encoder.

The mortality model intentionally remains unchanged.  This task-specific
subclass reuses the complete graph encoder and multi-view pooling stack, and
only replaces the final scalar mortality logit with ten LOS logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.graphmae.finetune_graph_model import HierarchicalEHRModel


def compact_time_slot_ids(batch_time: torch.Tensor) -> torch.Tensor:
    """Remove empty global slots while preserving within-stay time order.

    Filtering pre-ICU events can leave gaps in ``batch_time``.  The base time
    pooling path scatters into every ID up to the maximum, so an empty slot
    receives a default patient ID of zero.  That makes the patient IDs
    nonmonotonic for ``to_dense_batch`` and can fully mask a patient's tokens.
    """

    slots = batch_time.view(-1)
    if not slots.numel():
        raise ValueError("LOS time view received a batch with no measured time slots")
    if bool((slots < 0).any()):
        raise ValueError("LOS time view received negative time-slot IDs")
    return torch.unique(slots, sorted=True, return_inverse=True)[1]


class HierarchicalEHRLOSModel(HierarchicalEHRModel):
    """MVE-GNN with a categorical remaining-LOS prediction head."""

    def __init__(self, *args, num_los_classes: int = 10, **kwargs):
        if num_los_classes < 2:
            raise ValueError("num_los_classes must be at least two")
        super().__init__(*args, **kwargs)

        if not isinstance(self.classifier, nn.Sequential):
            raise TypeError("The base model classifier must be nn.Sequential")
        final_layer = self.classifier[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("The base model classifier must end with nn.Linear")

        self.num_los_classes = int(num_los_classes)
        self.classifier[-1] = nn.Linear(
            final_layer.in_features,
            self.num_los_classes,
            bias=final_layer.bias is not None,
        )

    def forward(self, data, mode="type"):
        if mode != "time":
            return super().forward(data, mode=mode)
        original_batch_time = data.batch_time
        data.batch_time = compact_time_slot_ids(original_batch_time)
        try:
            return super().forward(data, mode=mode)
        finally:
            # Keep the loader batch unchanged for diagnostics and any reuse.
            data.batch_time = original_batch_time
