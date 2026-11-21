"""Ten-class remaining-LOS head for the unchanged StraTS encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_strats import Strats


class StratsLOS(Strats):
    """StraTS/iStraTS encoder with an independent categorical LOS head.

    With ``load_ckpt_path`` set, the original architecture creates the
    forecasting projection used by the available pretrained checkpoints.  We
    retain that projection, discard only the binary mortality head, and add a
    ten-logit LOS head.
    """

    def __init__(self, args, num_los_classes: int = 10):
        if num_los_classes < 2:
            raise ValueError("num_los_classes must be at least two")
        super().__init__(args)
        if hasattr(self, "binary_head"):
            del self.binary_head

        if self.finetune:
            head_input_dim = args.V
        elif args.model_type == "istrats":
            head_input_dim = args.hid_dim + args.D
        else:
            head_input_dim = args.hid_dim * 2
        self.num_los_classes = int(num_los_classes)
        self.los_head = nn.Linear(head_input_dim, self.num_los_classes)

    def encode_patient(self, values, times, varis, obs_mask, demo):
        """Run the original triplet encoder and return one patient vector."""

        batch_size, _ = values.size()
        device = values.device
        if self.training:
            with torch.no_grad():
                variable_mask = (
                    torch.rand((batch_size, self.V), device=device) <= self.dropout
                ).int()
                for variable_index in range(self.V):
                    masked_positions = (
                        (varis == variable_index).int()
                        * variable_mask[:, variable_index : variable_index + 1]
                    )
                    obs_mask = obs_mask * (1 - masked_positions)

        demo_embedding = (
            self.demo_emb(demo) if self.args.model_type == "strats" else demo
        )
        triplet_embedding = (
            self.cve_time(times)
            + self.cve_value(values)
            + self.variable_emb(varis)
        )
        triplet_embedding = F.dropout(
            triplet_embedding, self.dropout, self.training
        )
        contextual_embedding = self.transformer(triplet_embedding, obs_mask)
        attention = self.fusion_att(contextual_embedding, obs_mask).unsqueeze(-1)
        if self.args.model_type == "istrats":
            time_series_embedding = (triplet_embedding * attention).sum(dim=1)
        else:
            time_series_embedding = (contextual_embedding * attention).sum(dim=1)
        return torch.cat((time_series_embedding, demo_embedding), dim=-1)

    def forward(self, values, times, varis, obs_mask, demo):
        patient_embedding = self.encode_patient(
            values, times, varis, obs_mask, demo
        )
        if self.finetune:
            patient_embedding = self.forecast_head(patient_embedding)
        return self.los_head(patient_embedding)
