"""GRU baseline adapted from the reference STraTS implementation."""

from argparse import Namespace

import torch
import torch.nn as nn

from .timeseries_model import TimeSeriesModel


class GRU_TS(TimeSeriesModel):
    """Encode a fixed-grid ``[value, mask, delta]`` time series with a GRU."""

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.gru = nn.GRU(
            args.V * 3,
            args.hid_dim,
            batch_first=True,
            # The reference model is a single-layer GRU, where PyTorch's
            # internal dropout is inactive.  The explicit layer below applies
            # the configured dropout to the final embedding.
            dropout=0.0,
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, ts, demo, labels=None):
        ts_emb = self.gru(ts)[1].reshape((ts.size(0), -1))
        ts_emb = self.dropout(ts_emb)
        demo_emb = self.demo_emb(demo)
        logits = self.binary_head(torch.cat((ts_emb, demo_emb), dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)
