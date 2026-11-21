"""Ten-class LOS heads for the existing time-series baseline encoders.

The mortality model classes are left untouched.  These subclasses use the
same feature extractors and replace only their one-logit classification head.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.modeling_gru import GRU_TS
from models.modeling_grud import GRUD_TS
from models.modeling_sand import SAND
from models.modeling_tcn import TCN_TS


class _LOSHead:
    def _make_los_head(self):
        self.binary_head = nn.Linear(self.binary_head.in_features, 10)


class GRULOS(_LOSHead, GRU_TS):
    def __init__(self, args):
        super().__init__(args)
        self._make_los_head()

    def forward(self, ts, demo):
        encoded = self.dropout(self.gru(ts)[1].reshape(ts.size(0), -1))
        return self.binary_head(torch.cat((encoded, self.demo_emb(demo)), dim=-1))


class GRUDLOS(_LOSHead, GRUD_TS):
    def __init__(self, args):
        super().__init__(args)
        self._make_los_head()

    def forward(self, x_t, m_t, delta_t, seq_len, demo):
        count = x_t.size(0)
        initial = (
            torch.zeros((count, self.hid_dim), device=x_t.device),
            torch.zeros((count, self.V), device=x_t.device),
        )
        encoded = self.grud((x_t, m_t, delta_t), initial)
        width = encoded.size(2)
        index = (seq_len - 1)[:, None, None].repeat(1, 1, width)
        encoded = self.dropout(torch.gather(encoded, 1, index)[:, 0, :])
        return self.binary_head(torch.cat((encoded, self.demo_emb(demo)), dim=-1))


class SANDLOS(_LOSHead, SAND):
    def __init__(self, args):
        super().__init__(args)
        self._make_los_head()

    def forward(self, ts, demo):
        encoded = self.input_embedding(ts.permute(0, 2, 1)).permute(0, 2, 1)
        encoded = F.dropout(
            encoded + self.positional_encoding, self.dropout, training=self.training
        )
        for layer in self.transformer:
            encoded = layer(encoded, self.mask)
        encoded = self.dense_interpolation(encoded)
        return self.binary_head(torch.cat((encoded, self.demo_emb(demo)), dim=-1))


class TCNLOS(_LOSHead, TCN_TS):
    def __init__(self, args):
        super().__init__(args)
        self._make_los_head()

    def forward(self, ts, demo):
        encoded = self.tcn(ts.permute(0, 2, 1))[:, :, -1]
        return self.binary_head(torch.cat((encoded, self.demo_emb(demo)), dim=-1))


LOS_TIMESERIES_MODELS = {
    "gru": GRULOS,
    "grud": GRUDLOS,
    "sand": SANDLOS,
    "tcn": TCNLOS,
}
