"""Temporal convolutional baseline adapted from the STraTS repository."""

import torch
import torch.nn as nn

from .timeseries_model import TimeSeriesModel


class StableWeightNormConv1d(nn.Conv1d):
    """Legacy weight-norm checkpoint layout with a safe per-filter norm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight_v = self.weight.detach().clone()
        weight_g = torch.linalg.vector_norm(weight_v, dim=(1, 2), keepdim=True)
        del self.weight
        self.weight_v = nn.Parameter(weight_v)
        self.weight_g = nn.Parameter(weight_g)

    def forward(self, x):
        # torch.nn.utils.weight_norm divides by ||weight_v|| without a lower
        # bound. A near-zero filter can therefore corrupt weight_v gradients.
        norm = torch.linalg.vector_norm(
            self.weight_v, dim=(1, 2), keepdim=True
        ).clamp_min(1e-6)
        weight = self.weight_g * (self.weight_v / norm)
        return self._conv_forward(x, weight, self.bias)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super().__init__()
        self.conv1 = StableWeightNormConv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = StableWeightNormConv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # The legacy implementation wrote to conv.weight after applying
        # weight_norm; its forward hook replaced that tensor from weight_v/g.
        # Keep the same effective Conv1d initialization for compatibility.
        with torch.no_grad():
            if self.downsample is not None:
                self.downsample.weight.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for index, out_channels in enumerate(num_channels):
            dilation = 2**index
            in_channels = num_inputs if index == 0 else num_channels[index - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)
        self.tcn = TemporalConvNet(
            args.V * 3,
            [args.hid_dim] * args.num_layers,
            args.kernel_size,
            args.dropout,
        )

    def forward(self, ts, demo, labels=None):
        ts = torch.permute(ts, (0, 2, 1))
        ts_emb = self.tcn(ts)[:, :, -1]
        demo_emb = self.demo_emb(demo)
        logits = self.binary_head(torch.cat((ts_emb, demo_emb), dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)
