"""SAnD baseline adapted from the reference STraTS implementation."""

from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .timeseries_model import TimeSeriesModel


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.hid_dim % args.num_heads != 0:
            raise ValueError("hid_dim must be divisible by num_heads")
        self.dk = args.hid_dim // args.num_heads
        self.Wq = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)))
        self.Wk = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)))
        self.Wv = nn.Parameter(torch.empty((args.hid_dim, args.hid_dim)))
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        self.Wo = nn.Linear(args.hid_dim, args.hid_dim, bias=False)
        self.num_heads = args.num_heads
        self.dropout = args.dropout

    def forward(self, x, mask):
        batch_size, timesteps, hidden_dim = x.size()
        queries = torch.matmul(x, self.Wq).view(
            batch_size, timesteps, self.num_heads, self.dk
        ) / np.sqrt(self.dk)
        keys = torch.matmul(x, self.Wk).view(
            batch_size, timesteps, self.num_heads, self.dk
        )
        values = torch.matmul(x, self.Wv).view(
            batch_size, timesteps, self.num_heads, self.dk
        )
        attention = torch.einsum("bthd,blhd->bhtl", queries, keys) + mask
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.einsum("bhtl,blhd->bthd", attention, values)
        return self.Wo(output.reshape((batch_size, timesteps, hidden_dim)))


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(args.hid_dim, args.hid_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(args.hid_dim * 2, args.hid_dim, 1),
        )

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(args)
        self.norm_mha = nn.LayerNorm(args.hid_dim)
        self.norm_ffn = nn.LayerNorm(args.hid_dim)
        self.dropout = args.dropout

    def forward(self, x, mask):
        residual = F.dropout(
            self.mha(x, mask), self.dropout, training=self.training
        )
        x = self.norm_mha(x + residual)
        residual = F.dropout(self.ffn(x), self.dropout, training=self.training)
        return self.norm_ffn(x + residual)


class DenseInterpolation(nn.Module):
    def __init__(self, args):
        super().__init__()
        columns = torch.arange(args.M).reshape((1, args.M)) / args.M
        rows = torch.arange(args.T).reshape((args.T, 1)) / args.T
        self.register_buffer("W", (1 - torch.abs(rows - columns)) ** 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(x.transpose(1, 2), self.W)
        return x.reshape((batch_size, -1))


class SAND(TimeSeriesModel):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.input_embedding = nn.Conv1d(args.V * 3, args.hid_dim, 1)
        self.positional_encoding = nn.Parameter(
            torch.empty((1, args.T, args.hid_dim))
        )
        nn.init.normal_(self.positional_encoding)
        indices = torch.arange(args.T)
        mask = torch.logical_and(
            indices[None, :] <= indices[:, None],
            indices[None, :] >= indices[:, None] - args.r,
        ).float()
        self.register_buffer("mask", (1 - mask) * torch.finfo(mask.dtype).min)
        self.dropout = args.dropout
        self.transformer = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.num_layers)]
        )
        self.dense_interpolation = DenseInterpolation(args)

    def forward(self, ts, demo, labels=None):
        ts_emb = self.input_embedding(ts.permute((0, 2, 1))).permute((0, 2, 1))
        ts_emb = ts_emb + self.positional_encoding
        ts_emb = F.dropout(ts_emb, self.dropout, training=self.training)
        for layer in self.transformer:
            ts_emb = layer(ts_emb, self.mask)
        ts_emb = self.dense_interpolation(ts_emb)
        demo_emb = self.demo_emb(demo)
        logits = self.binary_head(torch.cat((ts_emb, demo_emb), dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)
