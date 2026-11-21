"""Dependency-light base class shared by classic time-series baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesModel(nn.Module):
    """Common demographic embedding and binary/forecast heads.

    This mirrors ``models/StraTS/models.py`` but deliberately avoids importing
    its logging utilities (and their unrelated ``transformers`` dependency).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type != "istrats":
            self.demo_emb = nn.Sequential(
                nn.Linear(args.D, args.hid_dim * 2),
                nn.Tanh(),
                nn.Linear(args.hid_dim * 2, args.hid_dim),
            )
        if args.model_type == "istrats":
            embedding_size = args.hid_dim + args.D
        elif args.model_type == "sand":
            embedding_size = args.hid_dim * args.M + args.hid_dim
        else:
            embedding_size = args.hid_dim * 2

        self.pretrain = args.pretrain == 1
        self.finetune = args.load_ckpt_path is not None
        if self.pretrain:
            self.forecast_head = nn.Linear(embedding_size, args.V)
        elif self.finetune:
            self.forecast_head = nn.Linear(embedding_size, args.V)
            self.binary_head = nn.Linear(args.V, 1)
            self.register_buffer(
                "pos_class_weight", torch.tensor(float(args.pos_class_weight))
            )
        else:
            self.binary_head = nn.Linear(embedding_size, 1)
            self.register_buffer(
                "pos_class_weight", torch.tensor(float(args.pos_class_weight))
            )

    def binary_cls_final(self, logits, labels):
        if labels is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_class_weight
            )
        return torch.sigmoid(logits)

    def forecast_final(self, ts_emb, forecast_values, forecast_mask):
        prediction = self.forecast_head(ts_emb)
        return (
            forecast_mask * (prediction - forecast_values) ** 2
        ).sum() / forecast_mask.sum()
