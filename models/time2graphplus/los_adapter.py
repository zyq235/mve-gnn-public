"""Multiclass Time2Graph+ training without changing the mortality core."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from models.time2graphplus.core.model_gat import Time2GraphGAT
from models.time2graphplus.utils.gat_utils import GATDataloader, GATDataset


class LOSTime2GraphGAT(Time2GraphGAT):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("safe_small_batches", True)
        super().__init__(*args, **kwargs)

    def save_los_core(self, path):
        """Save the fitted shapelets/GAT without requiring a fitted XGBoost head."""
        torch.save({key: value for key, value in self.__dict__.items()
                    if key != "xgb"}, path)

    def load_los_core(self, path, device):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.__dict__.update(state)
        self.gat = self.gat.to(device)
        self.cuda = torch.device(device).type == "cuda"
        return self

    def fit_los(self, x, labels):
        labels = np.asarray(labels, dtype=np.int64)
        if set(np.unique(labels)) != set(range(10)):
            raise ValueError("Time2Graph+ LOS training requires all ten classes")
        if x.shape[1] != self.num_segment * self.seg_length:
            raise ValueError("Time2Graph+ segments do not match the 24h input")

        scaled = self._Time2GraphGAT__preprocess_input_data(x)
        self.learn_shapelets(
            x=scaled, y=labels, num_segment=self.num_segment,
            data_size=x.shape[-1],
        )
        features, adjacency = self.__gat_features__(scaled, train=True)
        counts = np.bincount(labels, minlength=10).astype(np.float64)
        weights = torch.as_tensor(
            len(labels) / (10 * counts), dtype=torch.float64
        )
        device = torch.device("cuda" if self.cuda else "cpu")
        self.gat = self.gat.to(device)
        weights = weights.to(device)
        optimizer = torch.optim.Adam(self.gat.parameters(), lr=self.lr, weight_decay=5e-4)
        source = GATDataset(feat=features, adj=adjacency, y=labels)
        for _ in range(self.niter):
            for feat, adj, target in GATDataloader(
                source, batch_size=self.batch_size, shuffle=True
            ):
                self.gat.train()
                optimizer.zero_grad(set_to_none=True)
                logits = self.gat(feat.to(device), adj.to(device))
                loss = F.nll_loss(logits, target.to(device).long(), weight=weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Time2Graph+ LOS GAT produced non-finite loss")
                loss.backward()
                optimizer.step()
        return self

    def los_embeddings(self, x, *, fit_feature_scaler=False):
        scaled = self._Time2GraphGAT__preprocess_input_data(x)
        features, adjacency = self.__gat_features__(scaled)
        return self._Time2GraphGAT__gat_hidden_feature(
            feat=features, adj=adjacency, X=x, train=fit_feature_scaler
        )
