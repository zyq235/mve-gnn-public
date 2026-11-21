from typing import Optional, Tuple
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.utils import to_dense_batch
from models.GAT import GATLayer
from models.GIN import GINLayer, MLP, ApplyNodeFunc
from edges_construct import build_constrained_edge_index, build_temporal_chain_edges, build_premodel_edge_index, get_mask_indices

class HybridGMAEEncoder(nn.Module):
    def __init__(self, in_dim, num_hidden, num_layers=4, nhead=4, dropout=0.0,
                 activation="prelu", norm="batchnorm", residual=True, k=10,
                 gnn_type="GAT", negative_slope=0.2,
                 learn_eps=False, aggr="add"):
        super().__init__()
        self.k = k
        self.num_layers = num_layers

        assert num_layers >= 2, "HybridGMAEEncoder requires at least 2 layers."

        self.split_layer = num_layers // 2

        self.layers_stage1 = nn.ModuleList()
        self.layers_stage2 = nn.ModuleList()

        # --- Stage 1 Construction ---
        # Layer 0: Input -> Hidden
        if gnn_type == "gat":
            self.layers_stage1.append(
                GATLayer(
                    in_dim=in_dim,
                    out_dim=num_hidden,
                    heads=nhead,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    residual=residual,
                    negative_slope=negative_slope,
                    k=None
                )
            )
            # Layer 1 ~ Split (Hidden -> Hidden)
            for _ in range(1, self.split_layer):
                self.layers_stage1.append(
                    GATLayer(
                        in_dim=num_hidden,
                        out_dim=num_hidden,
                        heads=nhead,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        k=None
                    )
                )

            # --- Stage 2 Construction ---
            # Layer Split ~ End (Hidden -> Hidden)
            for _ in range(self.split_layer, num_layers):
                self.layers_stage2.append(
                    GATLayer(
                        in_dim=num_hidden,
                        out_dim=num_hidden,
                        heads=nhead,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        k=None
                    )
                )
        elif gnn_type == "gin":
            mlp0 = ApplyNodeFunc(
                MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm),
                activation=activation,
                norm=norm
            )
            self.layers_stage1.append(
                GINLayer(mlp0, in_dim, num_hidden, residual=residual, learn_eps=learn_eps, aggr=aggr, k=None)
            )
            # Layer 1 ~ Split (Hidden -> Hidden)
            for _ in range(1, self.split_layer):
                mlp_h = ApplyNodeFunc(
                    MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm),
                    activation=activation,
                    norm=norm
                )
                self.layers_stage1.append(
                    GINLayer(mlp_h, num_hidden, num_hidden, residual=residual, learn_eps=learn_eps, aggr=aggr, k=None)
                )
    def forward(self, x, initial_edge_index, temporal_edge_index, subgraph_batch):

        # --- Stage 1: Context Absorption ---
        for layer in self.layers_stage1:
            x, _ = layer(x, edge_index=initial_edge_index, batch=subgraph_batch)

        # --- Intermediate: Dynamic Graph Construction ---
        # Limit KNN search to each adaptive time window.
        knn_edge_index = knn_graph(x, k=self.k, batch=subgraph_batch, loop=False)

        # Combine learned KNN edges with the temporal chains.
        refined_edge_index = torch.cat([temporal_edge_index, knn_edge_index], dim=1)
        refined_edge_index = torch.unique(refined_edge_index, dim=1)
        # --- Stage 2: Semantic Refinement ---
        for layer in self.layers_stage2:
            x, _ = layer(x, edge_index=refined_edge_index, batch=subgraph_batch)

        return x, refined_edge_index

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class GMAE_Regressor(nn.Module):
    def __init__(self, hidden_dim, out_dim=1, cat_emb=False):
        super().__init__()
        if cat_emb:
            input_dim = hidden_dim * 3
        else:
            input_dim = hidden_dim * 2
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, node_emb, var_emb):
        combined = torch.cat([node_emb, var_emb], dim=-1)
        pred_val = self.regressor(combined)
        return pred_val.squeeze(-1)


class CVE(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        int_dim = int(np.sqrt(hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = torch.matmul(x, self.W1) + self.b1
        x = self.activation(x)
        x = torch.matmul(x, self.W2)
        return x


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


class EHRLargeGraphPreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            encoder_type: str,
            decoder_type: str,
            activation: str = "prelu",
            feat_drop: float = 0.1,
            attn_drop: float = 0.1,
            mask_rate: float = 0.3,
            drop_edge_rate: float = 0.0,
            alpha_l: float = 2,
            loss_lmb: float = 1.0,
            var2ind_path: str = "data/mimic3/var2ind.json",
            device: str = "cpu",
            norm: str = "layernorm",
            k: int = 10,
            mlp_layers: int = 2,
            aggr="add",
            residual: bool = True,
            negative_slope: float = 0.2,
            cat_var: bool = False,
            mask_t1: float = 6.0,  #Lower bound of the safe window (hours)
            mask_t2: float = 12.0,
            mse_lmb: float = 0.5,
    ):
        super(EHRLargeGraphPreModel, self).__init__()
        self._mask_rate = mask_rate
        self.mask_t1 = mask_t1
        self.mask_t2 = mask_t2
        self.emb_dim = in_dim
        self.lmb = loss_lmb
        self.mse_lmb = mse_lmb
        self.cat_var = cat_var
        # 1. Embeddings
        self.cve_value = CVE(hid_dim=self.emb_dim)

        with open(var2ind_path) as f:
            self.Vardict = json.load(f)
        print("Emb length:", len(self.Vardict))
        self.variable_emb = nn.Embedding(len(self.Vardict), self.emb_dim)

        # 2. Hybrid Encoder (Layer 0-2 Constrained, Layer 3-4 KNN)
        if self.cat_var:
            in_dim = in_dim * 2
        self.encoder = HybridGMAEEncoder(
            in_dim=in_dim,
            num_hidden=num_hidden,
            num_layers=num_layers,
            nhead=nhead,
            dropout=feat_drop,
            k=k,
            norm=norm,
            gnn_type=encoder_type,
            residual=residual,
            negative_slope=negative_slope,
        )

        # 3. Decoder & Heads
        dec_in_dim = num_hidden  # GAT output dim
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # Decoder (GAT based)
        if decoder_type=="gat":
            self.decoder = GATLayer(dec_in_dim, in_dim, heads=nhead,
                        dropout=feat_drop,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        k=None)
        elif decoder_type=="gin":
            mlp_dec = MLP(mlp_layers, dec_in_dim, in_dim, in_dim, activation=activation, norm=norm)
            self.decoder = GINLayer(mlp_dec, dec_in_dim, in_dim, residual=False, learn_eps=False, aggr=aggr, k=None)

        self.regressor = GMAE_Regressor(hidden_dim=self.emb_dim, out_dim=1, cat_emb=self.cat_var)

        # 4. Loss
        self.sce_criterion = partial(sce_loss, alpha=alpha_l)
        self.mse_criterion = nn.MSELoss()
        self.device = device

    def get_temporal_mask_indices(self, g, mask_rate):
        """
        Generate mask indices with a dynamic safe window.
        Nodes within the first 'window' hours are preserved (not masked).
        """
        device = g.x.device
        # Handle batch pointer
        if hasattr(g, 'ptr'):
            ptr = g.ptr
        else:
            counts = torch.bincount(g.batch)
            ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(counts, dim=0)])

        masked_indices_list = []

        # Ensure x_time exists and is flattened. Assuming x_time is in minutes.
        times = g.x_time.squeeze()

        for i in range(ptr.numel() - 1):
            start_idx = ptr[i]
            end_idx = ptr[i + 1]

            # Sample a dynamic window threshold for the current graph (hours -> minutes)
            # Uniformly sample between t1 and t2
            window_hours = (self.mask_t1 + torch.rand(1, device=device) * (self.mask_t2 - self.mask_t1)).item()
            window_threshold_min = window_hours * 60.0

            # Get local time values
            local_times = times[start_idx:end_idx]
            if local_times.numel() > 0:
                relative_times = local_times - local_times.min()
            else:
                relative_times = local_times
            # Assume local_times are absolute. If they restart from 0 for each patient, use directly.
            # If they are continuous global timestamps, you might need: local_times - local_times.min()
            # Here we assume dataset provides relative times (or starts near 0) per patient.
            # Identify candidates: nodes appearing AFTER the safe window
            candidate_mask = relative_times > window_threshold_min
            candidate_indices_local = torch.nonzero(candidate_mask).squeeze(-1)

            num_candidates = candidate_indices_local.numel()

            if num_candidates > 0:
                # Calculate number of nodes to mask based on the candidates
                num_mask = int(num_candidates * mask_rate)

                if num_mask > 0:
                    # Randomly sample from candidates
                    perm = torch.randperm(num_candidates, device=device)
                    selected_local = candidate_indices_local[perm[:num_mask]]

                    # Convert to global indices
                    masked_indices_list.append(start_idx + selected_local)

        if len(masked_indices_list) > 0:
            return torch.cat(masked_indices_list, dim=0)
        else:
            return torch.tensor([], dtype=torch.long, device=device)


    def setup_input(self, g):
        """Embed observations and replace masked values with variable embeddings."""
        # try:
        var_emb = self.variable_emb(g.x_var.long().squeeze())
        # except:
        #     print(g.x_var.tolist())
        #     exit(-1)
        val_emb = self.cve_value(g.x_val.float())
        if self.cat_var:
            x_init = torch.concat([var_emb, val_emb], dim=1)
        else:
            x_init = var_emb + val_emb

        g.x = x_init
        # generate Mask
        # mask_indices = get_mask_indices(g, self._mask_rate)
        mask_indices = self.get_temporal_mask_indices(g, self._mask_rate)

        x_masked = x_init.clone()
        if mask_indices.numel() > 0:

            if self.cat_var:
                mask_val_emb = torch.zeros_like(val_emb)
                x_masked[mask_indices] = torch.concat([var_emb, mask_val_emb], dim=1)[mask_indices]
            else:
                x_masked[mask_indices] = var_emb[mask_indices]

        return x_masked, x_init, var_emb, mask_indices

    def construct_graphs(self, g, mask_indices, construct_type="FC"):
        """
        Stage 1: Temporal + (Intra - Mask2Mask)
        Stage 2: (Dynamic in Encoder) -> Temporal + KNN
        """
        # 1. Temporal Edges (Base Skeleton)
        temporal_edge_index = build_temporal_chain_edges(g, bidirectional=True)

        # 2. Subgraph Batch Vector
        subgraph_batch = g.batch_time

        # 3. Constrained Intra Edges (For Stage 1)
        if construct_type == "FC":
            intra_edge_index = build_constrained_edge_index(g, mask_indices, custom_batch=subgraph_batch)
        elif construct_type == "PreModel":
            intra_edge_index = build_premodel_edge_index(g, mask_indices, custom_batch=subgraph_batch)
        stage1_edge_index = torch.cat([temporal_edge_index, intra_edge_index], dim=1)
        stage1_edge_index = torch.unique(stage1_edge_index, dim=1)
        return stage1_edge_index, temporal_edge_index, subgraph_batch

    def embed(self, g):
        x_masked, x_init, var_emb, mask_indices = self.setup_input(g)
        stage1_edge_index, temporal_edge_index, subgraph_batch = self.construct_graphs(g, None)
        enc_rep, final_edge_index = self.encoder(
            x_init,
            stage1_edge_index,
            temporal_edge_index,
            subgraph_batch
        )
        return enc_rep, final_edge_index

    def forward(self, g):
        # 1. Setup Input
        x_masked, x_init, var_emb, mask_indices = self.setup_input(g)

        # 2. Construct Graphs
        stage1_edge_index, temporal_edge_index, subgraph_batch = self.construct_graphs(g, mask_indices)

        # 3. Hybrid Encoder Forward
        enc_rep, final_edge_index = self.encoder(
            x_masked,
            stage1_edge_index,
            temporal_edge_index,
            subgraph_batch
        )

        # 4. Decoding
        rep = self.encoder_to_decoder(enc_rep)

        recon, edge_index = self.decoder(rep, final_edge_index, batch=subgraph_batch)

        # 5. Loss Calculation
        if mask_indices.numel() > 0:
            # Reconstruction Loss (Feature)
            x_orig_masked = x_init[mask_indices]
            x_rec_masked = recon[mask_indices]
            sce_loss_val = self.sce_criterion(x_rec_masked, x_orig_masked)

            # Regression Loss (Value)
            val_orig_masked = g.x_val[mask_indices].squeeze()
            val_pred_masked = self.regressor(x_rec_masked, var_emb[mask_indices])
            mse_loss_val = self.mse_criterion(val_pred_masked, val_orig_masked)

            loss = self.lmb * sce_loss_val + self.mse_lmb * mse_loss_val
            loss_item = {"sce": sce_loss_val.item(), "mse": mse_loss_val.item()}
        else:
            loss = torch.tensor(0.0, device=g.x.device, requires_grad=True)
            loss_item = {"sce": 0.0, "mse": 0.0}

        return loss, loss_item


# ------------------------------------------------------------------------------
# 4. Build Model Wrapper
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    from types import SimpleNamespace
    from dataset.mimic3_patient_dataset import ICULargeGraphDataset
    # from torch.utils.data import DataLoader as Dataloader
    from torch_geometric.data import DataLoader
    import pickle

    pkl_pth = "../../data/mimic3/mimic_iii.pkl"
    with open(pkl_pth, "rb") as f:
        data, label, train_list, val_list, test_list = pickle.load(f)
    del data, label
    tsid_list = np.hstack((train_list, val_list, test_list))
    dataset = ICULargeGraphDataset(
        root="../../data/mimic3/",
        ts_id=tsid_list,
        # data_dir="path/to/data_dir",
        var2ind_path="../../data/mimic3/var2ind.json",
        var_mean_std_path="../../data/mimic3/var_mean_std.json",

    )


    args = SimpleNamespace(
        num_features=64, num_hidden=512, num_layers=4, num_heads=4,
        activation="prelu", in_drop=0.1, attn_drop=0.1,
        mask_rate=0.5, drop_edge_rate=0.0, alpha_l=2.0,
        var2ind_path="../../data/mimic3/var2ind.json",
        device="cuda"
    )
    model = build_model(args)
    print(model)
    data = next(iter(DataLoader(dataset, batch_size=2, shuffle=True)))
    loss, loss_item = model(data)

