import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGraphConv
from torch_geometric.utils import softmax, scatter, to_dense_batch, to_dense_adj
from models.graphmae.mimic_graph_extract import build_model
from models.graphmae.type_group_indices import normalize_batched_type_indices

try:
    from kmeans_pytorch import kmeans
except ImportError:
    kmeans = None


def batch_kmeans_processing(x, batch, num_clusters, batch_size=None):
    """
    Args:
        x: [Total_Nodes, Hidden_Dim]
        batch: [Total_Nodes]
        num_clusters: int (K)
        batch_size: int

    Returns:
        local_cluster_ids: [Total_Nodes]
    """
    device = x.device
    if batch_size is None:
        batch_size = batch.max().item() + 1

    local_cluster_ids = torch.zeros_like(batch)

    for i in range(batch_size):
        mask = (batch == i)
        if not mask.any():
            continue

        x_sub = x[mask]

        if x_sub.size(0) < num_clusters:

            c_ids = torch.arange(x_sub.size(0), device=device)
        else:
            try:
                c_ids, _ = kmeans(
                    X=x_sub,
                    num_clusters=num_clusters,
                    distance='euclidean',
                    device=device,
                )
                c_ids = c_ids.to(device)
            except Exception as e:
                print(f"KMeans failed for graph {i}: {e}. Fallback to random.")
                c_ids = torch.randint(0, num_clusters, (x_sub.size(0),), device=device)

        local_cluster_ids[mask] = c_ids.long()

    return local_cluster_ids


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        # output: [1, Seq_Len, Dim]
        return self.pe[:x.size(1), :].unsqueeze(0)


class AttentionPooling(nn.Module):
    """
    Input: Node Embeddings [N, D], Group Index [N]
    Output: Group Embeddings [Num_Groups, D]
    """

    def __init__(self, in_channels):
        super().__init__()
        self.attn_gate = nn.Linear(in_channels, 1)

    def forward(self, x, group_index, num_groups=None):
        scores = self.attn_gate(x).squeeze(-1)
        alpha = softmax(scores, group_index)

        x_weighted = x * alpha.unsqueeze(-1)
        out = scatter(x_weighted, group_index, dim=0, dim_size=num_groups, reduce='add')

        return out


class HierarchicalEHRModel(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 args,
                 num_classes_type=10,
                 num_clusters_k=8,
                 num_demo=2,
                 num_heads=4,
                 num_layers=2,
                 dropout=0.3,
                 aggr_type="attn",
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes_type = num_classes_type
        self.num_clusters_k = num_clusters_k
        self.dropout = dropout
        self.aggr_type = aggr_type
        variable_to_class = getattr(args, "variable_to_class", None)
        if variable_to_class is None:
            self.register_buffer(
                "variable_to_class_lookup", None, persistent=False
            )
        else:
            self.register_buffer(
                "variable_to_class_lookup",
                torch.as_tensor(variable_to_class, dtype=torch.long).view(-1).clone(),
                persistent=False,
            )
        # --- 1. Node Encoder (GNN Backbone) ---
        self.pre_model = build_model(args)

        # --- 2. Class-aware Pooling ---
        self.pool_attn = AttentionPooling(hidden_dim)
        self.diffpool_proj1 = nn.Linear(hidden_dim, num_clusters_k)
        self.gnn_coarse = DenseGraphConv(hidden_dim, hidden_dim)
        self.diffpool_proj2 = nn.Linear(hidden_dim, 1)
        # --- 3. Cross-Class Interaction (Transformer) ---
        max_groups = max(num_classes_type, num_clusters_k)
        self.group_embeddings = nn.Embedding(max_groups, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim * 4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.node_time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.slot_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.type_ln = nn.LayerNorm(hidden_dim)
        # --- 4. Prediction Head ---
        # Patient Level Readout (Attention over groups)
        self.readout_gate = nn.Linear(hidden_dim, 1)

        # Final MLP
        self.demo_encoder = nn.Sequential(
            nn.Linear(num_demo, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output Logit
        )

    def get_group_indices(self, x, data, batch_type, mode='type'):

        batch = data.batch
        if mode == 'type':
            return batch_type, self.num_classes_type

        elif mode == 'cluster':
            if kmeans is not None:
                cluster_idx = batch_kmeans_processing(x, batch, self.num_clusters_k, batch_size=2)
                return cluster_idx, self.num_clusters_k
            else:
                print("Warning: torch_cluster not installed, using random clusters.")
                return torch.randint(0, self.num_clusters_k, (x.size(0),), device=x.device), self.num_clusters_k

        elif mode == "time":
            batch_time = data.batch_time
            num_time_len = torch.unique(batch_time).size(0)
            return batch_time, num_time_len

        elif mode == 'diff_pool':
            # --- DiffPool Implementation ---
            # x: [Total_Nodes, Hidden_Dim] -> s_logits: [Total_Nodes, Num_Clusters]
            s_logits = self.diffpool_proj1(x)
            s = F.softmax(s_logits, dim=-1)  # [Total_Nodes, Num_Clusters]
            return s, self.num_clusters_k
        else:
            raise ValueError("Unknown mode")

    def forward(self, data, mode='type'):
        if mode == "type":
            data = normalize_batched_type_indices(
                data,
                self.num_classes_type,
                variable_to_class=self.variable_to_class_lookup,
            )
        x, edge_index, batch = data.x_val, data.edge_index, data.batch

        # --- 1. Node Encoding ---
        enc, edge_index = self.pre_model.embed(data)

        # --- 2. Grouping Strategy ---
        if mode == "diff":
            x_dense, mask = to_dense_batch(enc, batch)
            adj = to_dense_adj(edge_index, batch, max_num_nodes=x_dense.size(1))

            # --- Stage 1: Pooling (N -> K) ---
            s1_logits = self.diffpool_proj1(x_dense)
            s1_logits = s1_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
            s1 = F.softmax(s1_logits, dim=-1)
            x_pool1 = torch.bmm(s1.transpose(1, 2), x_dense)

            # 1.3 A_pool1 = S1_T * A * S1 => [B, K, K]
            adj_pool1 = torch.bmm(torch.bmm(s1.transpose(1, 2), adj), s1)

            # --- Stage 2: Processing (GNN on Coarse Graph) ---
            x_pool1 = self.gnn_coarse(x_pool1, adj_pool1)
            x_pool1 = F.relu(x_pool1)

            # --- Stage 3: Global Pooling (K -> 1) ---
            s2_logits = self.diffpool_proj2(x_pool1)
            s2 = F.softmax(s2_logits, dim=1)  # Softmax

            # 3.2 X_final = S2_T * X_pool1 => [B, 1, D]
            x_final = torch.bmm(s2.transpose(1, 2), x_pool1)
            x_final = x_final.squeeze(1)  # [B, D]
            patient_emb = x_final
        elif mode == "node_time":
            sub_enc = enc
            sub_batch = batch
            sub_time = data.x_time  # [N_sub, 1]

            h_input, mask = to_dense_batch(sub_enc, sub_batch, batch_size=data.num_graphs)
            h_time_val, _ = to_dense_batch(sub_time, sub_batch, batch_size=data.num_graphs)

            h_time_val = h_time_val / 1440.0
            time_emb = self.node_time_proj(h_time_val)  # [B, L, D]
            h_input = h_input + time_emb
            # 5. Transformer Masking
            src_key_padding_mask = ~mask

        else:
            group_info, num_groups = self.get_group_indices(enc, data, data.batch_type, mode)
            if mode == "cluster":
                global_group_idx = batch * num_groups + group_info
            elif mode == "time":
                if self.aggr_type == "attn":
                    slot_emb = self.pool_attn(enc, group_info)

                    slot_batch_idx = scatter(batch, group_info, dim=0, reduce='min')

                    h_input, mask = to_dense_batch(slot_emb, slot_batch_idx)

                    B, T, _ = h_input.size()
                    relative_time = torch.arange(T, device=h_input.device, dtype=torch.float32)
                    relative_time = relative_time.view(1, T, 1).expand(B, -1, -1)
                    pos_emb = self.time_proj(relative_time)
                    h_input = h_input + pos_emb
                    src_key_padding_mask = ~mask
                elif self.aggr_type == "combine":
                    slot_emb_attn = self.pool_attn(enc, group_info)
                    slot_emb_max = scatter(enc, group_info, dim=0, reduce='max')

                    slot_emb = torch.cat([slot_emb_attn, slot_emb_max], dim=-1)  # [Num_Slots, 2*Hidden]
                    slot_emb = self.slot_fusion(slot_emb)  # [Num_Slots, Hidden]

                    slot_batch_idx = scatter(batch, group_info, dim=0, reduce='min')

                    # h_input: [B, Max_T, D], mask: [B, Max_T]
                    h_input, mask = to_dense_batch(slot_emb, slot_batch_idx)

                    slot_real_time = scatter(data.x_time, group_info, dim=0, reduce='min')  # [Num_Slots, 1]
                    h_time, _ = to_dense_batch(slot_real_time, slot_batch_idx)

                    h_time_norm = h_time / 1440.0

                    pos_emb = self.time_proj(h_time_norm)
                    h_input = h_input + pos_emb

                    src_key_padding_mask = ~mask
                elif self.aggr_type == "max":
                    slot_emb = scatter(enc, group_info, dim=0, reduce='max')
                    slot_batch_idx = scatter(batch, group_info, dim=0, reduce='min')
                    h_input, mask = to_dense_batch(slot_emb, slot_batch_idx)
                    B, T, _ = h_input.size()
                    # [0, 1, 2, ..., T-1]
                    relative_time = torch.arange(T, device=h_input.device, dtype=torch.float32)
                    # [B, T, 1]
                    relative_time = relative_time.view(1, T, 1).expand(B, -1, -1)

                    pos_emb = self.time_proj(relative_time)
                    h_input = h_input + pos_emb

                    src_key_padding_mask = ~mask
            elif mode == "diff_pool":
                s = group_info
                x_dense, mask = to_dense_batch(enc, batch)
                s_dense, _ = to_dense_batch(s, batch)

                h_attn = torch.bmm(s_dense.transpose(1, 2), x_dense)
                x_expanded = x_dense.unsqueeze(2)
                s_penalty = torch.log(s_dense.unsqueeze(-1) + 1e-9)

                soft_masked_x = x_expanded + s_penalty
                soft_masked_x = soft_masked_x.masked_fill(~mask.unsqueeze(2).unsqueeze(3), -1e9)

                h_max = soft_masked_x.max(dim=1)[0]

                if self.aggr_type == "attn":
                    h_input = h_attn
                elif self.aggr_type == "max":
                    h_input = h_max
                elif self.aggr_type == "combine":
                    h_cat = torch.cat([h_attn, h_max], dim=-1)
                    h_input = self.slot_fusion(h_cat)
                else:
                    raise ValueError(f"Unknown aggr_type: {self.aggr_type}")

                h_input = self.type_ln(h_input)
                group_ids = torch.arange(num_groups, device=x.device).unsqueeze(0).expand(data.num_graphs, -1)
                type_emb = self.group_embeddings(group_ids)
                h_input = h_input + type_emb

                src_key_padding_mask = None

            elif mode == "type":
                global_group_idx = group_info
                total_groups = data.num_graphs * num_groups

                if self.aggr_type == "attn":
                    h_pooled_flat = self.pool_attn(enc, global_group_idx, num_groups=total_groups)
                elif self.aggr_type == 'max':
                    h_pooled_flat = scatter(enc, global_group_idx, dim=0, dim_size=total_groups, reduce='max')
                elif self.aggr_type == 'combine':
                    # 3. Combine: Attn + Max -> Linear Projection
                    h_attn = self.pool_attn(enc, global_group_idx, num_groups=total_groups)
                    h_max = scatter(enc, global_group_idx, dim=0, dim_size=total_groups, reduce='max')
                    # Concatenate and Fuse
                    h_cat = torch.cat([h_attn, h_max], dim=-1)  # [Total_Groups, 2*D]
                    h_pooled_flat = self.slot_fusion(h_cat)
                else:
                    raise ValueError(f"Unknown aggr_type: {self.aggr_type}")
                h_input = h_pooled_flat.view(data.num_graphs, num_groups, self.hidden_dim)
                h_input = self.type_ln(h_input)
                # Mask Calculation
                group_counts = scatter(torch.ones_like(global_group_idx, dtype=torch.float),
                                       global_group_idx, dim=0, dim_size=total_groups, reduce='add')
                group_counts = group_counts.view(data.num_graphs, num_groups)
                src_key_padding_mask = (group_counts == 0)

                # Add Embedding
                group_ids = torch.arange(num_groups, device=x.device).unsqueeze(0).expand(data.num_graphs, -1)
                type_emb = self.group_embeddings(group_ids)
                h_input = h_input + type_emb
            else:
                global_group_idx = group_info
                total_groups = data.num_graphs * num_groups

                h_pooled_flat = self.pool_attn(enc, global_group_idx, num_groups=total_groups)
                h_input = h_pooled_flat.view(data.num_graphs, num_groups, self.hidden_dim)

                # Mask Calculation
                group_counts = scatter(torch.ones_like(global_group_idx, dtype=torch.float),
                                           global_group_idx, dim=0, dim_size=total_groups, reduce='add')
                group_counts = group_counts.view(data.num_graphs, num_groups)
                src_key_padding_mask = (group_counts == 0)

                # Add Embedding
                group_ids = torch.arange(num_groups, device=x.device).unsqueeze(0).expand(data.num_graphs, -1)
                type_emb = self.group_embeddings(group_ids)
                h_input = h_input + type_emb

        if mode != "diff":
            h_trans = self.transformer(h_input, src_key_padding_mask=src_key_padding_mask)

            # --- 4. Readout ---
            scores = self.readout_gate(h_trans)
            if src_key_padding_mask is not None:
                scores = scores.masked_fill(src_key_padding_mask.unsqueeze(-1), -1e9)

            weights = F.softmax(scores, dim=1)
            patient_emb = (h_trans * weights).sum(dim=1)

        # --- 5. Prediction ---

        demo_emb = self.demo_encoder(data.demo.view(data.num_graphs, -1))
        final_emb = torch.cat([patient_emb, demo_emb], dim=1)
        logits = self.classifier(final_emb)

        return logits
