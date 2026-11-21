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


def get_mask_indices(batch_data: Batch, mask_rate: float):
    """Sample masked nodes per graph so small graphs retain visible nodes."""
    device = batch_data.x.device
    if hasattr(batch_data, 'ptr'):
        ptr = batch_data.ptr
    else:
        counts = torch.bincount(batch_data.batch)
        ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(counts, dim=0)])

    masked_indices_list = []
    for i in range(ptr.numel() - 1):
        start_idx = ptr[i]
        end_idx = ptr[i + 1]
        num_nodes = end_idx - start_idx
        # print(num_nodes, mask_rate)
        num_mask = int(num_nodes * mask_rate)
        if num_mask > 0:
            perm = torch.randperm(num_nodes, device=device)
            local_mask_indices = perm[:num_mask]
            global_mask_indices = local_mask_indices + start_idx
            masked_indices_list.append(global_mask_indices)

    if len(masked_indices_list) > 0:
        mask_indices = torch.cat(masked_indices_list, dim=0)
    else:
        mask_indices = torch.tensor([], dtype=torch.long, device=device)
    return mask_indices


import torch
from torch_geometric.utils import to_dense_batch


def build_constrained_edge_index(batch_data, mask_indices=None, custom_batch=None):
    """Build within-group edges, excluding edges between two masked nodes."""
    device = batch_data.x.device
    num_nodes = batch_data.num_nodes

    # Use time-window groups when the caller provides them.
    batch_vector = custom_batch if custom_batch is not None else batch_data.batch

    if mask_indices is not None:

        if mask_indices.dtype == torch.bool:
            mask_bool = mask_indices
        else:
            mask_bool = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            mask_bool[mask_indices] = True

        # mask_dense: [B, N_max], valid_mask: [B, N_max]
        mask_dense, valid_mask = to_dense_batch(mask_bool.float(), batch_vector)
        mask_dense = mask_dense.bool()
        valid_mask = valid_mask.bool()

        adj = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)

        mask_source = mask_dense.unsqueeze(2)
        mask_target = mask_dense.unsqueeze(1)
        is_mask_to_mask = mask_source & mask_target

        adj = adj & (~is_mask_to_mask)

    else:

        dummy = torch.zeros(num_nodes, device=device)
        _, valid_mask = to_dense_batch(dummy, batch_vector)
        valid_mask = valid_mask.bool()

        adj = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)


    # 4. Dense to Sparse
    nonzero_indices = torch.nonzero(adj)
    batch_idx = nonzero_indices[:, 0]
    row_idx = nonzero_indices[:, 1]
    col_idx = nonzero_indices[:, 2]

    # Convert local adjacency coordinates to global node indices.
    if hasattr(batch_data, 'ptr') and custom_batch is None:
        ptr = batch_data.ptr
    else:
        counts = torch.bincount(batch_vector)
        ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(counts, dim=0)])

    edge_batch_offsets = ptr[batch_idx]
    global_row = row_idx + edge_batch_offsets
    global_col = col_idx + edge_batch_offsets

    new_edge_index = torch.stack([global_row, global_col], dim=0)

    return new_edge_index


def build_temporal_chain_edges(data, bidirectional=True):
    """Connect successive observations of the same variable within each stay."""
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = data.x.shape[0]

    df_dict = {
        'node_idx': np.arange(num_nodes),
        'var': data.x_var.detach().cpu().numpy().squeeze(),
        'time': data.x_time.detach().cpu().numpy().squeeze()
    }

    if hasattr(data, 'batch') and data.batch is not None:
        df_dict['batch'] = data.batch.detach().cpu().numpy()
        group_keys = ['batch', 'var']
    else:
        group_keys = ['var']

    df = pd.DataFrame(df_dict)

    sort_cols = group_keys + ['time']
    df = df.sort_values(by=sort_cols)

    src_indices = df['node_idx'].values[:-1]
    dst_indices = df['node_idx'].values[1:]

    valid_mask = np.ones(len(src_indices), dtype=bool)
    for key in group_keys:
        key_values = df[key].values
        valid_mask &= (key_values[:-1] == key_values[1:])

    final_src = src_indices[valid_mask]
    final_dst = dst_indices[valid_mask]

    edge_index = torch.as_tensor(
        np.stack((final_src, final_dst)), dtype=torch.long, device=data.x.device
    )

    if bidirectional:
        row, col = edge_index
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        edge_index = torch.unique(edge_index, dim=1)

    return edge_index

def build_premodel_edge_index(data, bidirectional=True):
    pass
