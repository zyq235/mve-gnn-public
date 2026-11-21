import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, knn_graph

from .graphmae.utils import create_activation, create_norm


class GATLayer(nn.Module):
    """Apply graph attention, residual projection, normalization, and activation."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0,
                 activation="relu", norm="batchnorm", residual=True, k=None, negative_slope=0.2):
        super().__init__()
        self.k = k
        self.residual = residual
        self.heads = heads

        # Divide the requested output width evenly across concatenated heads.
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.out_channels_per_head = out_dim // heads

        self.conv = GATConv(
            in_channels=in_dim,
            out_channels=self.out_channels_per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
            negative_slope=negative_slope,
            add_self_loops=False
        )

        # Project the residual when the input and output widths differ.
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.norm = create_norm(norm)(out_dim) if norm else nn.Identity()
        self.act = create_activation(activation)

    def forward(self, x, edge_index=None, batch=None):
        """Apply attention on the given edges, or rebuild KNN edges within each graph if k is set."""

        if self.k is not None:
            edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        out = self.conv(x, edge_index)

        if self.residual:
            res = self.res_fc(x) if self.res_fc is not None else x
            out = out + res

        # 4. Norm + Activation
        out = self.norm(out)
        out = self.act(out)

        return out, edge_index


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout=0.1,
                 activation="relu",
                 norm="batchnorm",
                 residual=True,
                 heads=4,
                 encoding=False,
                 k=6
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim
        self.k = k

        layers = []

        # -------------------------
        # 1) Input Layer
        # -------------------------
        layers.append(
            GATLayer(
                in_dim=in_dim,
                out_dim=num_hidden,
                heads=heads,
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
                k=k
            )
        )

        # -------------------------
        # 2) Hidden Layers
        # -------------------------
        for _ in range(1, num_layers - 1):
            layers.append(
                GATLayer(
                    in_dim=num_hidden,
                    out_dim=num_hidden,
                    heads=heads,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    residual=residual,
                    k=k
                )
            )

        # -------------------------
        # 3) Output Layer
        # -------------------------
        layers.append(
            GATLayer(
                in_dim=num_hidden,
                out_dim=out_dim,
                heads=heads,
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
                k=k
            )
        )

        self.layers = nn.ModuleList(layers)

        self.head = nn.Identity()

    def forward(self, data, feat, return_hidden=False):
        x = feat
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        hidden_list = []

        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, new_edge_index = layer(x, edge_index, batch)
            hidden_list.append(x)
            edge_index = new_edge_index

        if return_hidden:
            return self.head(x), hidden_list, edge_index
        return self.head(x), edge_index

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)