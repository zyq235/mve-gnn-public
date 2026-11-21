import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, knn_graph

# import dgl.function as fn
# import dgl.ops as ops
# from dgl.utils import expand_as_pair

from .graphmae.utils import create_activation, NormLayer, create_norm

class GINLayer(nn.Module):
    """A wrapper containing PyG's GINConv + ApplyNodeFunc + Residual"""
    def __init__(self, apply_func, in_dim, out_dim, residual=False, learn_eps=False, aggr="sum", k=None):
        super().__init__()
        self.apply_func = apply_func
        self.residual = residual and (in_dim == out_dim)
        self.k=k
        self.conv = GINConv(
            nn=self.apply_func,
            eps=0,
            train_eps=learn_eps,
            aggr=aggr
        )

        if residual:
            if in_dim != out_dim:
                # learnable linear projection for residual
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

    def forward(self, x, edge_index, batch):
        if self.k is not None:
            edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        out = self.conv(x, edge_index)

        # residual
        if self.residual:
            out = out + x
        elif self.res_fc is not None:
            out = out + self.res_fc(x)

        return out, edge_index

class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 k=10
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim

        # For last layer, if encoding=True
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        layers = []

        # -------------------------
        # 1) Only one layer
        # -------------------------
        if num_layers == 1:
            mlp = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                mlp = ApplyNodeFunc(mlp, norm=norm, activation=activation)

            layers.append(
                GINLayer(mlp, in_dim, out_dim, residual=last_residual, learn_eps=learn_eps, aggr=aggr, k=k)
            )

        # -------------------------
        # 2) Multi-layer GIN
        # -------------------------
        else:
            # Input layer (no residual)
            mlp0 = ApplyNodeFunc(
                MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm),
                activation=activation,
                norm=norm
            )
            layers.append(
                GINLayer(mlp0, in_dim, num_hidden, residual=residual, learn_eps=learn_eps, aggr=aggr)
            )

            # Hidden layers
            for _ in range(1, num_layers - 1):
                mlp_h = ApplyNodeFunc(
                    MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm),
                    activation=activation,
                    norm=norm
                )
                layers.append(
                    GINLayer(mlp_h, num_hidden, num_hidden, residual=residual, learn_eps=learn_eps, aggr=aggr)
                )

            # Last layer
            mlp_last = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                mlp_last = ApplyNodeFunc(mlp_last, activation=activation, norm=norm)

            layers.append(
                GINLayer(mlp_last, num_hidden, out_dim, residual=last_residual, learn_eps=learn_eps, aggr=aggr)
            )

        self.layers = nn.ModuleList(layers)
        self.head = nn.Identity()

    def forward(self, data, feat, return_hidden=False):
        x, edge_index = feat, data.edge_index
        hidden_list = []

        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, edge_index = layer(x, edge_index, data.batch)
            hidden_list.append(x)

        if return_hidden:
            return self.head(x), hidden_list, edge_index
        return self.head(x), edge_index

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)




class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)