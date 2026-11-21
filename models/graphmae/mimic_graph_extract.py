from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GIN import GIN
import numpy as np
import json
from models.GAT import GAT
# from .gcn import GCN
# from .dot_gat import DotGAT
# from .loss_func import sce_loss
# from graphmae.utils import create_norm, drop_edge
from torch_geometric.data import Data
from models.graphmae.mimic_largegraph_premodel import EHRLargeGraphPreModel


class GMAE_Regressor(nn.Module):
    def __init__(self, hidden_dim, out_dim=1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, node_emb, var_emb):
        """Predict raw values from the decoded nodes and their variable embeddings."""
        # var_emb = variable_emb_layer(var_indices)  # [N, hidden_dim]

        combined = torch.cat([node_emb, var_emb], dim=-1)  # [N, 2 * hidden_dim]

        pred_val = self.regressor(combined)  # [N, out_dim]
        return pred_val.squeeze(-1)

def build_model(args):
    model = EHRLargeGraphPreModel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        nhead=args.num_heads,
        activation=args.activation,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        mask_rate=args.mask_rate,
        drop_edge_rate=args.drop_edge_rate,
        alpha_l=args.alpha_l,
        var2ind_path="data/mimic3/var2ind.json",
        loss_lmb=getattr(args, "loss_lmb", 1.0),
        mse_lmb=getattr(args, "mse_lmb", 0.5),
        device=args.device,
        norm=args.norm,
        encoder_type="gat",
        decoder_type="gat",
        k=args.knn,
        cat_var=args.cat_var,
        negative_slope=args.negative_slope,
        mask_t1=-100,
        mask_t2=-100

    )
    return model



def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            heads=nhead,
            # nhead=nhead,
            # nhead_out=nhead_out,
            # concat_out=concat_out,
            activation=activation,
            # feat_drop=dropout,
            # attn_drop=attn_drop,
            # negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            k=15
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class CVE(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        int_dim = int(np.sqrt(hid_dim))
        self.W1 = nn.Parameter(torch.empty(1,int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    # def forward(self, x):
    #     # x: bsz, max_len
    #     x = torch.unsqueeze(x, -1)
    #     x = torch.matmul(x,self.W1) + self.b1[None,None,:] # bsz,max_len,int_dim
    #     x = self.activation(x)
    #     x = torch.matmul(x,self.W2) #bsz,max_len,hid_dim
    #     return x

    def forward(self, x):
        # x: [N] or [N, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [N, 1]


        x = torch.matmul(x, self.W1) + self.b1  # [N, int_dim]
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # [N, hid_dim]
        return x


class EHRPreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "mlp",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            loss_lmb=0.2,
    ):
        super(EHRPreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.emb_dim = in_dim
        self.cve_value = CVE(hid_dim=self.emb_dim)
        with open("data/mimic3/var2ind.json") as f:
            self.Vardict = json.load(f)

        self.variable_emb = nn.Embedding(len(self.Vardict), self.emb_dim)

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            # enc_num_hidden = num_hidden // nhead
            enc_num_hidden = num_hidden
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,

        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,

        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        # elif encoder_type == "gat":
        #     self.encoder_to_decoder = nn.Linear(dec_in_dim * nhead, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.regressor = GMAE_Regressor(hidden_dim=self.emb_dim, out_dim=1)
        # * setup loss function
        self.sce_criterion = self.setup_loss_fn("sce", alpha_l)
        self.mse_criterion = self.setup_loss_fn("mse", alpha_l)
        self.lmb = loss_lmb
        self.drop_out = 0.2

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, var_emb, val_emb, mask_rate=0.3):
        num_nodes = var_emb.size(0)
        device = var_emb.device
        # rand = torch.rand(num_nodes, device=device)
        # mask_nodes = torch.where(rand < mask_rate)[0]
        # keep_nodes = torch.where(rand >= mask_rate)[0]
        if hasattr(g, 'ptr'):
            ptr = g.ptr
        else:
            _, counts = torch.unique(g.batch, return_counts=True)
            ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(0)])
        mask_nodes_list = []
        keep_nodes_list = []
        for i in range(ptr.numel() - 1):
            start_idx = ptr[i]
            end_idx = ptr[i + 1]
            num_nodes_i = end_idx - start_idx

            num_mask_i = int(mask_rate * num_nodes_i)

            perm = torch.randperm(num_nodes_i, device=device)

            mask_idx_local = perm[:num_mask_i]
            keep_idx_local = perm[num_mask_i:]

            mask_nodes_list.append(start_idx + mask_idx_local)
            keep_nodes_list.append(start_idx + keep_idx_local)

        mask_nodes = torch.cat(mask_nodes_list)
        keep_nodes = torch.cat(keep_nodes_list)
        out_x = val_emb + var_emb
        # masked_x = self.enc_mask_token.repeat(num_nodes,1) + var_emb
        masked_x = var_emb
        # out_x = (val_emb + var_emb).clone()
        # mask_token = torch.zeros_like(x[0])  # [F]
        # if len(mask_nodes) > 0:
        out_x[mask_nodes] = masked_x[mask_nodes]

        use_g = g.clone()
        use_g.x = out_x

        return use_g, out_x, (mask_nodes, keep_nodes)

    def emb_layer(self, g):
        val = g.x
        var = g.var.long()
        var_emb = self.variable_emb(var)
        val_emb = self.cve_value(val)
        # x = val + var_emb
        # x = F.dropout(x, self.drop_out, self.training)
        return var_emb, val_emb, val

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        var_emb, val_emb, val_values = self.emb_layer(g)
        # g.x = emb_x
        mse_loss, sce_loss = self.mask_attr_prediction(g, var_emb, val_emb, val_values)
        loss = self.lmb * sce_loss + mse_loss
        loss_item = {"sce": sce_loss.item(), "mse":mse_loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, g, var_emb, val_emb, val_values):
        val_emb = val_emb
        x_init = var_emb+val_emb
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, var_emb, val_emb, self._mask_rate)

        # if self._drop_edge_rate > 0:
        #     use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        # else:
        use_g = pre_use_g

        enc_rep, all_hidden, new_edge_index = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        # if self._decoder_type not in ("mlp", "linear"):
        #     # * remask, re-mask
        #     rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon, new_edge_index = self.decoder(pre_use_g, rep)

        x_init = x_init[mask_nodes]
        val_init = val_values[mask_nodes]
        x_rec = recon[mask_nodes]
        val_rec = self.regressor(x_rec, var_emb[mask_nodes])
        mse_loss = self.mse_criterion(val_init, val_rec)
        sce_loss = self.sce_criterion(x_rec, x_init)
        # loss = self.lmb * sce_loss + mse_loss
        return mse_loss, sce_loss

    def embed(self, g):
        var_emb, val_emb, val = self.emb_layer(g)
        emb_x = var_emb + val_emb
        # g.x = emb_x
        rep, new_edge_index = self.encoder(g, emb_x, return_hidden=False)
        return rep, new_edge_index

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

if __name__ == "__main__":
    from models.graphmae.utils import build_args
    args = build_args()

    model = build_model(args)
    print(model)
