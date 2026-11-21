"""GRU-D baseline adapted from the reference STraTS implementation."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

from .timeseries_model import TimeSeriesModel


class GRUDCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_gamma = nn.Parameter(torch.zeros(1, input_size))
        self.b_gamma = nn.Parameter(torch.zeros(1, input_size))
        self.W_gamma_h = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.b_gamma_h = nn.Parameter(torch.zeros(1, hidden_size))

        self.W_z = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.W_r = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.W = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.U_z = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        self.U_r = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        self.U = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        self.V_z = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.V_r = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.V = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.b_z = nn.Parameter(torch.zeros(1, hidden_size))
        self.b_r = nn.Parameter(torch.zeros(1, hidden_size))
        self.b = nn.Parameter(torch.zeros(1, hidden_size))

        for parameter in (self.W_z, self.W_r, self.W, self.V_z, self.V_r, self.V):
            nn.init.xavier_uniform_(parameter)
        for parameter in (self.U_z, self.U_r, self.U):
            nn.init.orthogonal_(parameter)
        self.reset_dropout_masks()

    def reset_dropout_masks(self):
        self._dropout_mask = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        self._recurrent_dropout_mask = [
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        ]
        self._masking_dropout_mask = [
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        ]

    @jit.script_method
    def forward(
        self,
        input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_t, m_t, delta_t = input
        h_tm1, xprev = state
        xprev = xprev * (1 - m_t) + m_t * x_t

        if self.dropout > 0 and self._dropout_mask[0].ndim == 0:
            self._dropout_mask = [
                F.dropout(torch.ones_like(x_t), self.dropout, self.training)
                for _ in range(3)
            ]
            self._recurrent_dropout_mask = [
                F.dropout(torch.ones_like(h_tm1), self.dropout, self.training)
                for _ in range(3)
            ]
            self._masking_dropout_mask = [
                F.dropout(torch.ones_like(m_t), self.dropout, self.training)
                for _ in range(3)
            ]

        gamma_t = torch.exp(
            -torch.clamp(self.W_gamma * delta_t + self.b_gamma, min=0)
        )
        x_t = m_t * x_t + (1 - m_t) * gamma_t * xprev
        gamma_ht = torch.exp(
            -torch.clamp(torch.matmul(delta_t, self.W_gamma_h) + self.b_gamma_h, min=0)
        )
        h_tm1 = gamma_ht * h_tm1

        if self.dropout > 0:
            x_t_z, x_t_r, x_t_h = (
                self._dropout_mask[0] * x_t,
                self._dropout_mask[1] * x_t,
                self._dropout_mask[2] * x_t,
            )
            m_t_z, m_t_r, m_t_h = (
                self._masking_dropout_mask[0] * m_t,
                self._masking_dropout_mask[1] * m_t,
                self._masking_dropout_mask[2] * m_t,
            )
            h_tm1_z, h_tm1_r, h_tm1_h = (
                self._recurrent_dropout_mask[0] * h_tm1,
                self._recurrent_dropout_mask[1] * h_tm1,
                self._recurrent_dropout_mask[2] * h_tm1,
            )
        else:
            x_t_z = x_t_r = x_t_h = x_t
            m_t_z = m_t_r = m_t_h = m_t
            h_tm1_z = h_tm1_r = h_tm1_h = h_tm1

        z_t = torch.sigmoid(
            torch.matmul(x_t_z, self.W_z)
            + torch.matmul(h_tm1_z, self.U_z)
            + torch.matmul(m_t_z, self.V_z)
            + self.b_z
        )
        r_t = torch.sigmoid(
            torch.matmul(x_t_r, self.W_r)
            + torch.matmul(h_tm1_r, self.U_r)
            + torch.matmul(m_t_r, self.V_r)
            + self.b_r
        )
        h_t = torch.tanh(
            torch.matmul(x_t_h, self.W)
            + torch.matmul(r_t * h_tm1_h, self.U)
            + torch.matmul(m_t_h, self.V)
            + self.b
        )
        h_t = (1 - z_t) * h_tm1 + z_t * h_t
        return h_t, (h_t, xprev)


class GRUD(jit.ScriptModule):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = GRUDCell(*cell_args)

    @jit.script_method
    def forward(
        self,
        input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x_steps = input[0].unbind(1)
        mask_steps = input[1].unbind(1)
        delta_steps = input[2].unbind(1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        self.cell.reset_dropout_masks()
        for index in range(len(x_steps)):
            output, state = self.cell(
                (x_steps[index], mask_steps[index], delta_steps[index]), state
            )
            outputs += [output]
        return torch.stack(outputs).transpose(0, 1)


class GRUD_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)
        self.grud = GRUD(args.V, args.hid_dim, args.dropout)
        self.V = args.V
        self.hid_dim = args.hid_dim
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x_t, m_t, delta_t, seq_len, demo, labels=None):
        batch_size = x_t.size(0)
        initial_state = (
            torch.zeros((batch_size, self.hid_dim), device=x_t.device),
            torch.zeros((batch_size, self.V), device=x_t.device),
        )
        ts_emb = self.grud((x_t, m_t, delta_t), initial_state)
        hidden_dim = ts_emb.size(2)
        index = (seq_len - 1)[:, None, None].repeat((1, 1, hidden_dim))
        ts_emb = torch.gather(ts_emb, 1, index)[:, 0, :]
        ts_emb = self.dropout(ts_emb)
        demo_emb = self.demo_emb(demo)
        logits = self.binary_head(torch.cat((ts_emb, demo_emb), dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)
