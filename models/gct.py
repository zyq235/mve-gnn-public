"""Graph Convolutional Transformer adapted to project ICU variable codes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _xavier_initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GCTLayer(nn.Module):
    """One GCT attention/FFN block with structural masking."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_feedforward: int,
        dropout: float,
        attention_normalizer: str,
        head_aggregation: str,
    ) -> None:
        super().__init__()
        if attention_normalizer not in {"softmax", "sigmoid"}:
            raise ValueError("attention_normalizer must be softmax or sigmoid")
        if head_aggregation not in {"concat", "sum"}:
            raise ValueError("head_aggregation must be concat or sum")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_feedforward = int(num_feedforward)
        self.dropout = float(dropout)
        self.attention_normalizer = attention_normalizer
        self.head_aggregation = head_aggregation

        projection_dim = self.hidden_dim * self.num_heads
        self.query = nn.Linear(self.hidden_dim, projection_dim, bias=False)
        self.key = nn.Linear(self.hidden_dim, projection_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, projection_dim, bias=False)
        self.aggregate = (
            nn.Linear(projection_dim, self.hidden_dim, bias=False)
            if self.num_heads > 1 and head_aggregation == "concat"
            else None
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.ffn_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_feedforward)]
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.apply(_xavier_initialize)

    def _heads(self, tensor: Tensor) -> Tensor:
        batch_size, num_codes, _ = tensor.shape
        return tensor.view(
            batch_size, num_codes, self.num_heads, self.hidden_dim
        ).permute(0, 2, 1, 3)

    def _learned_attention(
        self, hidden: Tensor, valid_mask: Tensor, guide: Tensor | None
    ) -> Tensor:
        query = self._heads(self.query(hidden))
        key = self._heads(self.key(hidden)).transpose(-1, -2)
        scores = torch.matmul(query, key) / math.sqrt(self.hidden_dim)
        key_mask = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -10000.0)
        if guide is not None:
            scores = scores.masked_fill(~guide[:, None, :, :], -10000.0)
        if self.attention_normalizer == "softmax":
            attention = torch.softmax(scores, dim=-1)
        else:
            attention = torch.sigmoid(scores) * key_mask
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return attention

    def forward(
        self,
        hidden: Tensor,
        valid_mask: Tensor,
        *,
        guide: Tensor | None = None,
        prior: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden = hidden * valid_mask.unsqueeze(-1)
        attention = (
            prior[:, None, :, :].expand(-1, self.num_heads, -1, -1)
            if prior is not None
            else self._learned_attention(hidden, valid_mask, guide)
        )
        values = self._heads(self.value(hidden))
        post_attention = torch.matmul(attention, values)
        if self.num_heads == 1:
            post_attention = post_attention.squeeze(1)
        elif self.head_aggregation == "concat":
            post_attention = post_attention.permute(0, 2, 1, 3).contiguous()
            post_attention = post_attention.view(
                hidden.shape[0], hidden.shape[1], -1
            )
            assert self.aggregate is not None
            post_attention = self.aggregate(post_attention)
        else:
            post_attention = post_attention.sum(dim=1)
        post_attention = self.attention_norm(post_attention + hidden)

        post_ffn = post_attention
        for index, layer in enumerate(self.ffn_layers):
            post_ffn = layer(post_ffn)
            if index < len(self.ffn_layers) - 1:
                post_ffn = F.relu(post_ffn)
            post_ffn = F.dropout(post_ffn, self.dropout, self.training)
        output = self.ffn_norm(post_ffn + post_attention)
        return output * valid_mask.unsqueeze(-1), attention


class GCTBinaryClassifier(nn.Module):
    """GCT binary classifier for the first 24 hours of an ICU stay.

    The first layer uses a normalized structural prior.  It links the visit
    token to every observed variable and links variables belonging to the same
    clinical class.  Later layers learn attention subject to that guide, as in
    the original GCT prior/guide design.
    """

    def __init__(
        self,
        *,
        num_variables: int,
        demo_dim: int = 2,
        hidden_dim: int = 128,
        num_heads: int = 1,
        num_stacks: int = 3,
        num_feedforward: int = 2,
        dropout: float = 0.1,
        attention_normalizer: str = "softmax",
        head_aggregation: str = "concat",
        use_prior: bool = True,
        use_guide: bool = True,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        if num_variables <= 0 or hidden_dim <= 0:
            raise ValueError("num_variables and hidden_dim must be positive")
        if num_stacks <= 0 or num_feedforward <= 0 or num_heads <= 0:
            raise ValueError("stack, feedforward and head counts must be positive")
        self.num_variables = int(num_variables)
        self.demo_dim = int(demo_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_prior = bool(use_prior)
        self.use_guide = bool(use_guide)
        self.output_attentions = bool(output_attentions)

        self.code_embedding = nn.Embedding(
            self.num_variables + 1, self.hidden_dim, padding_idx=0
        )
        self.value_projection = nn.Linear(1, self.hidden_dim)
        self.time_projection = nn.Linear(1, self.hidden_dim)
        self.visit_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.layers = nn.ModuleList(
            [
                GCTLayer(
                    hidden_dim,
                    num_heads,
                    num_feedforward,
                    dropout,
                    attention_normalizer,
                    head_aggregation,
                )
                for _ in range(num_stacks)
            ]
        )
        if self.demo_dim > 0:
            self.demo_projection: nn.Module | None = nn.Sequential(
                nn.Linear(self.demo_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_input = self.hidden_dim * 2
        else:
            self.demo_projection = None
            classifier_input = self.hidden_dim
        self.classifier = nn.Linear(classifier_input, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.code_embedding.weight[1:])
        with torch.no_grad():
            self.code_embedding.weight[0].zero_()
        nn.init.xavier_uniform_(self.visit_token)
        self.value_projection.apply(_xavier_initialize)
        self.time_projection.apply(_xavier_initialize)
        if self.demo_projection is not None:
            self.demo_projection.apply(_xavier_initialize)
        _xavier_initialize(self.classifier)

    @staticmethod
    def _structure(
        code_mask: Tensor, type_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = code_mask.shape[0]
        visit_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=code_mask.device)
        valid = torch.cat((visit_mask, code_mask.bool()), dim=1)
        code_same_type = type_ids.unsqueeze(2).eq(type_ids.unsqueeze(1))
        code_same_type = code_same_type & code_mask.unsqueeze(2) & code_mask.unsqueeze(1)
        num_tokens = valid.shape[1]
        guide = torch.zeros(
            batch_size, num_tokens, num_tokens, dtype=torch.bool, device=valid.device
        )
        guide[:, 1:, 1:] = code_same_type
        guide[:, 0, :] = valid
        guide[:, :, 0] = valid
        eye = torch.eye(num_tokens, dtype=torch.bool, device=valid.device)
        guide = (guide | eye.unsqueeze(0)) & valid.unsqueeze(2) & valid.unsqueeze(1)
        prior = guide.float()
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return valid, guide, prior

    @staticmethod
    def _kl_regularization(
        attentions: list[Tensor], valid_mask: Tensor
    ) -> Tensor:
        if len(attentions) < 2:
            return attentions[0].new_zeros(())
        query_mask = valid_mask[:, None, :, None].float()
        terms = []
        for previous, current in zip(attentions[:-1], attentions[1:]):
            term = previous * (
                torch.log(previous.clamp_min(1e-12))
                - torch.log(current.clamp_min(1e-12))
            )
            denominator = query_mask.sum().clamp_min(1.0) * previous.shape[1]
            terms.append((term * query_mask).sum() / denominator)
        return torch.stack(terms).mean()

    def forward(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        code_ids = batch["code_ids"].long()
        values = batch["values"].float()
        times = batch["times"].float()
        code_mask = batch["mask"].bool()
        type_ids = batch["type_ids"].long()
        if code_ids.shape != values.shape or code_ids.shape != times.shape:
            raise ValueError("code_ids, values and times must have identical shapes")
        if code_mask.shape != code_ids.shape or type_ids.shape != code_ids.shape:
            raise ValueError("mask and type_ids must match code_ids")
        if bool((code_ids < 0).any()) or bool((code_ids > self.num_variables).any()):
            raise ValueError("code_ids are outside the embedding vocabulary")

        code_hidden = (
            self.code_embedding(code_ids)
            + self.value_projection(values.unsqueeze(-1))
            + self.time_projection(times.unsqueeze(-1))
        )
        code_hidden = code_hidden * code_mask.unsqueeze(-1)
        visit = self.visit_token.expand(code_ids.shape[0], -1, -1)
        hidden = torch.cat((visit, code_hidden), dim=1)
        valid, guide, prior = self._structure(code_mask, type_ids)

        attentions: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            hidden, attention = layer(
                hidden,
                valid,
                guide=(
                    guide
                    if self.use_guide and (index > 0 or not self.use_prior)
                    else None
                ),
                prior=prior if self.use_prior and index == 0 else None,
            )
            attentions.append(attention)

        patient = hidden[:, 0]
        if self.demo_projection is not None:
            demo = batch.get("demo")
            if demo is None or demo.ndim != 2 or demo.shape[1] != self.demo_dim:
                raise ValueError(f"demo must have shape [batch, {self.demo_dim}]")
            patient = torch.cat((patient, self.demo_projection(demo.float())), dim=1)
        logits = self.classifier(patient).squeeze(-1)
        result: dict[str, Tensor | tuple[Tensor, ...]] = {
            "logits": logits,
            "kl_loss": (
                self._kl_regularization(attentions, valid)
                if self.use_prior
                else logits.new_zeros(())
            ),
        }
        if self.output_attentions:
            result["attentions"] = tuple(attentions)
        return result
