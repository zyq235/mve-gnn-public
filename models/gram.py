"""GRAM for patient-level binary outcomes on longitudinal ICU variables."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRAMEncoder(nn.Module):
    """Ontology-attentive code embeddings followed by a GRU."""

    def __init__(
        self,
        *,
        num_nodes: int,
        ancestor_index: Tensor,
        ancestor_mask: Tensor,
        embedding_dim: int = 128,
        attention_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        use_values: bool = True,
    ) -> None:
        super().__init__()
        if ancestor_index.ndim != 2 or ancestor_mask.shape != ancestor_index.shape:
            raise ValueError(
                "ancestor_index and ancestor_mask must be matching 2-D tensors"
            )
        if ancestor_index.shape[0] <= 0:
            raise ValueError("At least one leaf variable is required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.num_variables = int(ancestor_index.shape[0])
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_values = bool(use_values)
        self.register_buffer("ancestor_index", ancestor_index.long().clone())
        self.register_buffer("ancestor_mask", ancestor_mask.bool().clone())
        self.register_buffer(
            "leaf_index", torch.arange(self.num_variables, dtype=torch.long)
        )

        self.node_embedding = nn.Embedding(int(num_nodes), self.embedding_dim)
        self.attention_hidden = nn.Linear(
            self.embedding_dim * 2, int(attention_dim)
        )
        self.attention_score = nn.Linear(int(attention_dim), 1, bias=False)
        self.value_embedding = (
            nn.Embedding(self.num_variables, self.embedding_dim)
            if self.use_values
            else None
        )
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.node_embedding.weight, -0.1, 0.1)
        if self.value_embedding is not None:
            nn.init.uniform_(self.value_embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.attention_hidden.weight)
        nn.init.zeros_(self.attention_hidden.bias)
        nn.init.xavier_uniform_(self.attention_score.weight)

    def code_representations(self) -> tuple[Tensor, Tensor]:
        """Return GRAM leaf representations and leaf-to-ancestor attention."""

        ancestor_embeddings = self.node_embedding(self.ancestor_index)
        leaf_embeddings = self.node_embedding(self.leaf_index).unsqueeze(1)
        leaf_embeddings = leaf_embeddings.expand_as(ancestor_embeddings)
        scores = self.attention_score(
            torch.tanh(
                self.attention_hidden(
                    torch.cat((leaf_embeddings, ancestor_embeddings), dim=-1)
                )
            )
        ).squeeze(-1)
        scores = scores.masked_fill(
            ~self.ancestor_mask, torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=1)
        representations = torch.sum(
            ancestor_embeddings * attention.unsqueeze(-1), dim=1
        )
        return representations, attention

    def forward(
        self,
        presence: Tensor,
        values: Tensor,
        lengths: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if presence.ndim != 3:
            raise ValueError("presence must have shape [batch, time, variable]")
        if values.shape != presence.shape:
            raise ValueError("values and presence must have identical shapes")
        if presence.shape[-1] != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} variables, got {presence.shape[-1]}"
            )
        if lengths.ndim != 1 or lengths.shape[0] != presence.shape[0]:
            raise ValueError("lengths must contain one value per batch item")
        if bool((lengths <= 0).any()) or bool(
            (lengths > presence.shape[1]).any()
        ):
            raise ValueError("Sequence lengths are outside the padded time dimension")

        code_embeddings, attention = self.code_representations()
        visit_embeddings = presence.float() @ code_embeddings
        if self.value_embedding is not None:
            visit_embeddings = (
                visit_embeddings + values.float() @ self.value_embedding.weight
            )
        visit_embeddings = torch.tanh(visit_embeddings)
        packed = pack_padded_sequence(
            visit_embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        patient_embedding = self.output_dropout(hidden[-1])
        if return_attention:
            return patient_embedding, attention
        return patient_embedding


class GRAMBinaryClassifier(nn.Module):
    """GRAM encoder plus demographics and a one-logit mortality head."""

    def __init__(
        self,
        *,
        num_nodes: int,
        ancestor_index: Tensor,
        ancestor_mask: Tensor,
        demo_dim: int = 2,
        demo_hidden_dim: int = 16,
        embedding_dim: int = 128,
        attention_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        use_values: bool = True,
    ) -> None:
        super().__init__()
        self.demo_dim = int(demo_dim)
        self.encoder = GRAMEncoder(
            num_nodes=num_nodes,
            ancestor_index=ancestor_index,
            ancestor_mask=ancestor_mask,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_values=use_values,
        )
        if self.demo_dim > 0:
            self.demo_network: nn.Module | None = nn.Sequential(
                nn.Linear(self.demo_dim, int(demo_hidden_dim)),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_input = hidden_dim + int(demo_hidden_dim)
        else:
            self.demo_network = None
            classifier_input = hidden_dim
        self.classifier = nn.Linear(classifier_input, 1)

    def forward(
        self,
        presence: Tensor,
        values: Tensor,
        lengths: Tensor,
        demo: Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        encoded = self.encoder(
            presence, values, lengths, return_attention=return_attention
        )
        if return_attention:
            patient_embedding, attention = encoded
        else:
            patient_embedding = encoded
            attention = None
        if self.demo_network is not None:
            if demo is None or demo.ndim != 2 or demo.shape[1] != self.demo_dim:
                raise ValueError(
                    f"demo must have shape [batch, {self.demo_dim}]"
                )
            patient_embedding = torch.cat(
                (patient_embedding, self.demo_network(demo.float())), dim=1
            )
        logits = self.classifier(patient_embedding).squeeze(-1)
        if return_attention:
            assert attention is not None
            return logits, attention
        return logits

