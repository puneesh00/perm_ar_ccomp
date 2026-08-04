# src/tab_completion/model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


NUMERICAL = 0
CATEGORICAL = 1

OBSERVED = 0
QUERY = 1
IGNORED = 2


@dataclass
class TableTensorBatch:
    """
    Dense episode tensor batch.

    x_num:
        Normalized numerical values. Shape [B, N, D].
        Dummy 0 for non-numerical cells.

    x_cat:
        Local categorical ids. Shape [B, N, D].
        Dummy 0 for non-categorical cells.

    col_types:
        0 numerical, 1 categorical. Shape [D].

    cat_cardinalities:
        Number of valid categories per column. Shape [D].
        0 or 1 for numerical columns.

    cat_decode_types:
        Which categorical embedding/unembedding type each column uses. Shape [D].

        For v0, set cat_decode_types[j] = j for per-column categorical decoders,
        or set all categorical feature columns to a shared type if desired.
    """
    x_num: torch.Tensor
    x_cat: torch.Tensor
    col_types: torch.Tensor
    cat_cardinalities: torch.Tensor
    cat_decode_types: torch.Tensor


@dataclass
class ModelOutput:
    """
    Model predictions for all episode cells.

    num_mu:
        Numerical point prediction. Shape [B, N, D].

    cat_logits:
        Categorical logits. Shape [B, N, D, K_max].
        Invalid categories are already masked to -inf.

    h:
        Final hidden cell representations. Shape [B, N, D, d_model].
    """
    num_mu: torch.Tensor
    cat_logits: torch.Tensor
    h: torch.Tensor


@dataclass
class ModelConfig:
    d_model: int = 128
    max_episode_rows: int = 4096
    max_cols: int = 128
    k_max: int = 32
    num_cat_decode_types: int = 128
    num_row_layers: int = 2
    num_row_context_layers: int = 1
    n_heads: int = 4
    dropout: float = 0.1


class CellTokenizer(nn.Module):
    """
    Builds one embedding per episode cell.

    token_ij =
      value_embedding
    + row_embedding
    + column_embedding
    + type_embedding
    + role_embedding
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        self.row_emb = nn.Embedding(cfg.max_episode_rows, d)
        self.col_emb = nn.Embedding(cfg.max_cols, d)
        self.type_emb = nn.Embedding(2, d)
        self.role_emb = nn.Embedding(3, d)

        # Category-type-conditioned categorical input embeddings:
        # shape [num_cat_decode_types, K_max, d_model]
        self.cat_value_emb = nn.Parameter(
            torch.randn(cfg.num_cat_decode_types, cfg.k_max, d) * 0.02
        )

        # Null value embedding by column type:
        # null numerical, null categorical
        self.null_value_emb = nn.Embedding(2, d)

        # v0 numerical embedding: MLP(xtilde)
        self.num_value_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        batch: TableTensorBatch,
        observed_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        observed_mask, query_mask:
            Bool tensors of shape [B, N, D].
        """
        x_num = batch.x_num
        x_cat = batch.x_cat

        B, N, D = x_num.shape
        device = x_num.device

        if N > self.cfg.max_episode_rows:
            raise ValueError(f"N={N} exceeds max_episode_rows={self.cfg.max_episode_rows}.")
        if D > self.cfg.max_cols:
            raise ValueError(f"D={D} exceeds max_cols={self.cfg.max_cols}.")

        col_types = batch.col_types.to(device=device, dtype=torch.long)
        cat_decode_types = batch.cat_decode_types.to(device=device, dtype=torch.long)

        type_ids = col_types.view(1, 1, D).expand(B, N, D)
        cat_type_ids = cat_decode_types.view(1, 1, D).expand(B, N, D)

        # Roles: ignored by default, then observed/query.
        roles = torch.full((B, N, D), IGNORED, dtype=torch.long, device=device)
        roles = torch.where(observed_mask, torch.full_like(roles, OBSERVED), roles)
        roles = torch.where(query_mask, torch.full_like(roles, QUERY), roles)

        if torch.any(observed_mask & query_mask):
            raise ValueError("A cell cannot be both observed and queried.")

        # Start with null value embeddings by type.
        value_vec = self.null_value_emb(type_ids)  # [B, N, D, d]

        is_observed = roles == OBSERVED
        is_num = type_ids == NUMERICAL
        is_cat = type_ids == CATEGORICAL

        # Numerical observed values.
        num_obs = is_observed & is_num
        if torch.any(num_obs):
            num_vec_all = self.num_value_mlp(x_num.unsqueeze(-1))
            value_vec = torch.where(num_obs.unsqueeze(-1), num_vec_all, value_vec)

        # Categorical observed values.
        cat_obs = is_observed & is_cat
        if torch.any(cat_obs):
            x_cat_clamped = x_cat.clamp(min=0, max=self.cfg.k_max - 1).long()
            cat_type_clamped = cat_type_ids.clamp(
                min=0,
                max=self.cfg.num_cat_decode_types - 1,
            )

            # [B, N, D, d]
            cat_vec_all = self.cat_value_emb[cat_type_clamped, x_cat_clamped]
            value_vec = torch.where(cat_obs.unsqueeze(-1), cat_vec_all, value_vec)

        row_ids = torch.arange(N, device=device).view(1, N, 1).expand(B, N, D)
        col_ids = torch.arange(D, device=device).view(1, 1, D).expand(B, N, D)

        token = (
            value_vec
            + self.row_emb(row_ids)
            + self.col_emb(col_ids)
            + self.type_emb(type_ids)
            + self.role_emb(roles)
        )

        return self.norm(token)


class TypedCategoricalHead(nn.Module):
    """
    Category-type-specific unembedding.

    For query cell in column j:
        decode_type = cat_decode_types[j]
        logits_k = <h_ij, W_unemb[decode_type, k]> + b[decode_type, k]

    This avoids forcing category 0/1 to share the exact same decoder
    across all categorical columns.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model
        self.unemb = nn.Parameter(
            torch.randn(cfg.num_cat_decode_types, cfg.k_max, d) * (d ** -0.5)
        )
        self.bias = nn.Parameter(torch.zeros(cfg.num_cat_decode_types, cfg.k_max))

    def forward(
        self,
        h: torch.Tensor,
        cat_decode_types: torch.Tensor,
        cat_cardinalities: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:
            [B, N, D, d_model]

        cat_decode_types:
            [D]

        cat_cardinalities:
            [D]

        returns:
            logits [B, N, D, K_max]
        """
        B, N, D, d = h.shape
        device = h.device

        cat_decode_types = cat_decode_types.to(device=device, dtype=torch.long)
        cat_cardinalities = cat_cardinalities.to(device=device, dtype=torch.long)

        type_ids = cat_decode_types.view(1, 1, D).expand(B, N, D)
        type_ids = type_ids.clamp(min=0, max=self.cfg.num_cat_decode_types - 1)

        # [B, N, D, K, d]
        W = self.unemb[type_ids]
        b = self.bias[type_ids]

        logits = torch.einsum("bndh,bndkh->bndk", h, W) + b

        k_ids = torch.arange(self.cfg.k_max, device=device).view(1, 1, 1, self.cfg.k_max)
        valid = k_ids < cat_cardinalities.view(1, 1, D, 1).clamp(min=1)

        logits = logits.masked_fill(~valid, float("-inf"))
        return logits


class CellwiseCompletionModel(nn.Module):
    """
    v0 query-based table model.

    The model supports two conditioning modes per task:

    Transductive mode:
        context_row_mask is None.
        All rows can communicate through the row-context encoder and column
        summaries use all observed cells.

    Inductive-row mode:
        context_row_mask is provided, shape [B, N], True for context rows.
        Context rows attend only to context rows.
        Query rows attend to context rows plus themselves.
        Column summaries use context-row observed cells only.

    Use inductive-row mode for TabPFN/TabICL-style target prediction and
    label+feature tasks. Use transductive mode for random-cell/table
    completion tasks.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        self.tokenizer = CellTokenizer(cfg)

        row_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=4 * d,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=cfg.num_row_layers)

        if cfg.num_row_context_layers > 0:
            row_context_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg.n_heads,
                dim_feedforward=4 * d,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.row_context_encoder = nn.TransformerEncoder(
                row_context_layer,
                num_layers=cfg.num_row_context_layers,
            )
        else:
            self.row_context_encoder = None

        self.col_summary_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.final_norm = nn.LayerNorm(d)

        self.num_head = nn.Linear(d, 1)
        self.cat_head = TypedCategoricalHead(cfg)

    def _make_inductive_row_attn_mask(
        self,
        context_row_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build row self-attention mask for one batch element.

        context_row_mask:
            [N], True for context rows, False for query rows.

        Returns:
            [N, N] boolean mask where True means attention is blocked.

        Rule:
            context target rows attend to context source rows only.
            query target rows attend to context source rows plus themselves.
        """
        device = context_row_mask.device
        N = context_row_mask.shape[0]

        source_is_context = context_row_mask.view(1, N).expand(N, N)
        target_is_query = (~context_row_mask).view(N, 1).expand(N, N)
        same_row = torch.eye(N, dtype=torch.bool, device=device)

        allowed = source_is_context | (target_is_query & same_row)
        return ~allowed  # PyTorch bool mask: True means blocked.

    def _run_row_context_encoder(
        self,
        row_state: torch.Tensor,
        context_row_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        row_state:
            [B, N, d]

        context_row_mask:
            None for transductive mode, or [B, N] True for context rows.
        """
        if self.row_context_encoder is None:
            return row_state

        if context_row_mask is None:
            return self.row_context_encoder(row_state)

        B, N, d = row_state.shape
        outs = []
        for b in range(B):
            attn_mask = self._make_inductive_row_attn_mask(context_row_mask[b])
            out_b = self.row_context_encoder(row_state[b : b + 1], mask=attn_mask)
            outs.append(out_b)

        return torch.cat(outs, dim=0)

    def forward(
        self,
        batch: TableTensorBatch,
        observed_mask: torch.Tensor,
        query_mask: torch.Tensor,
        context_row_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        observed_mask, query_mask:
            Bool tensors [B, N, D].

        context_row_mask:
            None means transductive table-completion mode.

            If provided, shape [B, N], True for context rows and False for query
            rows. This enables inductive-row mode, where query rows cannot use
            other query rows through row attention or column summaries.
        """
        H = self.tokenizer(batch, observed_mask, query_mask)
        B, N, D, d = H.shape

        if context_row_mask is not None:
            context_row_mask = context_row_mask.to(device=H.device, dtype=torch.bool)
            if context_row_mask.shape != (B, N):
                raise ValueError(
                    f"context_row_mask shape {tuple(context_row_mask.shape)} "
                    f"must be {(B, N)}."
                )

        # Row-wise encoder over columns. This is within-row only, so it cannot
        # leak information between query rows.
        H_row = self.row_encoder(H.reshape(B * N, D, d)).reshape(B, N, D, d)

        # Row-context encoder over row summaries. This is where we enforce
        # inductive row semantics if context_row_mask is provided.
        row_state = H_row.mean(dim=2)  # [B, N, d]
        row_state = self._run_row_context_encoder(row_state, context_row_mask)

        # Column summaries.
        # Transductive: use all observed cells.
        # Inductive: use only context-row observed cells, preventing query row A
        # from seeing query row B through a column summary.
        if context_row_mask is None:
            col_observed_mask = observed_mask
        else:
            col_observed_mask = observed_mask & context_row_mask[:, :, None]

        obs_float = col_observed_mask.float().unsqueeze(-1)  # [B, N, D, 1]
        denom = obs_float.sum(dim=1).clamp_min(1.0)          # [B, D, 1]
        col_state = (H_row * obs_float).sum(dim=1) / denom   # [B, D, d]
        col_state = self.col_summary_mlp(col_state)

        H_final = self.final_norm(
            H_row
            + row_state[:, :, None, :]
            + col_state[:, None, :, :]
        )

        num_mu = self.num_head(H_final).squeeze(-1)
        cat_logits = self.cat_head(
            H_final,
            batch.cat_decode_types,
            batch.cat_cardinalities,
        )

        return ModelOutput(num_mu=num_mu, cat_logits=cat_logits, h=H_final)
