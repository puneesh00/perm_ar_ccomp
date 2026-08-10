# src/tab_completion/model_tabpfn_v1.py
"""
Reference reimplementation of TabPFN v1's core architecture (Hollmann et al.,
2022, "TabPFN: A Transformer That Solves Small Tabular Classification
Problems in a Second"), adapted from automl/nanoTabPFN's model.py
(github.com/automl/nanoTabPFN, Apache-2.0) -- a minimal educational
reimplementation of the original by the same AutoML group, validated against
the paper.

Purpose: a capacity-matched, known-good REFERENCE model to disentangle
whether our custom two-stream axial model (model_perm_ar.py) is stuck on
this prior because of a modeling bug, insufficient training time, or a real
capacity/task-difficulty ceiling. Same synthetic prior, same eval harness
(logreg/rf/xgb context baselines from scripts/train_synthetic.py), same
optimizer recipe -- only the model differs.

Deliberately much simpler than model_perm_ar.py, and this is a property of
the real architecture, not a simplification we're introducing for this
diagnostic:
  - a single nn.Linear(1, d_model) embeds every feature cell's raw
    (per-episode, context-only) normalized scalar value -- shared across
    ALL columns. No column-identity embedding, no row-identity embedding.
    Column/row identity is purely implicit in which attention axis-group a
    cell is routed through at each layer, never an added vector.
  - a separate nn.Linear(1, d_model) embeds the target column; context rows
    get their true label, query rows get the context-label mean as a
    placeholder (mean-imputation, not a learned "unknown" token).
  - per layer: feature-axis (within-row) self-attention, then datapoint-axis
    (cross-row) self-attention with train/test masking -- context rows
    attend to context rows only, query rows attend to context rows only
    (never to each other, so one query row's prediction can't depend on any
    other query row sharing the episode) -- then a 2-layer GELU MLP.
  - features are normalized using CONTEXT-ONLY mean/std, computed fresh per
    episode: query rows must never influence preprocessing statistics, which
    is what a real train/test split enforces and our main synthetic data
    pipeline (which standardizes over the full table) does not.

Only supports target-column prediction (row-token-style models don't have
per-cell addressability for arbitrary completion), which is exactly the
`target` sampler / cat_acc metric this whole investigation has been tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TabPFNV1Config:
    d_model: int = 128
    n_heads: int = 4
    mlp_hidden: int = 512
    n_layers: int = 6
    n_classes: int = 2
    dropout: float = 0.0


class FeatureEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, n_context: int) -> torch.Tensor:
        # x: [B, N, D_feat] -> [B, N, D_feat, d_model]
        x = x.unsqueeze(-1)
        mean = x[:, :n_context].mean(dim=1, keepdim=True)
        std = x[:, :n_context].std(dim=1, keepdim=True) + 1e-6
        x = ((x - mean) / std).clamp(min=-100.0, max=100.0)
        return self.linear(x)


class TargetEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, y_context: torch.Tensor, n_rows: int) -> torch.Tensor:
        # y_context: [B, n_context] -> [B, N, 1, d_model]
        mean = y_context.mean(dim=1, keepdim=True)
        n_query = n_rows - y_context.shape[1]
        pad = mean.expand(-1, n_query)
        y = torch.cat([y_context, pad], dim=1)
        return self.linear(y.unsqueeze(-1)).unsqueeze(2)


class TabPFNV1Layer(nn.Module):
    def __init__(self, cfg: TabPFNV1Config):
        super().__init__()
        d = cfg.d_model
        self.feature_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True, dropout=cfg.dropout)
        self.datapoint_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True, dropout=cfg.dropout)
        self.linear1 = nn.Linear(d, cfg.mlp_hidden)
        self.linear2 = nn.Linear(cfg.mlp_hidden, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, src: torch.Tensor, n_context: int) -> torch.Tensor:
        B, N, C, d = src.shape

        x = src.reshape(B * N, C, d)
        x = self.feature_attn(x, x, x)[0] + x
        src = self.norm1(x.reshape(B, N, C, d))

        x = src.transpose(1, 2).reshape(B * C, N, d)
        ctx = x[:, :n_context]
        left = self.datapoint_attn(ctx, ctx, ctx)[0]
        right = self.datapoint_attn(x[:, n_context:], ctx, ctx)[0]
        x = torch.cat([left, right], dim=1) + x
        src = self.norm2(x.reshape(B, C, N, d).transpose(1, 2))

        h = self.linear2(F.gelu(self.linear1(src)))
        return self.norm3(h + src)


class TabPFNV1Model(nn.Module):
    def __init__(self, cfg: TabPFNV1Config):
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = FeatureEncoder(cfg.d_model)
        self.target_encoder = TargetEncoder(cfg.d_model)
        self.layers = nn.ModuleList([TabPFNV1Layer(cfg) for _ in range(cfg.n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.n_classes),
        )

    def forward(self, x_feat: torch.Tensor, y_context: torch.Tensor, n_context: int) -> torch.Tensor:
        """
        x_feat: [B, N, D_feat] float, all rows' feature columns, context rows
            first (rows 0..n_context-1), query rows after.
        y_context: [B, n_context] float, context rows' true target class id.
        Returns logits [B, N - n_context, n_classes] for the query rows.
        """
        x = self.feature_encoder(x_feat, n_context)
        y = self.target_encoder(y_context, x.shape[1])
        src = torch.cat([x, y], dim=2)
        for layer in self.layers:
            src = layer(src, n_context)
        target_repr = src[:, n_context:, -1, :]
        return self.decoder(target_repr)
