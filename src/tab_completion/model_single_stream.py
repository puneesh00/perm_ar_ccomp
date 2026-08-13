# src/tab_completion/model_single_stream.py
"""
Single-stream reference-recipe model: TabPFN-v1's structural idea (one token
per cell, row-axis + col-axis attention, context/query masking, no
row/col-identity embeddings) but built from OUR OWN tokenizer/decoder
components (num_value_mlp, cat_value_emb, TypedCategoricalHead) instead of
TabPFN-v1's simpler scalar-only ones, plus the same informative per-column
context-derived placeholder added to the two-stream model's PermARTokenizer.

Purpose: isolate whether the rest of our architecture (value encoders,
per-column-type categorical embeddings, typed decode heads) is sound once
two confounds are removed: the two-stream parameter tax (see
model_perm_ar.py's compute_task_loss_onepass_batched docstring for why
two-stream costs ~2x params per layer) and the uninformative fixed-null
query placeholder. If this model reaches TabPFN-v1's accuracy, that's
evidence our embedding/decode design was never the problem -- only the
two-stream overhead and the placeholder were. If it doesn't, something else
in the design still needs finding.

Only supports parallel-style (no reveal-order) prediction: a single token
per cell can't simultaneously be "value-bearing evidence for later cells"
and "leak-free for its own prediction" the way two-stream's content/query
split can -- see model_perm_ar.py's module docstring for why two-stream
exists at all. That tradeoff is fine here because query cells never carry
their own true value in the first place (only the informative placeholder,
computed from OTHER cells) -- so unlike two-stream, no masking is needed to
prevent a query cell from leaking to itself; the only masking that matters
is cross-row, restricting which rows may serve as evidence for which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tab_completion.model import (
    NUMERICAL,
    CATEGORICAL,
    ModelConfig,
    ModelOutput,
    TableTensorBatch,
    TypedCategoricalHead,
    expand_per_col,
)
from tab_completion.sampling import CompletionTask
from tab_completion.factorization import FactorizationPlan
from tab_completion.model_perm_ar import (
    RANK_OBSERVED,
    RANK_NEVER,
    build_rank_tensor,
    build_rank_tensor_batched,
    get_context_row_mask_from_task,
    get_context_row_mask_batched,
    StepLossOutput,
)


class SingleStreamTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # No row_emb, no col_emb: matches the current (ablated) two-stream
        # tokenizer -- column/row identity is purely implicit in which
        # attention axis-group a cell is routed through.
        #
        # drop_type_origin_emb removes these two as well: type_emb (num vs
        # cat) and origin_emb (0 = observed this episode, 1 = a query cell)
        # were the last additive, non-value signals left in the tokenizer.
        # See ModelConfig.drop_type_origin_emb for the TabPFNV1 precedent.
        if not cfg.drop_type_origin_emb:
            self.type_emb = nn.Embedding(2, d)
            self.origin_emb = nn.Embedding(2, d)

        # unified_cat_encoding drops this entirely -- categorical cells are
        # cast to float and routed through num_value_mlp instead (see
        # forward()), so there is nothing for this table to encode.
        if not cfg.unified_cat_encoding:
            self.cat_value_emb = nn.Parameter(
                torch.randn(cfg.num_cat_decode_types, cfg.k_max, d) * 0.02
            )
        self.num_value_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.norm = nn.LayerNorm(d)

    def forward(self, batch: TableTensorBatch, rank: torch.Tensor) -> torch.Tensor:
        """
        rank: [B, N, D] long. Returns one token per cell: [B, N, D, d_model].
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
        type_ids = expand_per_col(col_types, B, N, D)
        cat_type_ids = expand_per_col(cat_decode_types, B, N, D)

        is_num = type_ids == NUMERICAL
        is_cat = type_ids == CATEGORICAL

        observed = rank == RANK_OBSERVED
        observed_num = observed & is_num
        observed_cat = observed & is_cat

        x_cat_clamped = x_cat.clamp(min=0, max=self.cfg.k_max - 1).long()
        cat_type_clamped = cat_type_ids.clamp(min=0, max=self.cfg.num_cat_decode_types - 1)

        if self.cfg.unified_cat_encoding:
            # TabPFNV1Model's actual recipe (model_tabpfn_v1.py's
            # FeatureEncoder, fed by train_tabpfn_v1_baseline.py:57-61):
            # every column becomes one continuous-valued channel -- numerics
            # keep their real value, categoricals are just their raw
            # (post-shuffle, non-ordinal) id cast to float -- and both go
            # through the SAME num_value_mlp. No cat_value_emb, no
            # per-column split on the encode side at all.
            x_unified = torch.where(is_num, x_num, x_cat_clamped.to(x_num.dtype))

            unified_sum = (x_unified * observed.to(x_unified.dtype)).sum(dim=1)
            unified_count = observed.to(x_unified.dtype).sum(dim=1).clamp(min=1.0)
            col_mean = unified_sum / unified_count  # [B, D]

            if self.cfg.context_normalize:
                unified_sq_sum = (x_unified.pow(2) * observed.to(x_unified.dtype)).sum(dim=1)
                col_var = (unified_sq_sum / unified_count - col_mean.pow(2)).clamp(min=0.0)
                col_std = torch.sqrt(col_var + 1e-6)  # [B, D]
                x_unified_input = (x_unified - col_mean.unsqueeze(1)) / col_std.unsqueeze(1)
                x_unified_input = x_unified_input.clamp(min=-100.0, max=100.0)
                placeholder_mean = torch.zeros_like(col_mean)
            else:
                x_unified_input = x_unified
                placeholder_mean = col_mean

            true_value = self.num_value_mlp(x_unified_input.unsqueeze(-1))
            placeholder_value = self.num_value_mlp(
                placeholder_mean.unsqueeze(1).expand(B, N, D).unsqueeze(-1)
            )

        else:
            # --- context(observed)-only column stats for numerical columns.
            # Used both to optionally re-standardize x_num per episode
            # (context_normalize) and, either way, to build the query
            # placeholder below. ---
            num_sum = (x_num * observed_num.to(x_num.dtype)).sum(dim=1)
            num_count = observed_num.to(x_num.dtype).sum(dim=1).clamp(min=1.0)
            col_mean_num = num_sum / num_count  # [B, D]

            if self.cfg.context_normalize:
                # Mirrors TabPFNV1Model.FeatureEncoder: every cell (context
                # AND query) is re-standardized using context-only mean/std,
                # freshly computed each episode -- instead of relying on the
                # static per-table z-score baked in at data-generation time
                # in synthetic_data_tabpfn.py, which pools context+query rows
                # and never adapts to whichever rows this episode's sampler
                # actually picked as context.
                num_sq_sum = (x_num.pow(2) * observed_num.to(x_num.dtype)).sum(dim=1)
                col_var_num = (num_sq_sum / num_count - col_mean_num.pow(2)).clamp(min=0.0)
                col_std_num = torch.sqrt(col_var_num + 1e-6)  # [B, D]
                x_num_input = (x_num - col_mean_num.unsqueeze(1)) / col_std_num.unsqueeze(1)
                x_num_input = x_num_input.clamp(min=-100.0, max=100.0)
                # In normalized space the context mean is 0 by construction,
                # so the placeholder is just the zero vector fed through the MLP.
                placeholder_mean_num = torch.zeros_like(col_mean_num)
            else:
                x_num_input = x_num
                placeholder_mean_num = col_mean_num

            # --- true value, used for observed cells ---
            num_vec_all = self.num_value_mlp(x_num_input.unsqueeze(-1))
            cat_vec_all = self.cat_value_emb[cat_type_clamped, x_cat_clamped]
            true_value = torch.where(is_num.unsqueeze(-1), num_vec_all, cat_vec_all)

            # --- informative placeholder, used for query cells. Same recipe
            # as the two-stream model's query stream: numerical gets the
            # observed-cell mean fed through the same value MLP a real value
            # would use; categorical gets the observed-frequency-weighted
            # average of that column's category embeddings (not a mean of raw
            # ids -- those are deliberately non-ordinal). ---
            placeholder_num_vec = self.num_value_mlp(
                placeholder_mean_num.unsqueeze(1).expand(B, N, D).unsqueeze(-1)
            )

            k_max = self.cfg.k_max
            x_cat_onehot = F.one_hot(x_cat_clamped, num_classes=k_max).to(dtype=self.cat_value_emb.dtype)
            x_cat_onehot = x_cat_onehot * observed_cat.unsqueeze(-1).to(x_cat_onehot.dtype)
            cat_counts = x_cat_onehot.sum(dim=1)  # [B, D, K]
            cat_totals = cat_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
            cat_probs = cat_counts / cat_totals

            cat_type_bd = cat_type_clamped[:, 0, :]  # [B, D] -- constant across rows
            emb_table_bd = self.cat_value_emb[cat_type_bd]  # [B, D, K, d]
            placeholder_cat_vec_bd = torch.einsum("bdk,bdkh->bdh", cat_probs, emb_table_bd)
            placeholder_cat_vec = placeholder_cat_vec_bd.unsqueeze(1).expand(
                B, N, D, placeholder_cat_vec_bd.shape[-1]
            )

            placeholder_value = torch.where(is_num.unsqueeze(-1), placeholder_num_vec, placeholder_cat_vec)

        value = torch.where(observed.unsqueeze(-1), true_value, placeholder_value)

        if self.cfg.drop_type_origin_emb:
            return self.norm(value)

        pos = self.type_emb(type_ids)
        is_query_cell = (rank != RANK_OBSERVED) & (rank != RANK_NEVER)
        origin = self.origin_emb(is_query_cell.long())

        return self.norm(value + pos + origin)


class SingleStreamAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [G, L, d] (self-attention: same tensor used as q, k, and v).
        attn_mask: bool [G, L, L] (True = allowed) or None (full attention).
        """
        G, L, d = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(G, L, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(G, L, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(G, L, H, hd).transpose(1, 2)

        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        else:
            mask = attn_mask.unsqueeze(1)
            has_any = mask.any(dim=-1, keepdim=True)
            safe_mask = mask | (~has_any)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=safe_mask, dropout_p=self.dropout if self.training else 0.0
            )
            out = out * has_any.to(out.dtype)

        out = out.transpose(1, 2).reshape(G, L, d)
        return self.out_proj(out)


def _col_axis_key_allowed(rank: torch.Tensor, context_row_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Returns bool [G, N] (G = B*D groups): whether row n is valid evidence
    for ANY reader in that group. Same rule regardless of the reader's own
    identity -- context/observed rows never read query rows, so this is
    naturally leak-free without needing a separate query stream.

    inductive_rows (context_row_mask given): key allowed iff that ROW is a
    context row -- a clean row-level split, matching TabPFN-v1 exactly.
    transductive (context_row_mask is None): key allowed iff that specific
    CELL is observed -- generalizes to random_cell/table-completion tasks
    with no clean row split.
    """
    B, N, D = rank.shape
    if context_row_mask is not None:
        return context_row_mask.unsqueeze(1).expand(B, D, N).reshape(B * D, N)
    observed = rank == RANK_OBSERVED
    return observed.permute(0, 2, 1).reshape(B * D, N)


class SingleStreamLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, axis: str):
        super().__init__()
        if axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
        self.axis = axis
        self.post_ln = cfg.post_ln
        d = cfg.d_model

        self.attn = SingleStreamAttention(d, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def _reshape_for_axis(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D, d = x.shape
        if self.axis == "row":
            return x.reshape(B * N, D, d)
        return x.permute(0, 2, 1, 3).reshape(B * D, N, d)

    def _reshape_back(self, x: torch.Tensor, B: int, N: int, D: int) -> torch.Tensor:
        d = x.shape[-1]
        if self.axis == "row":
            return x.reshape(B, N, D, d)
        return x.reshape(B, D, N, d).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        rank: torch.Tensor,
        context_row_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, D, d = x.shape
        x_in = self._reshape_for_axis(x)

        if self.axis == "row":
            # Full, unmasked: safe because query cells never carry their
            # true value, so no attention pattern among them can leak.
            attn_mask = None
        else:
            key_allowed = _col_axis_key_allowed(rank, context_row_mask)  # [G, N]
            G = key_allowed.shape[0]
            attn_mask = key_allowed.unsqueeze(1).expand(G, N, N)

        if self.post_ln:
            # TabPFNV1Layer's layout: attn/ffn operate on the raw (already
            # normalized-once, by the previous stage) input, residual add
            # happens first, THEN LayerNorm -- see ModelConfig.post_ln.
            attn_out = self.attn(x_in, attn_mask)
            x_in = self.norm1(x_in + self.dropout(attn_out))
            x_in = self.norm2(x_in + self.ffn(x_in))
        else:
            x_normed = self.norm1(x_in)
            attn_out = self.attn(x_normed, attn_mask)
            x_in = x_in + self.dropout(attn_out)
            x_in = x_in + self.ffn(self.norm2(x_in))
        return self._reshape_back(x_in, B, N, D)


class PairedSingleStreamLayer(nn.Module):
    """
    TabPFNV1Layer's layout (model_tabpfn_v1.py): row-axis attention, then
    col-axis attention, THEN one FFN application -- instead of
    SingleStreamLayer's default of a separate FFN after each single-axis
    attention. Same total attention ops, half the FFN applications, so
    num_row_layers can be doubled within the same FFN parameter/compute
    budget. Requires num_row_layers == num_row_context_layers (one paired
    row+col block per count) -- see SingleStreamModel.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.post_ln = cfg.post_ln
        d = cfg.d_model
        self.row_attn = SingleStreamAttention(d, cfg.n_heads, cfg.dropout)
        self.col_attn = SingleStreamAttention(d, cfg.n_heads, cfg.dropout)
        self.row_norm = nn.LayerNorm(d)
        self.col_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rank: torch.Tensor,
        context_row_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, D, d = x.shape

        # row axis: full, unmasked (safe -- query cells never carry their
        # true value, see SingleStreamLayer's docstring for why).
        x_row = x.reshape(B * N, D, d)
        if self.post_ln:
            # TabPFNV1Layer's layout: attn operates on the raw input,
            # residual add first, THEN LayerNorm -- see ModelConfig.post_ln.
            row_out = self.row_attn(x_row, None)
            x_row = self.row_norm(x_row + self.dropout(row_out))
        else:
            x_row_normed = self.row_norm(x_row)
            row_out = self.row_attn(x_row_normed, None)
            x_row = x_row + self.dropout(row_out)
        x = x_row.reshape(B, N, D, d)

        # col axis: masked by context_row_mask/observed-ness.
        x_col = x.permute(0, 2, 1, 3).reshape(B * D, N, d)
        key_allowed = _col_axis_key_allowed(rank, context_row_mask)  # [G, N]
        G = key_allowed.shape[0]
        attn_mask = key_allowed.unsqueeze(1).expand(G, N, N)
        if self.post_ln:
            col_out = self.col_attn(x_col, attn_mask)
            x_col = self.col_norm(x_col + self.dropout(col_out))
        else:
            x_col_normed = self.col_norm(x_col)
            col_out = self.col_attn(x_col_normed, attn_mask)
            x_col = x_col + self.dropout(col_out)
        x = x_col.reshape(B, D, N, d).permute(0, 2, 1, 3)

        if self.post_ln:
            x = self.ffn_norm(x + self.ffn(x))
        else:
            x = x + self.ffn(self.ffn_norm(x))
        return x


class SingleStreamModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = SingleStreamTokenizer(cfg)

        if cfg.tabpfn_style_layers:
            if cfg.num_row_layers != cfg.num_row_context_layers:
                raise ValueError(
                    "tabpfn_style_layers requires num_row_layers == "
                    f"num_row_context_layers (got {cfg.num_row_layers} vs "
                    f"{cfg.num_row_context_layers}) -- one paired row+col "
                    "block per count."
                )
            self.layers = nn.ModuleList(
                [PairedSingleStreamLayer(cfg) for _ in range(cfg.num_row_layers)]
            )
        else:
            axes = []
            n_row, n_col = cfg.num_row_layers, cfg.num_row_context_layers
            i = j = 0
            while i < n_row or j < n_col:
                if i < n_row:
                    axes.append("row")
                    i += 1
                if j < n_col:
                    axes.append("col")
                    j += 1
            if not axes:
                axes = ["row"]

            self.layers = nn.ModuleList([SingleStreamLayer(cfg, axis) for axis in axes])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.num_head = nn.Linear(cfg.d_model, 1)
        self.cat_head = TypedCategoricalHead(cfg)

    def forward(
        self,
        batch: TableTensorBatch,
        rank: torch.Tensor,
        context_row_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        x = self.tokenizer(batch, rank)
        use_checkpoint = self.training and self.cfg.activation_checkpointing
        for layer in self.layers:
            if use_checkpoint:
                x = checkpoint(
                    lambda x_in, _layer=layer: _layer(x_in, rank, context_row_mask),
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                x = layer(x, rank, context_row_mask)

        h = self.final_norm(x)
        num_mu = self.num_head(h).squeeze(-1)
        cat_logits = self.cat_head(h, batch.cat_decode_types, batch.cat_cardinalities)
        return ModelOutput(num_mu=num_mu, cat_logits=cat_logits, h=h)


# ---------------------------------------------------------------------
# Loss wrappers, single-task and batched -- mirror
# model_perm_ar.py's compute_task_loss_onepass(_batched) exactly, same
# StepLossOutput type, so they drop into train_synthetic.py the same way.
# ---------------------------------------------------------------------


def compute_task_loss_single_stream(
    model: SingleStreamModel,
    full,
    task: CompletionTask,
    plan: FactorizationPlan,
    device: torch.device,
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
) -> StepLossOutput:
    from tab_completion.episode_utils import task_to_torch_batch
    from tab_completion.losses import typed_mse_ce_loss

    batch = task_to_torch_batch(full, task, device)

    rank_np = build_rank_tensor(task, plan)
    rank_t = torch.as_tensor(rank_np[None, :, :], dtype=torch.long, device=device)

    context_row_mask_np = get_context_row_mask_from_task(task)
    context_row_mask_t = None
    if context_row_mask_np is not None:
        context_row_mask_t = torch.as_tensor(
            context_row_mask_np[None, :], dtype=torch.bool, device=device
        )

    out = model(batch, rank_t, context_row_mask=context_row_mask_t)

    query_t = torch.as_tensor(task.query_mask[None, :, :], dtype=torch.bool, device=device)
    loss_out = typed_mse_ce_loss(out, batch, query_t, num_weight=num_weight, cat_weight=cat_weight)

    metrics = dict(loss_out.metrics)
    metrics["factorization_steps"] = float(plan.num_steps)
    metrics["query_cells"] = float(task.num_query_cells)

    return StepLossOutput(loss=loss_out.loss, metrics=metrics)


def compute_task_loss_single_stream_batched(
    model: SingleStreamModel,
    full_list: list,
    task_list: list[CompletionTask],
    plan_list: list[FactorizationPlan],
    device: torch.device,
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
) -> StepLossOutput:
    from tab_completion.episode_utils import tasks_to_torch_batch
    from tab_completion.losses import typed_mse_ce_loss

    batch = tasks_to_torch_batch(full_list, task_list, device)

    rank_np = build_rank_tensor_batched(task_list, plan_list)
    rank_t = torch.as_tensor(rank_np, dtype=torch.long, device=device)

    context_row_mask_np = get_context_row_mask_batched(task_list)
    context_row_mask_t = None
    if context_row_mask_np is not None:
        context_row_mask_t = torch.as_tensor(context_row_mask_np, dtype=torch.bool, device=device)

    out = model(batch, rank_t, context_row_mask=context_row_mask_t)

    query_np = np.stack([task.query_mask for task in task_list])
    query_t = torch.as_tensor(query_np, dtype=torch.bool, device=device)
    loss_out = typed_mse_ce_loss(out, batch, query_t, num_weight=num_weight, cat_weight=cat_weight)

    metrics = dict(loss_out.metrics)
    metrics["factorization_steps"] = float(np.mean([plan.num_steps for plan in plan_list]))
    metrics["query_cells"] = float(np.mean([task.num_query_cells for task in task_list]))

    return StepLossOutput(loss=loss_out.loss, metrics=metrics)
