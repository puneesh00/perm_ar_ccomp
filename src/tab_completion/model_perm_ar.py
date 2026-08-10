# src/tab_completion/model_perm_ar.py
"""
One-pass permutation-AR completion model.

The original CellwiseCompletionModel (model.py) trains perm-AR factorizations
by literally looping over plan.steps and calling model.forward() once per
step, with teacher forcing implemented by mutating observed_mask between
calls. That's O(num_steps) full forward+backward passes chained into one
autograd graph -- correct, but for cell-wise AR on a task like random_cell
(~600 query cells/episode) it means ~600 full-table forward passes per
training step, which is both extremely slow and memory-hungry (each pass's
activations stay alive for the backward pass).

Standard autoregressive language models don't pay this cost because the
token order is fixed, so a single causal attention mask handles teacher
forcing for the whole sequence in one pass. Perm-AR here uses a *random*
per-episode permutation instead of a fixed order, so the fixed-mask trick
doesn't directly apply -- but permutation language modeling (XLNet) solves
exactly this with two-stream self-attention:

  - a content stream that always carries a cell's true value, used only as
    attention keys/values (so later-revealed cells can read it once their
    turn comes)
  - a query stream that never sees its own value, used only to produce the
    prediction, restricted by masking to attend to strictly earlier-revealed
    content

This file adapts that idea to the row x column grid this codebase uses
(rather than a flat 1-D token sequence): attention alternates between a
"row-axis" pass (attend across the D columns within a fixed row) and a
"col-axis" pass (attend across the N rows within a fixed column), each
masked by a per-cell "reveal rank" derived from the existing
FactorizationPlan. This replaces the original's row_encoder (within-row
attention, kept) and row_context_encoder + column mean-pooling (cross-row,
here replaced by real attention so it can be masked per-query-position).

Nothing in sampling.py or factorization.py needs to change: any sampler
(TargetPredictionSampler, RandomCellSampler, ColumnBlockSampler,
RowBlockSampler, LabelFeatureSampler, MixtureSampler) and any factorizer
(ParallelFactorizer, PermARFactorizer with any unit) already produces a
(CompletionTask, FactorizationPlan) pair; build_rank_tensor() below turns
that pair into the per-cell rank this model consumes. Parallel factorization
is just the special case where every query cell shares rank 1 -- the model
doesn't need to know which factorization produced its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------------------------------------------------------
# Rank tensor: turns (task, plan) into a per-cell "when is this revealed"
# integer, which is all this model needs from either a sampler or a
# factorizer.
# ---------------------------------------------------------------------

RANK_OBSERVED = 0
RANK_NEVER = 1 << 30  # cells that are neither observed nor ever queried


def build_rank_tensor(task: CompletionTask, plan: FactorizationPlan) -> np.ndarray:
    """
    rank[i, j] =
      0            if (i, j) is in task.observed_mask (always visible)
      step_idx + 1 if (i, j) is queried in plan.steps[step_idx]
      RANK_NEVER   otherwise (neither observed nor queried this episode)

    For parallel factorization (one step containing every query cell) every
    query cell gets rank 1 -- content-stream cells can then see all rank-0
    (observed) cells plus themselves, and query-stream cells can only see
    rank-0 cells, exactly reproducing today's single-pass behavior. For
    cell-wise perm-AR, each query cell gets its own rank in permutation
    order, which is what lets later cells see earlier (teacher-forced) ones
    while earlier cells can't see later ones -- all computed in one pass
    instead of one pass per step.
    """
    shape = task.observed_mask.shape
    rank = np.full(shape, RANK_NEVER, dtype=np.int64)
    rank[task.observed_mask] = RANK_OBSERVED

    for step_idx, coords in enumerate(plan.steps):
        if len(coords) == 0:
            continue
        rank[coords[:, 0], coords[:, 1]] = step_idx + 1

    return rank


def get_context_row_mask_from_task(task: CompletionTask) -> Optional[np.ndarray]:
    """
    Same semantics as the helper of the same name in scripts/train_synthetic.py
    (duplicated here rather than imported, to avoid this library module
    depending on the training script).
    """
    mode = task.meta.get("conditioning_mode", "transductive")

    if mode == "transductive":
        return None

    if mode == "inductive_rows":
        context_rows = task.meta.get("context_rows_local", None)
        if context_rows is None:
            raise ValueError(
                "Task has conditioning_mode='inductive_rows' but no "
                "context_rows_local in task.meta."
            )
        context_rows = np.asarray(context_rows, dtype=np.int64)
        mask = np.zeros(task.observed_mask.shape[0], dtype=bool)
        mask[context_rows] = True
        return mask

    raise ValueError(f"Unknown conditioning_mode={mode!r}")


# ---------------------------------------------------------------------
# Tokenizer: builds the content stream (always has the true value) and
# the query stream (position-only, never has a value) per cell.
# ---------------------------------------------------------------------


class PermARTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # No row_emb: context/query rows are an exchangeable set (no
        # meaningful row order), and origin_emb below already carries the
        # one row-related distinction that does matter (context vs. query).
        # An absolute per-row-index embedding would just be nuisance
        # variation the model has to learn to ignore.
        #
        # No col_emb either (ablation, mirroring the row_emb removal):
        # nanoTabPFN's reference TabPFN-v1 reimplementation has no
        # column-identity embedding at all -- a cell's column identity is
        # purely implicit in which attention axis-group it's routed through
        # at each layer (the col-axis reshape groups same-column cells
        # together across the whole stack), never an added vector. Testing
        # whether the same "this was pure nuisance variation" story that
        # applied to row_emb also applies here.
        #
        # drop_type_origin_emb removes type_emb and origin_emb too: content
        # always carries the true value (no ambiguity origin_emb could
        # resolve, and XLNet's own content stream is just e(x_i), nothing
        # else), and origin_emb is provably constant wherever the query
        # stream's own output is ever used (every query_mask cell has
        # is_query_cell == True by construction) -- see ModelConfig.
        if not cfg.drop_type_origin_emb:
            self.type_emb = nn.Embedding(2, d)
            # 0 = originally observed (rank == RANK_OBSERVED), 1 = a query
            # cell this episode (finite rank > 0). Purely informational --
            # it does not carry the value, so it can't leak anything.
            self.origin_emb = nn.Embedding(2, d)

        # unified_cat_encoding drops this entirely -- categorical cells are
        # cast to float and routed through num_value_mlp instead (see
        # forward()), matching model_single_stream.py's SingleStreamTokenizer.
        if not cfg.unified_cat_encoding:
            self.cat_value_emb = nn.Parameter(
                torch.randn(cfg.num_cat_decode_types, cfg.k_max, d) * 0.02
            )
        self.num_value_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.content_norm = nn.LayerNorm(d)
        self.query_norm = nn.LayerNorm(d)

    def forward(
        self,
        batch: TableTensorBatch,
        rank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        rank: [B, N, D] long.

        Returns (content, query), each [B, N, D, d_model].
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
            # TabPFNV1Model's actual recipe (see model_single_stream.py's
            # SingleStreamTokenizer for the full rationale): every column
            # becomes one continuous-valued channel -- numerics keep their
            # real value, categoricals are their raw (post-shuffle,
            # non-ordinal) id cast to float -- both through the SAME
            # num_value_mlp. No cat_value_emb, no per-column split.
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

            # --- content stream: always the true value, for every cell ---
            content_value = self.num_value_mlp(x_unified_input.unsqueeze(-1))
            # --- query stream: informative placeholder (context mean, or 0
            # in normalized space, fed through the same value MLP). ---
            query_value = self.num_value_mlp(
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
                # freshly computed each episode.
                num_sq_sum = (x_num.pow(2) * observed_num.to(x_num.dtype)).sum(dim=1)
                col_var_num = (num_sq_sum / num_count - col_mean_num.pow(2)).clamp(min=0.0)
                col_std_num = torch.sqrt(col_var_num + 1e-6)  # [B, D]
                x_num_input = (x_num - col_mean_num.unsqueeze(1)) / col_std_num.unsqueeze(1)
                x_num_input = x_num_input.clamp(min=-100.0, max=100.0)
                placeholder_mean_num = torch.zeros_like(col_mean_num)
            else:
                x_num_input = x_num
                placeholder_mean_num = col_mean_num

            # --- content stream: always the true value, for every cell ---
            num_vec_all = self.num_value_mlp(x_num_input.unsqueeze(-1))
            cat_vec_all = self.cat_value_emb[cat_type_clamped, x_cat_clamped]
            content_value = torch.where(is_num.unsqueeze(-1), num_vec_all, cat_vec_all)

            # --- query stream: never its own value, but an *informative*
            # placeholder rather than a fixed learned constant -- numerical
            # gets the observed-cell mean fed through the same value MLP a
            # real value would use; categorical gets the
            # observed-frequency-weighted average of that column's category
            # embeddings (not a mean of raw ids -- those are deliberately
            # non-ordinal, per bin_by_realized_values' shuffle). ---
            placeholder_num_vec = self.num_value_mlp(
                placeholder_mean_num.unsqueeze(1).expand(B, N, D).unsqueeze(-1)
            )

            k_max = self.cfg.k_max
            x_cat_onehot = F.one_hot(x_cat_clamped, num_classes=k_max).to(dtype=self.cat_value_emb.dtype)
            x_cat_onehot = x_cat_onehot * observed_cat.unsqueeze(-1).to(x_cat_onehot.dtype)
            cat_counts = x_cat_onehot.sum(dim=1)  # [B, D, K]
            cat_totals = cat_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
            cat_probs = cat_counts / cat_totals  # [B, D, K]

            cat_type_bd = cat_type_clamped[:, 0, :]  # [B, D] -- decode type is constant across rows
            emb_table_bd = self.cat_value_emb[cat_type_bd]  # [B, D, K, d]
            placeholder_cat_vec_bd = torch.einsum("bdk,bdkh->bdh", cat_probs, emb_table_bd)
            placeholder_cat_vec = placeholder_cat_vec_bd.unsqueeze(1).expand(B, N, D, placeholder_cat_vec_bd.shape[-1])

            query_value = torch.where(is_num.unsqueeze(-1), placeholder_num_vec, placeholder_cat_vec)

        if self.cfg.drop_type_origin_emb:
            content = self.content_norm(content_value)
            query = self.query_norm(query_value)
            return content, query

        pos = self.type_emb(type_ids)

        is_query_cell = (rank != RANK_OBSERVED) & (rank != RANK_NEVER)
        origin = self.origin_emb(is_query_cell.long())

        content = self.content_norm(content_value + pos + origin)
        query = self.query_norm(query_value + pos + origin)

        return content, query


# ---------------------------------------------------------------------
# Masked axial multi-head attention.
# ---------------------------------------------------------------------


def _rank_visibility(rank_reader: torch.Tensor, rank_key: torch.Tensor, allow_self: bool) -> torch.Tensor:
    """
    rank_reader: [..., L, 1], rank_key: [..., 1, S] (broadcastable).
    Returns bool [..., L, S]: True where the key is allowed evidence for
    that reader under teacher forcing (key revealed strictly before the
    reader's own rank, or the reader reading its own content).

    allow_self=True  -> content-stream self-attention (a cell always sees
                         its own content; that's what lets later cells read
                         it once revealed).
    allow_self=False -> query-stream cross-attention (a cell must never see
                         its own value, revealed or not).
    """
    earlier = rank_key < rank_reader
    if allow_self:
        # NOTE: this is a value tie (rank_key == rank_reader), not a position
        # check -- it's True for every pair of *different* cells that share
        # a rank, not just the diagonal. That's intentional and needed: all
        # rank-0 (originally observed) cells must mutually see each other
        # (no permutation applies to given evidence), and cells within the
        # same grouped-AR step (unit="column"/"row") must too, since by the
        # time anything with a strictly later rank reads them, the whole
        # step's group is teacher-forced-revealed together anyway. It does
        # NOT let same-rank cells influence each other's own *prediction*
        # (query_mask below only ever allows strictly-earlier ranks), and
        # cross-row ties are further restricted by row_gate in _axis_masks.
        is_self = rank_key == rank_reader
        return earlier | is_self
    return earlier


class MaskedMultiheadAttention(nn.Module):
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

    def forward(
        self,
        query_in: torch.Tensor,
        kv_in: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_in: [G, L, d]   (G = axial "batch" -- e.g. B*N groups for row-axis)
        kv_in:    [G, S, d]
        attn_mask: bool [G, L, S], True = allowed.

        Every row of attn_mask must have at least one True entry (a reader
        with zero visible keys produces NaNs from softmax); callers ensure
        this by always allowing content self-attention, and by the caller
        skipping query-stream reads for cells with no valid evidence.
        """
        G, L, d = query_in.shape
        S = kv_in.shape[1]
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(query_in).view(G, L, H, hd).transpose(1, 2)  # [G,H,L,hd]
        k = self.k_proj(kv_in).view(G, S, H, hd).transpose(1, 2)     # [G,H,S,hd]
        v = self.v_proj(kv_in).view(G, S, H, hd).transpose(1, 2)     # [G,H,S,hd]

        mask = attn_mask.unsqueeze(1)  # [G,1,L,S], broadcasts across heads

        # A reader can legitimately have zero allowed keys along a single
        # axis pass -- e.g. a query-stream cell in a fully-masked row
        # (RowBlockSampler defaults to query_frac_cols=1.0) has no same-row
        # evidence yet on the row-axis pass. Softmax over an all-False mask
        # is NaN, so we widen those rows to a dummy all-True mask purely to
        # keep the op finite, then explicitly zero their *output* below --
        # widening the mask alone (without the zeroing) would let a
        # no-evidence query position attend to everything, including its
        # own true value and not-yet-revealed cells. That must never reach
        # the residual stream.
        has_any = mask.any(dim=-1, keepdim=True)
        safe_mask = mask | (~has_any)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=safe_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out * has_any.to(out.dtype)
        out = out.transpose(1, 2).reshape(G, L, d)
        return self.out_proj(out)


def _reshape_for_axis(x: torch.Tensor, axis: str) -> torch.Tensor:
    # x: [B, N, D, d] -> [G, L, d]
    B, N, D, d = x.shape
    if axis == "row":
        return x.reshape(B * N, D, d)
    return x.permute(0, 2, 1, 3).reshape(B * D, N, d)


def _reshape_back(x: torch.Tensor, axis: str, B: int, N: int, D: int) -> torch.Tensor:
    d = x.shape[-1]
    if axis == "row":
        return x.reshape(B, N, D, d)
    return x.reshape(B, D, N, d).permute(0, 2, 1, 3)


def _compute_axis_masks(
    axis: str, rank: torch.Tensor, row_gate: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    rank: [B, N, D]. Returns (content_mask, query_mask), each bool shaped
    [G, L, L] for this axis (G, L as in _reshape_for_axis).

    row_gate: optional [B, N] bool, True = context row. When given
    (inductive_rows conditioning): any row may use context rows as
    evidence; a row may always use itself; no row may use a *different*
    query row -- directly, or indirectly through a context row that itself
    read that other query row (that indirect path is exactly the leak this
    gate exists to close, since rank ties at rank 0 would otherwise let a
    query row's own observed cells reach a context row's representation and
    propagate from there). Only relevant to the col axis (that's the
    cross-row direction); ignored for row axis.
    """
    B, N, D = rank.shape

    if axis == "row":
        r = rank.reshape(B * N, D)
        reader = r.unsqueeze(-1)  # [G, D, 1]
        key = r.unsqueeze(-2)  # [G, 1, D]
        content_mask = _rank_visibility(reader, key, allow_self=True)
        query_mask = _rank_visibility(reader, key, allow_self=False)
        return content_mask, query_mask

    # col axis: G = B*D groups, L = S = N
    r = rank.permute(0, 2, 1).reshape(B * D, N)  # [G, N]
    reader = r.unsqueeze(-1)  # [G, N, 1]
    key = r.unsqueeze(-2)  # [G, 1, N]
    content_mask = _rank_visibility(reader, key, allow_self=True)
    query_mask = _rank_visibility(reader, key, allow_self=False)

    if row_gate is not None:
        # row_gate: [B, N] -> broadcast to the same [G, N] groups (D copies per batch item)
        g = row_gate.unsqueeze(1).expand(B, D, N).reshape(B * D, N)
        key_is_context = g.unsqueeze(-2)  # [G, 1, N]
        same_row = torch.eye(N, device=rank.device, dtype=torch.bool).unsqueeze(0)
        # Allowed row-wise evidence, for ANY reader (context or query): a
        # context-row key, or the reader's own row. This blocks context from
        # reading query rows too (not just query-from-query), which is what
        # closes the indirect leak -- see docstring above.
        row_allowed = key_is_context | same_row
        content_mask = content_mask & row_allowed
        query_mask = query_mask & row_allowed

    return content_mask, query_mask


class AxialTwoStreamLayer(nn.Module):
    """
    One layer of masked two-stream attention along a single axis (row or
    column), pre-norm, with a feedforward block on each stream. Content
    does masked self-attention (sees itself + earlier-revealed cells along
    this axis); query does masked cross-attention into content (sees only
    strictly-earlier-revealed cells).
    """

    def __init__(self, cfg: ModelConfig, axis: str):
        super().__init__()
        if axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
        self.axis = axis
        self.post_ln = cfg.post_ln
        d = cfg.d_model

        self.content_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        # share_stream_attn (XLNet-style): the query stream reuses
        # content_attn's weights instead of having its own -- attention is
        # still invoked twice (same activations), but the two streams no
        # longer double the attention parameter count. See ModelConfig.
        self.query_attn = (
            self.content_attn
            if cfg.share_stream_attn
            else MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        )

        self.content_norm1 = nn.LayerNorm(d)
        self.content_norm2 = nn.LayerNorm(d)
        self.query_norm1 = nn.LayerNorm(d)
        self.query_norm2 = nn.LayerNorm(d)

        def ffn():
            return nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
            )

        self.content_ffn = ffn()
        self.query_ffn = self.content_ffn if cfg.share_stream_attn else ffn()
        self.dropout = nn.Dropout(cfg.dropout)

    def _reshape_for_axis(self, x: torch.Tensor) -> torch.Tensor:
        return _reshape_for_axis(x, self.axis)

    def _reshape_back(self, x: torch.Tensor, B: int, N: int, D: int) -> torch.Tensor:
        return _reshape_back(x, self.axis, B, N, D)

    def _axis_masks(self, rank: torch.Tensor, row_gate: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return _compute_axis_masks(self.axis, rank, row_gate)

    def forward(
        self,
        content: torch.Tensor,
        query: torch.Tensor,
        rank: torch.Tensor,
        row_gate: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D, d = content.shape
        content_mask, query_mask = self._axis_masks(rank, row_gate)

        c_in = self._reshape_for_axis(content)
        q_in = self._reshape_for_axis(query)

        # XLNet's equations have BOTH streams read the previous layer's
        # content (h^(m-1)), not this layer's just-updated content -- so
        # query's KV below is always the pre-update content snapshot
        # content_attn itself used as its own K/V (c_kv), never content_out.
        # Reading content_out would let the query stream see one layer
        # "ahead" of what XLNet specifies.
        if self.post_ln:
            # TabPFNV1Layer's layout: attn/ffn operate on the raw
            # (previous-stage) input, residual add first, THEN LayerNorm.
            c_kv = c_in
            c_attn = self.content_attn(c_in, c_kv, content_mask)
            c_in = self.content_norm1(c_in + self.dropout(c_attn))
            c_in = self.content_norm2(c_in + self.content_ffn(c_in))
            content_out = self._reshape_back(c_in, B, N, D)

            q_attn = self.query_attn(q_in, c_kv, query_mask)
            q_in = self.query_norm1(q_in + self.dropout(q_attn))
            q_in = self.query_norm2(q_in + self.query_ffn(q_in))
        else:
            c_normed = self.content_norm1(c_in)
            c_kv = c_normed
            c_attn = self.content_attn(c_normed, c_kv, content_mask)
            c_in = c_in + self.dropout(c_attn)
            c_in = c_in + self.content_ffn(self.content_norm2(c_in))
            content_out = self._reshape_back(c_in, B, N, D)

            q_normed = self.query_norm1(q_in)
            q_attn = self.query_attn(q_normed, c_kv, query_mask)
            q_in = q_in + self.dropout(q_attn)
            q_in = q_in + self.query_ffn(self.query_norm2(q_in))
        query_out = self._reshape_back(q_in, B, N, D)

        return content_out, query_out


class PairedAxialTwoStreamLayer(nn.Module):
    """
    TabPFNV1Layer's layout (model_tabpfn_v1.py) applied to the two-stream
    model: row-axis attention (content+query), then col-axis attention
    (content+query), THEN one FFN application per stream -- instead of
    AxialTwoStreamLayer's default of a separate FFN after each single-axis
    attention. Same total attention ops, half the FFN applications, so
    num_row_layers can be doubled within the same FFN parameter/compute
    budget. Requires num_row_layers == num_row_context_layers (one paired
    row+col block per count) -- see PermARCompletionModel.

    Row and col attention are kept as separate weights (mirroring
    TabPFNV1Layer's separate feature_attn/datapoint_attn -- axes are not
    shared with each other); share_stream_attn independently controls
    whether content and query share weights *within* each axis, same as
    AxialTwoStreamLayer. The content_out-staleness fix applies per axis
    sub-step: each axis's query reads the pre-that-step content snapshot,
    not the version that axis's own content-attention just produced.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.post_ln = cfg.post_ln
        d = cfg.d_model

        self.row_content_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.row_query_attn = (
            self.row_content_attn
            if cfg.share_stream_attn
            else MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        )
        self.col_content_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.col_query_attn = (
            self.col_content_attn
            if cfg.share_stream_attn
            else MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        )

        self.row_content_norm = nn.LayerNorm(d)
        self.row_query_norm = nn.LayerNorm(d)
        self.col_content_norm = nn.LayerNorm(d)
        self.col_query_norm = nn.LayerNorm(d)
        self.content_ffn_norm = nn.LayerNorm(d)
        self.query_ffn_norm = nn.LayerNorm(d)

        def ffn():
            return nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
            )

        self.content_ffn = ffn()
        self.query_ffn = self.content_ffn if cfg.share_stream_attn else ffn()
        self.dropout = nn.Dropout(cfg.dropout)

    def _axis_step(
        self,
        content: torch.Tensor,
        query: torch.Tensor,
        rank: torch.Tensor,
        row_gate: Optional[torch.Tensor],
        axis: str,
        content_attn: nn.Module,
        query_attn: nn.Module,
        content_norm: nn.Module,
        query_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D, d = content.shape
        content_mask, query_mask = _compute_axis_masks(axis, rank, row_gate)

        c_in = _reshape_for_axis(content, axis)
        q_in = _reshape_for_axis(query, axis)

        if self.post_ln:
            # TabPFNV1Layer's layout: attn operates on the raw input,
            # residual add first, THEN LayerNorm -- see ModelConfig.post_ln.
            # Both streams read the same pre-update content (c_in, raw),
            # matching XLNet's "both streams read h^(m-1)" requirement.
            c_attn = content_attn(c_in, c_in, content_mask)
            c_out = content_norm(c_in + self.dropout(c_attn))
            content_out = _reshape_back(c_out, axis, B, N, D)

            q_attn = query_attn(q_in, c_in, query_mask)  # pre-this-step content, see class docstring
            q_out = query_norm(q_in + self.dropout(q_attn))
            query_out = _reshape_back(q_out, axis, B, N, D)
        else:
            c_normed = content_norm(c_in)
            c_attn = content_attn(c_normed, c_normed, content_mask)
            c_in = c_in + self.dropout(c_attn)
            content_out = _reshape_back(c_in, axis, B, N, D)

            q_normed = query_norm(q_in)
            q_attn = query_attn(q_normed, c_normed, query_mask)  # pre-this-step content, see class docstring
            q_in = q_in + self.dropout(q_attn)
            query_out = _reshape_back(q_in, axis, B, N, D)

        return content_out, query_out

    def forward(
        self,
        content: torch.Tensor,
        query: torch.Tensor,
        rank: torch.Tensor,
        row_gate: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content, query = self._axis_step(
            content, query, rank, row_gate, "row",
            self.row_content_attn, self.row_query_attn,
            self.row_content_norm, self.row_query_norm,
        )
        content, query = self._axis_step(
            content, query, rank, row_gate, "col",
            self.col_content_attn, self.col_query_attn,
            self.col_content_norm, self.col_query_norm,
        )
        if self.post_ln:
            content = self.content_ffn_norm(content + self.content_ffn(content))
            query = self.query_ffn_norm(query + self.query_ffn(query))
        else:
            content = content + self.content_ffn(self.content_ffn_norm(content))
            query = query + self.query_ffn(self.query_ffn_norm(query))
        return content, query


# ---------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------


class PermARCompletionModel(nn.Module):
    """
    Drop-in alternative to CellwiseCompletionModel for perm-AR training:
    scores *every* query cell of an episode (any rank, any factorization
    unit) in a single forward pass instead of one pass per AR step.

    forward() signature intentionally mirrors the original: give it a
    TableTensorBatch and a per-cell rank (see build_rank_tensor), get back
    the same ModelOutput(num_mu, cat_logits, h) that typed_mse_ce_loss
    already knows how to score -- so the loss function and eval plumbing
    in scripts/train_synthetic.py don't need to change, only how the model
    is called (see compute_task_loss_onepass below).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = PermARTokenizer(cfg)

        if cfg.tabpfn_style_layers:
            if cfg.num_row_layers != cfg.num_row_context_layers:
                raise ValueError(
                    "tabpfn_style_layers requires num_row_layers == "
                    f"num_row_context_layers (got {cfg.num_row_layers} vs "
                    f"{cfg.num_row_context_layers}) -- one paired row+col "
                    "block per count."
                )
            self.layers = nn.ModuleList(
                [PairedAxialTwoStreamLayer(cfg) for _ in range(cfg.num_row_layers)]
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

            self.layers = nn.ModuleList([AxialTwoStreamLayer(cfg, axis) for axis in axes])

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.num_head = nn.Linear(cfg.d_model, 1)
        self.cat_head = TypedCategoricalHead(cfg)

    def forward(
        self,
        batch: TableTensorBatch,
        rank: torch.Tensor,
        context_row_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        rank: [B, N, D] long (see build_rank_tensor).
        context_row_mask: [B, N] bool, True = context row. None = transductive
            (matches CellwiseCompletionModel's convention).
        """
        content, query = self.tokenizer(batch, rank)

        for layer in self.layers:
            content, query = layer(content, query, rank, context_row_mask)

        h = self.final_norm(query)
        num_mu = self.num_head(h).squeeze(-1)
        cat_logits = self.cat_head(h, batch.cat_decode_types, batch.cat_cardinalities)

        return ModelOutput(num_mu=num_mu, cat_logits=cat_logits, h=h)


# ---------------------------------------------------------------------
# One-pass replacement for scripts/train_synthetic.py's compute_task_loss.
# Same signature/return type (StepLossOutput has .loss/.metrics), so it can
# be swapped in directly; not wired into train_synthetic.py here since that
# file is left untouched -- see the module docstring.
# ---------------------------------------------------------------------


@dataclass
class StepLossOutput:
    loss: torch.Tensor
    metrics: dict


def compute_task_loss_onepass(
    model: PermARCompletionModel,
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


# ---------------------------------------------------------------------
# Real-batch replacement for compute_task_loss_onepass: instead of a Python
# loop calling the model once per task at B=1 (train_synthetic.py's
# `for _ in range(args.batch_tasks): loss_fn(...)` then averaging), this
# stacks B independently-sampled (full, task, plan) triples into one real
# [B, N, D] batch and calls the model ONCE. Same gradient math either way
# (both average B independently-sampled episode losses before backward) --
# this only changes how that compute is scheduled on the GPU, which matters
# a lot: B sequential B=1 forward passes badly underuse a GPU's parallelism
# compared to one batched call.
#
# Requires every task in the batch to share the same (N, D) shape, which
# holds whenever n_context/n_query/n_cols are fixed by config -- true for
# every TargetPredictionSampler + ParallelFactorizer run this codebase has
# been using. Not a general replacement for the per-task path: samplers with
# a per-episode-variable number of query cells/cols (RandomCellSampler,
# ColumnBlockSampler) or perm-AR plans with a per-episode-variable step
# count don't fit a single dense batch this way without padding machinery
# this doesn't implement -- see build_rank_tensor_batched's ValueError.
# ---------------------------------------------------------------------


def build_rank_tensor_batched(task_list: list[CompletionTask], plan_list: list[FactorizationPlan]) -> np.ndarray:
    """Stacks build_rank_tensor(task, plan) over a batch into [B, N, D]."""
    ranks = [build_rank_tensor(task, plan) for task, plan in zip(task_list, plan_list)]
    shapes = {r.shape for r in ranks}
    if len(shapes) > 1:
        raise ValueError(
            f"build_rank_tensor_batched requires every task to share the same "
            f"(N, D) shape; got shapes {shapes}."
        )
    return np.stack(ranks)


def get_context_row_mask_batched(task_list: list[CompletionTask]) -> Optional[np.ndarray]:
    """Stacks get_context_row_mask_from_task(task) over a batch into [B, N]."""
    masks = [get_context_row_mask_from_task(task) for task in task_list]
    if all(m is None for m in masks):
        return None
    if any(m is None for m in masks):
        raise ValueError(
            "get_context_row_mask_batched requires every task in the batch to "
            "share the same conditioning_mode (all transductive or all "
            "inductive_rows), not a mix."
        )
    return np.stack(masks)


def compute_task_loss_onepass_batched(
    model: PermARCompletionModel,
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
