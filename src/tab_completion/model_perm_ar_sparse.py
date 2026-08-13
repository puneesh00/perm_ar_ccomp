# src/tab_completion/model_perm_ar_sparse.py
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
# Sparse XLNet-style query stream.
#
# The content stream is dense because every cell's true value may become
# teacher-forced context for a later target. The query stream exists only for
# cells whose losses are actually evaluated. This is the tabular analogue of
# XLNet's target_mapping / partial-prediction path: unselected positions do not
# need query representations at all.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SparseAxisLayout:
    """Packing layout for sparse query states along one axial attention pass."""

    axis: str
    active_group_ids: torch.Tensor  # [G_active], indexes B*N row groups or B*D col groups
    slot_to_query: torch.Tensor  # [G_active, Lq_max], -1 denotes padding
    valid: torch.Tensor  # [G_active, Lq_max]
    query_ranks: torch.Tensor  # [G_active, Lq_max]
    query_rows: torch.Tensor  # [G_active, Lq_max], needed for inductive col gating

    def pack(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pack [Q,d] query states into [G_active,Lq_max,d]."""
        if hidden.ndim != 2:
            raise ValueError(f"Expected sparse query hidden [Q,d], got {tuple(hidden.shape)}")
        safe = self.slot_to_query.clamp_min(0)
        packed = hidden.index_select(0, safe.reshape(-1)).reshape(
            self.slot_to_query.shape[0], self.slot_to_query.shape[1], hidden.shape[-1]
        )
        return packed * self.valid.unsqueeze(-1).to(packed.dtype)

    def unpack(self, packed: torch.Tensor, num_queries: int) -> torch.Tensor:
        """Undo pack(), returning query states in their original [Q,d] order."""
        valid_flat = self.valid.reshape(-1)
        query_index = self.slot_to_query.reshape(-1)[valid_flat]
        values = packed.reshape(-1, packed.shape[-1])[valid_flat]
        out = packed.new_empty((num_queries, packed.shape[-1]))
        return out.index_copy(0, query_index, values)


@dataclass(frozen=True)
class SparseQueryState:
    """Only the target/query cells carry a query-stream hidden state."""

    hidden: torch.Tensor  # [Q,d]
    coords: torch.Tensor  # [Q,3] columns are (batch,row,col)
    ranks: torch.Tensor  # [Q]
    batch_size: int
    n_rows: int
    n_cols: int

    @property
    def num_queries(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def batch_index(self) -> torch.Tensor:
        return self.coords[:, 0]

    @property
    def row_index(self) -> torch.Tensor:
        return self.coords[:, 1]

    @property
    def col_index(self) -> torch.Tensor:
        return self.coords[:, 2]

    def with_hidden(self, hidden: torch.Tensor) -> "SparseQueryState":
        if hidden.shape != self.hidden.shape:
            raise ValueError(
                f"Sparse query shape changed from {tuple(self.hidden.shape)} to {tuple(hidden.shape)}"
            )
        return SparseQueryState(
            hidden=hidden,
            coords=self.coords,
            ranks=self.ranks,
            batch_size=self.batch_size,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
        )


@dataclass(frozen=True)
class AxisAttentionCache:
    """Masks/layout that are constant across every block in one forward pass."""

    content_mask: torch.Tensor  # [G_total,L,L]
    query_layout: SparseAxisLayout
    query_mask: torch.Tensor  # [G_active,Lq_max,L]


@dataclass(frozen=True)
class GlobalBridgeCache:
    """Layout/mask for the sparse query -> flattened-content global bridge pass.

    No content_mask: this pass never updates content, only query, so there's
    no content self-attention to mask.
    """

    query_layout: SparseAxisLayout
    query_mask: torch.Tensor  # [B_active,Lq_max,N*D]


@dataclass(frozen=True)
class TwoStreamAttentionCache:
    row: AxisAttentionCache
    col: AxisAttentionCache
    global_bridge: Optional[GlobalBridgeCache] = None

    def for_axis(self, axis: str) -> AxisAttentionCache:
        if axis == "row":
            return self.row
        if axis == "col":
            return self.col
        raise ValueError(f"Unknown axis {axis!r}")


def _build_sparse_axis_layout(query: SparseQueryState, axis: str) -> SparseAxisLayout:
    """
    Group sparse query cells by their physical row or column.

    Within each active group we pad only to that minibatch's maximum number of
    queried cells in a group. The query stream is therefore O(Qd), not O(BNDd).
    """
    if axis == "row":
        group_ids = query.batch_index * query.n_rows + query.row_index
    elif axis == "col":
        group_ids = query.batch_index * query.n_cols + query.col_index
    elif axis == "global":
        # One group per batch item: every query in an episode attends the
        # same flattened content grid for that episode.
        group_ids = query.batch_index
    else:
        raise ValueError(f"Unknown axis {axis!r}")

    order = torch.argsort(group_ids)
    sorted_groups = group_ids.index_select(0, order)
    active_group_ids, counts = torch.unique_consecutive(sorted_groups, return_counts=True)

    # Q > 0 is validated in the tokenizer.
    max_count = int(counts.max().item())
    n_active = int(active_group_ids.numel())

    group_slot = torch.repeat_interleave(
        torch.arange(n_active, device=group_ids.device, dtype=torch.long), counts
    )
    starts = torch.cumsum(counts, dim=0) - counts
    position_in_group = torch.arange(
        query.num_queries, device=group_ids.device, dtype=torch.long
    ) - torch.repeat_interleave(starts, counts)

    slot_to_query = torch.full(
        (n_active, max_count), -1, dtype=torch.long, device=group_ids.device
    )
    slot_to_query[group_slot, position_in_group] = order
    valid = slot_to_query >= 0
    safe = slot_to_query.clamp_min(0)

    query_ranks = query.ranks.index_select(0, safe.reshape(-1)).reshape(n_active, max_count)
    query_rows = query.row_index.index_select(0, safe.reshape(-1)).reshape(n_active, max_count)
    query_ranks = torch.where(valid, query_ranks, torch.zeros_like(query_ranks))
    query_rows = torch.where(valid, query_rows, torch.zeros_like(query_rows))

    return SparseAxisLayout(
        axis=axis,
        active_group_ids=active_group_ids,
        slot_to_query=slot_to_query,
        valid=valid,
        query_ranks=query_ranks,
        query_rows=query_rows,
    )


def _rank_visibility(
    rank_reader: torch.Tensor, rank_key: torch.Tensor, allow_same_rank: bool
) -> torch.Tensor:
    """Return True where a content key is visible under the reveal-rank order."""
    visible = rank_key < rank_reader
    if allow_same_rank:
        # Rank 0 is the jointly observed context. Grouped AR steps are also
        # jointly teacher-forced after the step, so their content states may
        # mutually interact. Query readers remain strictly earlier-only.
        visible = visible | (rank_key == rank_reader)
    return visible


def _axis_rank(rank: torch.Tensor, axis: str) -> torch.Tensor:
    B, N, D = rank.shape
    if axis == "row":
        return rank.reshape(B * N, D)
    if axis == "col":
        # .contiguous(): permute+reshape collapses to a non-contiguous view
        # whenever B==1 (dropping a size-1 leading dim never needs a copy),
        # which otherwise propagates a transposed stride pattern into every
        # mask built from this rank -- and a non-contiguous (non-stride-1
        # last dim) attn_mask silently disqualifies SDPA's fused kernels,
        # forcing the ~4x-more-expensive math fallback. Confirmed via
        # profiler: this was the content-stream col-axis self-attention
        # (the single most expensive op in the model) falling back to math.
        return rank.permute(0, 2, 1).reshape(B * D, N).contiguous()
    if axis == "global":
        return rank.reshape(B, N * D)
    raise ValueError(f"Unknown axis {axis!r}")


def _compute_content_axis_mask(
    axis: str,
    rank: torch.Tensor,
    row_gate: Optional[torch.Tensor],
) -> torch.Tensor:
    """Dense content-stream mask. Content always has at least its own key."""
    B, N, D = rank.shape
    r = _axis_rank(rank, axis)
    mask = _rank_visibility(r.unsqueeze(-1), r.unsqueeze(-2), allow_same_rank=True)

    if axis == "col" and row_gate is not None:
        g = row_gate.unsqueeze(1).expand(B, D, N).reshape(B * D, N)
        key_is_context = g.unsqueeze(-2)
        same_row = torch.eye(N, device=rank.device, dtype=torch.bool).unsqueeze(0)
        mask = mask & (key_is_context | same_row)

    return mask


def _compute_sparse_query_axis_mask(
    query: SparseQueryState,
    layout: SparseAxisLayout,
    rank: torch.Tensor,
    row_gate: Optional[torch.Tensor],
) -> torch.Tensor:
    """Strictly-earlier content visibility for only the selected query cells."""
    key_ranks = _axis_rank(rank, layout.axis).index_select(0, layout.active_group_ids)
    mask = key_ranks.unsqueeze(1) < layout.query_ranks.unsqueeze(-1)
    mask = mask & layout.valid.unsqueeze(-1)

    if layout.axis == "col" and row_gate is not None:
        # A sparse query at row i may read a context row or its own row, but
        # never a different query row. Strict rank masking still prevents it
        # from reading its own target cell.
        batch_ids = torch.div(
            layout.active_group_ids, query.n_cols, rounding_mode="floor"
        )
        key_is_context = row_gate.index_select(0, batch_ids).unsqueeze(1)
        key_rows = torch.arange(query.n_rows, device=rank.device, dtype=torch.long)
        same_row = layout.query_rows.unsqueeze(-1) == key_rows.view(1, 1, -1)
        mask = mask & (key_is_context | same_row)

    if layout.axis == "global" and row_gate is not None:
        # Same inductive-row rule as col-axis, applied over the flattened
        # N*D content grid: a query at row i may read a context row or its
        # own row, never a different query row -- without this, the global
        # bridge would let one held-out query row leak into another's
        # prediction, defeating context_row_mask's whole purpose.
        n_cols = query.n_cols
        key_row_of_pos = torch.arange(
            query.n_rows * n_cols, device=rank.device, dtype=torch.long
        ) // n_cols  # [N*D]
        key_is_context = row_gate.index_select(0, layout.active_group_ids)  # [B_active,N]
        key_is_context_flat = key_is_context.index_select(1, key_row_of_pos).unsqueeze(1)  # [B_active,1,N*D]
        same_row = layout.query_rows.unsqueeze(-1) == key_row_of_pos.view(1, 1, -1)
        mask = mask & (key_is_context_flat | same_row)

    return mask


def _build_global_bridge_cache(
    rank: torch.Tensor,
    query: SparseQueryState,
    row_gate: Optional[torch.Tensor],
) -> GlobalBridgeCache:
    layout = _build_sparse_axis_layout(query, "global")
    mask = _compute_sparse_query_axis_mask(query, layout, rank, row_gate)
    return GlobalBridgeCache(query_layout=layout, query_mask=mask)


def _global_query_bridge_step(
    *,
    content: torch.Tensor,
    query: SparseQueryState,
    cache: GlobalBridgeCache,
    bridge_attn: "MaskedMultiheadAttention",
    content_norm: nn.Module,
    query_norm: nn.Module,
    dropout: nn.Module,
    post_ln: bool,
) -> SparseQueryState:
    """
    One sparse query -> flattened-content global attention pass, giving each
    query direct access to every earlier-ranked cell in its episode
    regardless of row/column alignment. Content is only ever read here, never
    updated -- see ModelConfig.global_query_bridge for why this suffices.
    """
    B, N, D, d = content.shape
    content_flat = content.reshape(B, N * D, d)
    q_in = cache.query_layout.pack(query.hidden)

    c_kv = content_flat.index_select(0, cache.query_layout.active_group_ids)
    if not post_ln:
        c_kv = content_norm(c_kv)
    q_base = q_in if post_ln else query_norm(q_in)

    out = bridge_attn(q_base, c_kv, cache.query_mask, allow_empty_rows=True)

    if post_ln:
        q_out_packed = query_norm(q_in + dropout(out))
    else:
        q_out_packed = q_in + dropout(out)

    query_hidden = cache.query_layout.unpack(q_out_packed, query.num_queries)
    return query.with_hidden(query_hidden)


def _build_attention_cache(
    rank: torch.Tensor,
    query: SparseQueryState,
    row_gate: Optional[torch.Tensor],
    build_global_bridge: bool = False,
) -> TwoStreamAttentionCache:
    row_layout = _build_sparse_axis_layout(query, "row")
    col_layout = _build_sparse_axis_layout(query, "col")

    row_cache = AxisAttentionCache(
        content_mask=_compute_content_axis_mask("row", rank, row_gate),
        query_layout=row_layout,
        query_mask=_compute_sparse_query_axis_mask(query, row_layout, rank, row_gate),
    )
    col_cache = AxisAttentionCache(
        content_mask=_compute_content_axis_mask("col", rank, row_gate),
        query_layout=col_layout,
        query_mask=_compute_sparse_query_axis_mask(query, col_layout, rank, row_gate),
    )
    global_cache = (
        _build_global_bridge_cache(rank, query, row_gate) if build_global_bridge else None
    )
    return TwoStreamAttentionCache(row=row_cache, col=col_cache, global_bridge=global_cache)


# ---------------------------------------------------------------------
# Tokenizer: dense true-value content, sparse trainable query token.
# ---------------------------------------------------------------------


class PermARTokenizer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.share_stream_parameters = bool(cfg.share_stream_attn)

        if not cfg.drop_type_origin_emb:
            self.type_emb = nn.Embedding(2, d)
            self.origin_emb = nn.Embedding(2, d)

        if not cfg.unified_cat_encoding:
            self.cat_value_emb = nn.Parameter(
                torch.randn(cfg.num_cat_decode_types, cfg.k_max, d) * 0.02
            )

        self.num_value_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # XLNet's g^(0)=w. Only selected prediction cells receive a copy.
        self.query_token = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

        self.content_norm = nn.LayerNorm(d)
        self.query_norm = None if self.share_stream_parameters else nn.LayerNorm(d)

    def _normalize_query(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.content_norm if self.query_norm is None else self.query_norm
        return norm(x)

    def forward(
        self,
        batch: TableTensorBatch,
        rank: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, SparseQueryState]:
        """
        Args:
            rank: [B,N,D] reveal ranks.
            prediction_mask: optional bool [B,N,D]. Query states are built
                only here. If omitted, every finite-rank target is predicted.

        Returns:
            content: [B,N,D,d] true-value content stream.
            query: sparse query state with hidden shape [Q,d].
        """
        x_num = batch.x_num
        x_cat = batch.x_cat
        B, N, D = x_num.shape
        device = x_num.device

        if rank.shape != (B, N, D):
            raise ValueError(f"rank shape {tuple(rank.shape)} != table shape {(B, N, D)}")
        if N > self.cfg.max_episode_rows:
            raise ValueError(f"N={N} exceeds max_episode_rows={self.cfg.max_episode_rows}.")
        if D > self.cfg.max_cols:
            raise ValueError(f"D={D} exceeds max_cols={self.cfg.max_cols}.")

        finite_target = (rank > RANK_OBSERVED) & (rank < RANK_NEVER)
        if prediction_mask is None:
            prediction_mask = finite_target
        else:
            prediction_mask = prediction_mask.to(device=device, dtype=torch.bool)
            if prediction_mask.shape != rank.shape:
                raise ValueError(
                    f"prediction_mask shape {tuple(prediction_mask.shape)} != rank shape {tuple(rank.shape)}"
                )
            illegal = prediction_mask & ~finite_target
            if bool(illegal.any().item()):
                raise ValueError(
                    "prediction_mask may select only finite-rank query cells "
                    "(not observed or RANK_NEVER cells)."
                )

        coords = prediction_mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            raise ValueError("No prediction cells selected; sparse query stream would be empty.")

        col_types = batch.col_types.to(device=device, dtype=torch.long)
        cat_decode_types = batch.cat_decode_types.to(device=device, dtype=torch.long)
        type_ids = expand_per_col(col_types, B, N, D)
        cat_type_ids = expand_per_col(cat_decode_types, B, N, D)

        is_num = type_ids == NUMERICAL
        observed = rank == RANK_OBSERVED
        observed_num = observed & is_num

        x_cat_clamped = x_cat.clamp(min=0, max=self.cfg.k_max - 1).long()
        cat_type_clamped = cat_type_ids.clamp(
            min=0, max=self.cfg.num_cat_decode_types - 1
        )

        if self.cfg.unified_cat_encoding:
            x_unified = torch.where(is_num, x_num, x_cat_clamped.to(x_num.dtype))
            observed_f = observed.to(x_unified.dtype)
            count = observed_f.sum(dim=1).clamp(min=1.0)
            mean = (x_unified * observed_f).sum(dim=1) / count

            if self.cfg.context_normalize:
                sq_mean = (x_unified.pow(2) * observed_f).sum(dim=1) / count
                std = torch.sqrt((sq_mean - mean.pow(2)).clamp(min=0.0) + 1e-6)
                x_input = ((x_unified - mean.unsqueeze(1)) / std.unsqueeze(1)).clamp(
                    min=-100.0, max=100.0
                )
            else:
                x_input = x_unified

            content_value = self.num_value_mlp(x_input.unsqueeze(-1))
        else:
            observed_num_f = observed_num.to(x_num.dtype)
            count = observed_num_f.sum(dim=1).clamp(min=1.0)
            mean = (x_num * observed_num_f).sum(dim=1) / count

            if self.cfg.context_normalize:
                sq_mean = (x_num.pow(2) * observed_num_f).sum(dim=1) / count
                std = torch.sqrt((sq_mean - mean.pow(2)).clamp(min=0.0) + 1e-6)
                x_num_input = ((x_num - mean.unsqueeze(1)) / std.unsqueeze(1)).clamp(
                    min=-100.0, max=100.0
                )
            else:
                x_num_input = x_num

            num_vec = self.num_value_mlp(x_num_input.unsqueeze(-1))
            cat_vec = self.cat_value_emb[cat_type_clamped, x_cat_clamped]
            content_value = torch.where(is_num.unsqueeze(-1), num_vec, cat_vec)

        if self.cfg.drop_type_origin_emb:
            content = self.content_norm(content_value)
            query_hidden = self.query_token.unsqueeze(0).expand(coords.shape[0], -1)
            query_hidden = self._normalize_query(query_hidden)
        else:
            origin_ids = finite_target.long()
            content = self.content_norm(
                content_value + self.type_emb(type_ids) + self.origin_emb(origin_ids)
            )

            b, r, c = coords.unbind(dim=1)
            query_type = self.type_emb(type_ids[b, r, c])
            query_origin = self.origin_emb(torch.ones_like(b))
            query_hidden = self._normalize_query(
                self.query_token.unsqueeze(0) + query_type + query_origin
            )

        b, r, c = coords.unbind(dim=1)
        query = SparseQueryState(
            hidden=query_hidden,
            coords=coords,
            ranks=rank[b, r, c],
            batch_size=B,
            n_rows=N,
            n_cols=D,
        )
        return content, query


# ---------------------------------------------------------------------
# Masked axial multi-head attention.
# ---------------------------------------------------------------------


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def project_kv(self, kv_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        G, S, d = kv_in.shape
        H, hd = self.n_heads, self.head_dim
        k = self.k_proj(kv_in).view(G, S, H, hd).transpose(1, 2)
        v = self.v_proj(kv_in).view(G, S, H, hd).transpose(1, 2)
        return k, v

    def attend_projected(
        self,
        query_in: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        *,
        allow_empty_rows: bool,
    ) -> torch.Tensor:
        G, L, d = query_in.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(query_in).view(G, L, H, hd).transpose(1, 2)

        mask = None if attn_mask is None else attn_mask.unsqueeze(1)
        has_any = None
        if mask is not None and allow_empty_rows:
            has_any = mask.any(dim=-1, keepdim=True)
            # SDPA cannot softmax an all-masked row. Widen only for numerical
            # safety, then explicitly zero that row's attention contribution.
            mask = mask | (~has_any)
        if mask is not None and mask.stride(-1) != 1:
            # SDPA's fused (non-math) kernels require attn_mask to be
            # stride-1 in the last dim; silently falling back to math costs
            # ~4x memory on exactly the calls where that's most expensive
            # (see _axis_rank's col-axis comment for how this arises).
            mask = mask.contiguous()

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        if has_any is not None:
            out = out * has_any.to(out.dtype)

        out = out.transpose(1, 2).reshape(G, L, d)
        out = self.out_proj(out)
        if has_any is not None:
            # Also remove out_proj's bias on no-evidence rows.
            row_has_any = has_any[:, 0, :, 0].unsqueeze(-1)
            out = out * row_has_any.to(out.dtype)
        return out

    def forward(
        self,
        query_in: torch.Tensor,
        kv_in: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        *,
        allow_empty_rows: bool = False,
    ) -> torch.Tensor:
        k, v = self.project_kv(kv_in)
        return self.attend_projected(
            query_in, k, v, attn_mask, allow_empty_rows=allow_empty_rows
        )


def _reshape_for_axis(x: torch.Tensor, axis: str) -> torch.Tensor:
    B, N, D, d = x.shape
    if axis == "row":
        return x.reshape(B * N, D, d)
    if axis == "col":
        return x.permute(0, 2, 1, 3).reshape(B * D, N, d)
    raise ValueError(f"Unknown axis {axis!r}")


def _reshape_back(x: torch.Tensor, axis: str, B: int, N: int, D: int) -> torch.Tensor:
    d = x.shape[-1]
    if axis == "row":
        return x.reshape(B, N, D, d)
    if axis == "col":
        return x.reshape(B, D, N, d).permute(0, 2, 1, 3)
    raise ValueError(f"Unknown axis {axis!r}")


def _two_stream_axis_step(
    *,
    content: torch.Tensor,
    query: SparseQueryState,
    axis: str,
    axis_cache: AxisAttentionCache,
    content_attn: MaskedMultiheadAttention,
    query_attn: MaskedMultiheadAttention,
    content_norm: nn.Module,
    query_norm: nn.Module,
    dropout: nn.Module,
    post_ln: bool,
) -> tuple[torch.Tensor, SparseQueryState]:
    """
    One axial h/g update. Both streams read the same pre-update content.

    When attention weights are shared, K/V are projected once and reused for
    content and query attention, matching XLNet's two-stream implementation.
    """
    B, N, D, _ = content.shape
    c_in = _reshape_for_axis(content, axis)
    q_in = axis_cache.query_layout.pack(query.hidden)

    if post_ln:
        c_kv = c_in
        q_base = q_in
    else:
        c_kv = content_norm(c_in)
        q_base = query_norm(q_in)

    if query_attn is content_attn:
        k, v = content_attn.project_kv(c_kv)
        c_attn = content_attn.attend_projected(
            c_kv,
            k,
            v,
            axis_cache.content_mask,
            allow_empty_rows=False,
        )
        q_k = k.index_select(0, axis_cache.query_layout.active_group_ids)
        q_v = v.index_select(0, axis_cache.query_layout.active_group_ids)
        q_attn = query_attn.attend_projected(
            q_base,
            q_k,
            q_v,
            axis_cache.query_mask,
            allow_empty_rows=True,
        )
    else:
        c_attn = content_attn(
            c_kv, c_kv, axis_cache.content_mask, allow_empty_rows=False
        )
        q_kv = c_kv.index_select(0, axis_cache.query_layout.active_group_ids)
        q_attn = query_attn(
            q_base,
            q_kv,
            axis_cache.query_mask,
            allow_empty_rows=True,
        )

    if post_ln:
        c_out_axis = content_norm(c_in + dropout(c_attn))
        q_out_packed = query_norm(q_in + dropout(q_attn))
    else:
        c_out_axis = c_in + dropout(c_attn)
        q_out_packed = q_in + dropout(q_attn)

    content_out = _reshape_back(c_out_axis, axis, B, N, D)
    query_hidden = axis_cache.query_layout.unpack(q_out_packed, query.num_queries)
    return content_out, query.with_hidden(query_hidden)


class AxialTwoStreamLayer(nn.Module):
    """One row- or column-axis two-stream layer with a pointwise FFN."""

    def __init__(self, cfg: ModelConfig, axis: str):
        super().__init__()
        if axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
        self.axis = axis
        self.post_ln = cfg.post_ln
        self.share_stream_parameters = bool(cfg.share_stream_attn)
        d = cfg.d_model

        self.content_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.query_attn = (
            None
            if self.share_stream_parameters
            else MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        )

        self.content_norm1 = nn.LayerNorm(d)
        self.query_norm1 = None if self.share_stream_parameters else nn.LayerNorm(d)
        self.content_norm2 = nn.LayerNorm(d)
        self.query_norm2 = None if self.share_stream_parameters else nn.LayerNorm(d)

        self.content_ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
        )
        self.query_ffn = (
            None
            if self.share_stream_parameters
            else nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(4 * d, d),
            )
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        content: torch.Tensor,
        query: SparseQueryState,
        cache: TwoStreamAttentionCache,
    ) -> tuple[torch.Tensor, SparseQueryState]:
        query_attn = self.content_attn if self.query_attn is None else self.query_attn
        query_norm1 = self.content_norm1 if self.query_norm1 is None else self.query_norm1

        content, query = _two_stream_axis_step(
            content=content,
            query=query,
            axis=self.axis,
            axis_cache=cache.for_axis(self.axis),
            content_attn=self.content_attn,
            query_attn=query_attn,
            content_norm=self.content_norm1,
            query_norm=query_norm1,
            dropout=self.dropout,
            post_ln=self.post_ln,
        )

        query_ffn = self.content_ffn if self.query_ffn is None else self.query_ffn
        query_norm2 = self.content_norm2 if self.query_norm2 is None else self.query_norm2
        if self.post_ln:
            content = self.content_norm2(content + self.content_ffn(content))
            q = query_norm2(query.hidden + query_ffn(query.hidden))
        else:
            content = content + self.content_ffn(self.content_norm2(content))
            q = query.hidden + query_ffn(query_norm2(query.hidden))
        return content, query.with_hidden(q)


class PairedAxialTwoStreamLayer(nn.Module):
    """
    TabPFN-style row attention, then column attention, then one shared FFN.

    The content stream is dense. The query stream is sparse and contains only
    selected prediction cells. With share_stream_attn=True, attention, FFN,
    and LayerNorm parameters are reused between h and g as in XLNet.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.post_ln = cfg.post_ln
        self.share_stream_parameters = bool(cfg.share_stream_attn)
        self.global_query_bridge = bool(cfg.global_query_bridge)
        d = cfg.d_model

        self.row_content_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.row_query_attn = (
            None
            if self.share_stream_parameters
            else MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        )
        self.col_content_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.col_query_attn = (
            None
            if self.share_stream_parameters
            else MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
        )
        if self.global_query_bridge:
            self.bridge_attn = MaskedMultiheadAttention(d, cfg.n_heads, cfg.dropout)
            self.bridge_content_norm = nn.LayerNorm(d)
            self.bridge_query_norm = nn.LayerNorm(d)
        else:
            self.bridge_attn = None
            self.bridge_content_norm = None
            self.bridge_query_norm = None

        self.row_content_norm = nn.LayerNorm(d)
        self.row_query_norm = None if self.share_stream_parameters else nn.LayerNorm(d)
        self.col_content_norm = nn.LayerNorm(d)
        self.col_query_norm = None if self.share_stream_parameters else nn.LayerNorm(d)
        self.content_ffn_norm = nn.LayerNorm(d)
        self.query_ffn_norm = None if self.share_stream_parameters else nn.LayerNorm(d)

        self.content_ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d)
        )
        self.query_ffn = (
            None
            if self.share_stream_parameters
            else nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(4 * d, d),
            )
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def _axis_step(
        self,
        content: torch.Tensor,
        query: SparseQueryState,
        cache: TwoStreamAttentionCache,
        axis: str,
        content_attn: MaskedMultiheadAttention,
        query_attn: Optional[MaskedMultiheadAttention],
        content_norm: nn.Module,
        query_norm: Optional[nn.Module],
    ) -> tuple[torch.Tensor, SparseQueryState]:
        return _two_stream_axis_step(
            content=content,
            query=query,
            axis=axis,
            axis_cache=cache.for_axis(axis),
            content_attn=content_attn,
            query_attn=content_attn if query_attn is None else query_attn,
            content_norm=content_norm,
            query_norm=content_norm if query_norm is None else query_norm,
            dropout=self.dropout,
            post_ln=self.post_ln,
        )

    def forward(
        self,
        content: torch.Tensor,
        query: SparseQueryState,
        cache: TwoStreamAttentionCache,
    ) -> tuple[torch.Tensor, SparseQueryState]:
        content, query = self._axis_step(
            content,
            query,
            cache,
            "row",
            self.row_content_attn,
            self.row_query_attn,
            self.row_content_norm,
            self.row_query_norm,
        )
        content, query = self._axis_step(
            content,
            query,
            cache,
            "col",
            self.col_content_attn,
            self.col_query_attn,
            self.col_content_norm,
            self.col_query_norm,
        )

        if self.global_query_bridge and cache.global_bridge is not None:
            query = _global_query_bridge_step(
                content=content,
                query=query,
                cache=cache.global_bridge,
                bridge_attn=self.bridge_attn,
                content_norm=self.bridge_content_norm,
                query_norm=self.bridge_query_norm,
                dropout=self.dropout,
                post_ln=self.post_ln,
            )

        query_ffn = self.content_ffn if self.query_ffn is None else self.query_ffn
        query_ffn_norm = (
            self.content_ffn_norm if self.query_ffn_norm is None else self.query_ffn_norm
        )
        if self.post_ln:
            content = self.content_ffn_norm(content + self.content_ffn(content))
            q = query_ffn_norm(query.hidden + query_ffn(query.hidden))
        else:
            content = content + self.content_ffn(self.content_ffn_norm(content))
            q = query.hidden + query_ffn(query_ffn_norm(query.hidden))
        return content, query.with_hidden(q)


# ---------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------


class PermARCompletionModel(nn.Module):
    """
    One-pass permutation-AR model with a dense content stream and sparse
    target-mapped query stream.

    This preserves the exact two-stream semantics at every selected target:
    h carries true content; g starts from a trainable query token and reads
    only strictly earlier h states. Query states are never keys/values, so
    computing g only for prediction cells is exact, not an approximation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = PermARTokenizer(cfg)

        if cfg.global_query_bridge and not cfg.tabpfn_style_layers:
            raise ValueError(
                "global_query_bridge is only wired into PairedAxialTwoStreamLayer "
                "(tabpfn_style_layers=True) -- AxialTwoStreamLayer alternates single "
                "axes, so 'after row+col, before FFN' isn't well-defined per-layer "
                "there. Not implemented for tabpfn_style_layers=False."
            )

        if cfg.tabpfn_style_layers:
            if cfg.num_row_layers != cfg.num_row_context_layers:
                raise ValueError(
                    "tabpfn_style_layers requires num_row_layers == "
                    f"num_row_context_layers (got {cfg.num_row_layers} vs "
                    f"{cfg.num_row_context_layers})."
                )
            self.layers = nn.ModuleList(
                [PairedAxialTwoStreamLayer(cfg) for _ in range(cfg.num_row_layers)]
            )
        else:
            axes: list[str] = []
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
            self.layers = nn.ModuleList([AxialTwoStreamLayer(cfg, a) for a in axes])

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.num_head = nn.Linear(cfg.d_model, 1)
        self.cat_head = TypedCategoricalHead(cfg)

    @staticmethod
    def _scatter_query_hidden(query: SparseQueryState) -> torch.Tensor:
        """Scatter final [Q,d] states once for compatibility with existing heads/loss."""
        B, N, D = query.batch_size, query.n_rows, query.n_cols
        flat_index = (
            query.batch_index * (N * D) + query.row_index * D + query.col_index
        )
        full = query.hidden.new_zeros((B * N * D, query.hidden.shape[-1]))
        full = full.index_copy(0, flat_index, query.hidden)
        return full.reshape(B, N, D, query.hidden.shape[-1])

    def forward(
        self,
        batch: TableTensorBatch,
        rank: torch.Tensor,
        context_row_mask: Optional[torch.Tensor] = None,
        prediction_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            rank: [B,N,D] reveal ranks.
            context_row_mask: optional [B,N] inductive-row context mask.
            prediction_mask: optional [B,N,D] subset for which g states and
                losses are required. Defaults to all finite-rank targets.
        """
        content, query = self.tokenizer(batch, rank, prediction_mask)
        cache = _build_attention_cache(
            rank, query, context_row_mask, build_global_bridge=self.cfg.global_query_bridge
        )

        use_checkpoint = self.training and self.cfg.activation_checkpointing
        # Fixed across every layer -- only query.hidden actually changes
        # layer to layer, so only it (plus content) needs to be a
        # checkpoint()-tracked tensor; these are captured by run_block's
        # closure instead.
        coords, ranks = query.coords, query.ranks
        batch_size, n_rows, n_cols = query.batch_size, query.n_rows, query.n_cols

        for layer in self.layers:
            if not use_checkpoint:
                content, query = layer(content, query, cache)
                continue

            def run_block(
                content_in: torch.Tensor,
                query_hidden_in: torch.Tensor,
                *,
                _layer=layer,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                query_in = SparseQueryState(
                    hidden=query_hidden_in,
                    coords=coords,
                    ranks=ranks,
                    batch_size=batch_size,
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
                content_out, query_out = _layer(content_in, query_in, cache)
                return content_out, query_out.hidden

            content, query_hidden = checkpoint(
                run_block, content, query.hidden, use_reentrant=False, preserve_rng_state=True
            )
            query = query.with_hidden(query_hidden)

        query = query.with_hidden(self.final_norm(query.hidden))
        h = self._scatter_query_hidden(query)
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


# ---------------------------------------------------------------------
# XLNet-style partial prediction: subsample which query cells actually get a
# g-state and a loss term this step, instead of always scoring every cell in
# task.query_mask. Uniform subsampling + typed_mse_ce_loss's mean reduction
# over queried cells (see losses.py) is already an unbiased Monte Carlo
# estimator of the full-query-set mean loss -- no reweighting needed. This is
# a small change on top of the sparse tokenizer specifically because
# PermARTokenizer already takes an arbitrary prediction_mask; subsampling
# just means passing it a smaller one.
# ---------------------------------------------------------------------


def _subsample_query_mask(
    query_mask: np.ndarray, max_predict_cells: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-batch-item: keep at most max_predict_cells True entries, chosen uniformly at random."""
    query_mask = query_mask.copy()
    for b in range(query_mask.shape[0]):
        flat = query_mask[b].reshape(-1)
        true_idx = np.flatnonzero(flat)
        if true_idx.size > max_predict_cells:
            drop = rng.choice(true_idx, size=true_idx.size - max_predict_cells, replace=False)
            flat[drop] = False
    return query_mask


def compute_task_loss_onepass(
    model: PermARCompletionModel,
    full,
    task: CompletionTask,
    plan: FactorizationPlan,
    device: torch.device,
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
    max_predict_cells: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
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

    query_np = task.query_mask[None, :, :]
    if max_predict_cells is not None:
        if rng is None:
            raise ValueError("max_predict_cells requires an rng.")
        query_np = _subsample_query_mask(query_np, max_predict_cells, rng)
    query_t = torch.as_tensor(query_np, dtype=torch.bool, device=device)
    out = model(
        batch,
        rank_t,
        context_row_mask=context_row_mask_t,
        prediction_mask=query_t,
    )
    loss_out = typed_mse_ce_loss(out, batch, query_t, num_weight=num_weight, cat_weight=cat_weight)

    metrics = dict(loss_out.metrics)
    metrics["factorization_steps"] = float(plan.num_steps)
    # Cells actually scored this step -- equals task.num_query_cells unless
    # max_predict_cells subsampled it down.
    metrics["query_cells"] = float(query_np.sum())

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
    max_predict_cells: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
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

    query_np = np.stack([task.query_mask for task in task_list])
    if max_predict_cells is not None:
        if rng is None:
            raise ValueError("max_predict_cells requires an rng.")
        query_np = _subsample_query_mask(query_np, max_predict_cells, rng)
    query_t = torch.as_tensor(query_np, dtype=torch.bool, device=device)
    out = model(
        batch,
        rank_t,
        context_row_mask=context_row_mask_t,
        prediction_mask=query_t,
    )
    loss_out = typed_mse_ce_loss(out, batch, query_t, num_weight=num_weight, cat_weight=cat_weight)

    metrics = dict(loss_out.metrics)
    metrics["factorization_steps"] = float(np.mean([plan.num_steps for plan in plan_list]))
    # Cells actually scored this step (per-episode mean) -- equals the plain
    # task-average unless max_predict_cells subsampled it down.
    metrics["query_cells"] = float(query_np.sum(axis=(1, 2)).mean())

    return StepLossOutput(loss=loss_out.loss, metrics=metrics)
