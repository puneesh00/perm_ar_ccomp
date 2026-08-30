# src/tab_completion/episode_utils.py

from __future__ import annotations

import numpy as np
import torch

from tab_completion.sampling import CompletionTask
from tab_completion.model import TableTensorBatch, CATEGORICAL
from tab_completion.synthetic_data import FullSyntheticTable
from tab_completion.model_perm_ar import get_context_row_mask_from_task


def _densify_queried_categorical_columns(
    full: FullSyntheticTable,
    task: CompletionTask,
    cols: np.ndarray,
    x_cat_local: np.ndarray,
    cat_card_local: np.ndarray,
) -> None:
    """
    Mutates x_cat_local/cat_card_local IN PLACE: for every categorical
    column with at least one queried cell this episode, remaps that
    column's realized values (from the permitted-evidence rows only) to a
    dense local codebook 0..K_C-1 and shrinks cat_card_local for that
    column to K_C -- instead of the raw full-table ids / full-table K the
    slice otherwise carries unchanged.

    Fixes a real non-identifiability bug: without this, two episodes whose
    context/evidence realizes a DIFFERENT K_full-subset of a column produce
    IDENTICAL normalized/embedded representations for the query placeholder
    (context_normalize's z-score and the frequency-weighted categorical
    placeholder both only see AGGREGATE evidence statistics, never which
    raw ids were actually involved) while the decode head is required to
    use a DIFFERENT slot mapping each time -- e.g. context {0,1} and
    context {1,2} out of a 3-class column both z-normalize to the same
    +-1 pair, but slot 0 has to mean class 0 in one episode and class 1 in
    the other. Neither origin_emb nor decoder capacity can resolve this;
    only fixing what the raw ids ARE can. Matches official TabPFN's own
    precedent (LabelEncoder fit on the training/context rows, applied to
    both train and test) and this repo's own train_tabpfn_v1_baseline.
    build_xy, which already does exactly this for its one fixed target
    column -- this generalizes it to every queried categorical column
    under the general completion-task samplers (random_cell, column_block,
    row_block, cell_block, label_feature), not just target-prediction.

    Evidence rows per queried column: context rows under inductive_rows
    conditioning (mirrors what row-axis attention already restricts a
    query row to see); every observed row in that column under
    transductive (no such restriction exists there either) -- see
    get_context_row_mask_from_task.

    A cell whose raw value never appears in that column's evidence set
    (only possible for a query cell, or for an observed cell belonging to
    a non-context row under an irregular per-row query schedule, e.g.
    cell_block -- TargetPredictionSampler's context/query split can't
    produce this) is mapped to -100, torch's cross_entropy ignore_index:
    losses.py's typed_mse_ce_loss passes ignore_index=-100 explicitly, so
    an out-of-evidence QUERY cell contributes zero to the loss (there is no
    learnable signal for it) while still counting as wrong in the accuracy
    metric (-100 never equals a predicted class in [0, K_C)). An
    out-of-evidence OBSERVED cell instead gets defensively clamped to 0 by
    the tokenizers' existing `.clamp(min=0, ...)` on x_cat (pre-existing
    behavior in this codebase, not new) -- a degraded-but-safe aliasing for
    an edge case none of this repo's actual training runs hit.
    """
    col_types_local = full.col_types[cols]
    query_mask = task.query_mask
    observed_mask = task.observed_mask
    context_row_mask = get_context_row_mask_from_task(task)  # [n_ep] bool or None

    for local_c in range(len(cols)):
        if col_types_local[local_c] != CATEGORICAL:
            continue
        if not query_mask[:, local_c].any():
            continue  # not queried this episode -- leave raw ids/full-K alone

        col_observed = observed_mask[:, local_c]
        evidence_mask = (context_row_mask & col_observed) if context_row_mask is not None else col_observed

        if not evidence_mask.any():
            # No evidence at all for this column this episode -- every
            # query cell in it is unlearnable by construction.
            x_cat_local[query_mask[:, local_c], local_c] = -100
            continue

        full_k = int(cat_card_local[local_c])
        realized = np.unique(x_cat_local[evidence_mask, local_c])
        realized = realized[(realized >= 0) & (realized < full_k)]
        k_c = int(len(realized))

        remap = np.full(full_k, fill_value=-100, dtype=np.int64)
        remap[realized] = np.arange(k_c, dtype=np.int64)

        col_vals = x_cat_local[:, local_c]
        valid_idx = (col_vals >= 0) & (col_vals < full_k)
        looked_up = remap[np.where(valid_idx, col_vals, 0)]
        x_cat_local[:, local_c] = np.where(valid_idx, looked_up, -100)
        cat_card_local[local_c] = k_c


def task_to_torch_batch(
    full: FullSyntheticTable,
    task: CompletionTask,
    device: torch.device,
) -> TableTensorBatch:
    """
    Slice full synthetic table according to task.row_idx and task.col_idx.

    Returns a TableTensorBatch with B=1.
    """
    rows = task.row_idx
    cols = task.col_idx

    x_num_np = full.x_num[np.ix_(rows, cols)]
    x_cat_np = full.x_cat[np.ix_(rows, cols)].copy()
    cat_card_np = full.cat_cardinalities[cols].copy()
    _densify_queried_categorical_columns(full, task, cols, x_cat_np, cat_card_np)

    return TableTensorBatch(
        x_num=torch.as_tensor(x_num_np[None, :, :], dtype=torch.float32, device=device),
        x_cat=torch.as_tensor(x_cat_np[None, :, :], dtype=torch.long, device=device),
        col_types=torch.as_tensor(full.col_types[cols], dtype=torch.long, device=device),
        cat_cardinalities=torch.as_tensor(
            cat_card_np,
            dtype=torch.long,
            device=device,
        ),
        cat_decode_types=torch.as_tensor(
            full.cat_decode_types[cols],
            dtype=torch.long,
            device=device,
        ),
    )


def tasks_to_torch_batch(
    full_list: list[FullSyntheticTable],
    task_list: list[CompletionTask],
    device: torch.device,
) -> TableTensorBatch:
    """
    Batched counterpart to task_to_torch_batch: stacks B independently
    sampled (full, task) pairs into one real [B, N, D] batch instead of a
    Python loop of B separate B=1 forward passes.

    Requires every task's (row_idx, col_idx) to have the same length --
    true whenever n_context/n_query/n_cols are fixed by config, which is
    the case for TargetPredictionSampler runs (all runs this was written
    for). col_types/cat_cardinalities are stacked to [B, D] rather than
    left as a single shared [D] vector, since which columns are
    numerical/categorical is itself re-randomized per fresh table (see
    synthetic_data_tabpfn.py's p_categorical draw) -- model.expand_per_col
    is what makes the rest of the model accept this [B, D] form.
    cat_decode_types is left as a single shared [D]: it's always
    np.arange(n_cols), deterministic across every table.
    """
    n = len(task_list)
    if n == 0:
        raise ValueError("tasks_to_torch_batch requires at least one task.")
    if len(full_list) != n:
        raise ValueError("full_list and task_list must have the same length.")

    N = len(task_list[0].row_idx)
    D = len(task_list[0].col_idx)
    for task in task_list:
        if len(task.row_idx) != N or len(task.col_idx) != D:
            raise ValueError(
                "tasks_to_torch_batch requires every task to share the same "
                f"(N, D) shape; got ({len(task.row_idx)}, {len(task.col_idx)}) "
                f"vs ({N}, {D})."
            )

    x_num_np = np.stack(
        [full.x_num[np.ix_(t.row_idx, t.col_idx)] for full, t in zip(full_list, task_list)]
    )

    x_cat_list = []
    cat_card_list = []
    for full, t in zip(full_list, task_list):
        x_cat_t = full.x_cat[np.ix_(t.row_idx, t.col_idx)].copy()
        cat_card_t = full.cat_cardinalities[t.col_idx].copy()
        _densify_queried_categorical_columns(full, t, t.col_idx, x_cat_t, cat_card_t)
        x_cat_list.append(x_cat_t)
        cat_card_list.append(cat_card_t)
    x_cat_np = np.stack(x_cat_list)
    cat_card_np = np.stack(cat_card_list)

    col_types_np = np.stack(
        [full.col_types[t.col_idx] for full, t in zip(full_list, task_list)]
    )
    # Deterministic (always arange(n_cols)) -- shared [D], not stacked.
    cat_decode_types_np = full_list[0].cat_decode_types[task_list[0].col_idx]

    return TableTensorBatch(
        x_num=torch.as_tensor(x_num_np, dtype=torch.float32, device=device),
        x_cat=torch.as_tensor(x_cat_np, dtype=torch.long, device=device),
        col_types=torch.as_tensor(col_types_np, dtype=torch.long, device=device),
        cat_cardinalities=torch.as_tensor(cat_card_np, dtype=torch.long, device=device),
        cat_decode_types=torch.as_tensor(cat_decode_types_np, dtype=torch.long, device=device),
    )


def mask_to_torch(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert local episode mask [N, D] to batched torch mask [1, N, D].
    """
    return torch.as_tensor(mask[None, :, :], dtype=torch.bool, device=device)


def make_step_query_mask(
    shape: tuple[int, int],
    step_coords: np.ndarray,
) -> np.ndarray:
    """
    Build local query mask for one factorization step.

    shape:
        episode shape [N, D]

    step_coords:
        local coordinates [k, 2]
    """
    q = np.zeros(shape, dtype=bool)
    q[step_coords[:, 0], step_coords[:, 1]] = True
    return q
