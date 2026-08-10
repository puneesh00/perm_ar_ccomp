# src/tab_completion/episode_utils.py

from __future__ import annotations

import numpy as np
import torch

from tab_completion.sampling import CompletionTask
from tab_completion.model import TableTensorBatch
from tab_completion.synthetic_data import FullSyntheticTable


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
    x_cat_np = full.x_cat[np.ix_(rows, cols)]

    return TableTensorBatch(
        x_num=torch.as_tensor(x_num_np[None, :, :], dtype=torch.float32, device=device),
        x_cat=torch.as_tensor(x_cat_np[None, :, :], dtype=torch.long, device=device),
        col_types=torch.as_tensor(full.col_types[cols], dtype=torch.long, device=device),
        cat_cardinalities=torch.as_tensor(
            full.cat_cardinalities[cols],
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
    x_cat_np = np.stack(
        [full.x_cat[np.ix_(t.row_idx, t.col_idx)] for full, t in zip(full_list, task_list)]
    )
    col_types_np = np.stack(
        [full.col_types[t.col_idx] for full, t in zip(full_list, task_list)]
    )
    cat_card_np = np.stack(
        [full.cat_cardinalities[t.col_idx] for full, t in zip(full_list, task_list)]
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