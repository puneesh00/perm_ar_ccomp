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