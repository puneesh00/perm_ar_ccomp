# src/tab_completion/losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn.functional as F

from tab_completion.model import (
    NUMERICAL,
    CATEGORICAL,
    TableTensorBatch,
    ModelOutput,
    expand_per_col,
)


@dataclass
class LossOutput:
    loss: torch.Tensor
    # Values are detached GPU tensors (0.0/0 sentinels on the empty-cell-type
    # branches), not eagerly synced Python scalars -- see typed_mse_ce_loss.
    # Callers convert with float()/int() themselves, at whatever cadence they
    # actually need a host-side value.
    metrics: Dict[str, Union[torch.Tensor, float, int]]


def typed_mse_ce_loss(
    pred: ModelOutput,
    batch: TableTensorBatch,
    query_mask: torch.Tensor,
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
) -> LossOutput:
    """
    v0 typed loss:
      numerical queried cells -> MSE
      categorical queried cells -> cross entropy

    All numerical targets are assumed normalized.
    """
    device = batch.x_num.device
    col_types = batch.col_types.to(device=device, dtype=torch.long)

    B, N, D = query_mask.shape
    type_ids = expand_per_col(col_types, B, N, D)

    num_query = query_mask & (type_ids == NUMERICAL)
    cat_query = query_mask & (type_ids == CATEGORICAL)

    losses = []
    metrics: Dict[str, float] = {}

    if torch.any(num_query):
        y_num = batch.x_num[num_query]
        mu = pred.num_mu[num_query]
        num_loss = F.mse_loss(mu, y_num)
        losses.append(num_weight * num_loss)

        # Stay on-device (no .cpu()/float()/int()) -- this runs once per
        # micro-batch, and under grad accumulation that can be dozens of
        # times per optimizer step. Forcing a host sync here every call
        # serializes the accumulation loop instead of letting CUDA overlap
        # consecutive micro-batches. Callers that need Python scalars
        # (logging, eval) already convert with float()/int() themselves,
        # typically already gated behind a log/eval-interval check -- so the
        # sync now happens there instead, at the same reduced frequency.
        metrics["loss_num"] = num_loss.detach()
        metrics["num_mse"] = num_loss.detach()
        metrics["num_cells"] = num_query.sum().detach()
    else:
        metrics["loss_num"] = 0.0
        metrics["num_mse"] = 0.0
        metrics["num_cells"] = 0

    if torch.any(cat_query):
        logits = pred.cat_logits[cat_query]  # [num_cat_query, K_max]
        targets = batch.x_cat[cat_query].long()

        # targets may contain -100 (episode_utils._densify_queried_
        # categorical_columns' OOV sentinel: a query cell whose true class
        # never appeared in this column's context/evidence, so there is no
        # learnable codebook slot for it). ignore_index=-100 excludes those
        # cells from the loss entirely rather than crashing or training
        # against an arbitrary target; if EVERY queried cell in this
        # micro-batch is such a cell, cross_entropy's mean-reduction over
        # zero non-ignored elements returns nan by design -- the caller's
        # grad-accum loop already treats that the same as any other
        # micro-batch-level numerical issue, not special-cased here.
        cat_loss = F.cross_entropy(logits, targets, ignore_index=-100)
        losses.append(cat_weight * cat_loss)

        with torch.no_grad():
            pred_class = logits.argmax(dim=-1)
            # -100 never equals a predicted class in [0, K_C), so an OOV
            # cell is automatically counted wrong here -- no special-casing
            # needed to satisfy "count it as incorrect, not silently
            # dropped" at eval time.
            acc = (pred_class == targets).float().mean()

        metrics["loss_cat"] = cat_loss.detach()
        metrics["cat_acc"] = acc.detach()
        metrics["cat_cells"] = cat_query.sum().detach()
    else:
        metrics["loss_cat"] = 0.0
        metrics["cat_acc"] = 0.0
        metrics["cat_cells"] = 0

    if not losses:
        raise ValueError("No queried numerical or categorical cells found.")

    loss = sum(losses)
    metrics["loss_total"] = loss.detach()
    metrics["query_cells"] = query_mask.sum().detach()

    return LossOutput(loss=loss, metrics=metrics)