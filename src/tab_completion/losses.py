# src/tab_completion/losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
    metrics: Dict[str, float]


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

        metrics["loss_num"] = float(num_loss.detach().cpu())
        metrics["num_mse"] = float(num_loss.detach().cpu())
        metrics["num_cells"] = int(num_query.sum().detach().cpu())
    else:
        metrics["loss_num"] = 0.0
        metrics["num_mse"] = 0.0
        metrics["num_cells"] = 0

    if torch.any(cat_query):
        logits = pred.cat_logits[cat_query]  # [num_cat_query, K_max]
        targets = batch.x_cat[cat_query].long()

        cat_loss = F.cross_entropy(logits, targets)
        losses.append(cat_weight * cat_loss)

        with torch.no_grad():
            pred_class = logits.argmax(dim=-1)
            acc = (pred_class == targets).float().mean()

        metrics["loss_cat"] = float(cat_loss.detach().cpu())
        metrics["cat_acc"] = float(acc.detach().cpu())
        metrics["cat_cells"] = int(cat_query.sum().detach().cpu())
    else:
        metrics["loss_cat"] = 0.0
        metrics["cat_acc"] = 0.0
        metrics["cat_cells"] = 0

    if not losses:
        raise ValueError("No queried numerical or categorical cells found.")

    loss = sum(losses)
    metrics["loss_total"] = float(loss.detach().cpu())
    metrics["query_cells"] = int(query_mask.sum().detach().cpu())

    return LossOutput(loss=loss, metrics=metrics)