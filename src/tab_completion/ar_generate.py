# src/tab_completion/ar_generate.py
"""
Genuine (non-teacher-forced) autoregressive generation for two_stream_ar /
two_stream_ar_sparse checkpoints trained with factorization=perm_ar.

Why this exists: training (and every eval path built so far --
compute_task_loss_onepass(_batched), scripts/eval_openml_incontext.py) is
teacher-forced. PermARTokenizer's content stream always carries a cell's
literal ground-truth value regardless of when that cell is "revealed" --
what makes AR training correct is the rank-based ATTENTION MASK (a reader at
rank R can only attend to content at rank < R, via _rank_visibility in
model_perm_ar.py / the sparse equivalent), not any value substitution in the
content tensor. That's the right design for training (avoids exposure bias,
and lets every step be scored in one forward pass), but it means every AR
eval number produced by this codebase so far let the model condition on
GROUND TRUTH for any already-"revealed" cell -- not available at real
inference time, where the whole point of a later AR step is to predict
something no one has told you yet.

This module runs one real forward pass PER AR STEP instead: at step i, only
cells revealed at steps < i (plus the episode's original context) get
rank=RANK_OBSERVED, using the MODEL'S OWN prediction from whichever earlier
step revealed them (substituted into the working value tensors -- not the
ground truth, which we no longer even look at until final scoring). Step i's
own coords get rank=1 (the only positive rank present this call). Every
not-yet-revealed cell gets RANK_NEVER -- fully invisible this pass, exactly
like a real generation process that cannot see future predictions at all.
Cost is O(num_steps) forward passes instead of the training path's O(1).
For unit="column"/"row", PermARFactorizer already batches optimally (one
step = one column/row, covering every row that has it queried, so
num_steps = number of distinct columns/rows queried -- typically small).
For unit="cell", PermARFactorizer's training-time global cross-row shuffle
would force needless serialization here (two cells in different rows can
never affect each other -- see model_perm_ar_sparse.py's inductive-row
masking -- so there's no reason to run them as separate sequential passes
just because a global shuffle interleaved them); _row_batched_cell_steps
below gives each row its own independent order instead and batches by
depth-within-row, so num_steps = max missing cells in any single row rather
than the total across the whole episode. group_size > 1 (reveal several of
a row's own cells together per step) is still available on top of that for
a further speed/fidelity trade.

Only supports the two_stream_ar / two_stream_ar_sparse family (arbitrary
reveal order via the rank tensor). single_stream is architecturally
parallel-only (see model_single_stream.py's module docstring) and has no
notion of a "later" AR step to generate into -- there is nothing for this
module to do for that architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from tab_completion.model import TableTensorBatch, CATEGORICAL, NUMERICAL
from tab_completion.sampling import CompletionTask
from tab_completion.synthetic_data import FullSyntheticTable
from tab_completion.factorization import PermARFactorizer
from tab_completion.model_perm_ar import (
    RANK_OBSERVED,
    RANK_NEVER,
    get_context_row_mask_from_task,
)


def numeric_context_stats(
    x_num: np.ndarray, observed_mask: np.ndarray
) -> tuple:
    """
    Per-column (context-only) mean/std, bit-for-bit matching the formula
    every tokenizer (SingleStreamTokenizer, PermARTokenizer, the sparse
    equivalent) computes internally for context_normalize: mean/std of the
    OBSERVED cells in each column, dim=0 here since x_num is [N, D] (no
    batch dim) vs the model's [B, N, D].

    Needed because context_normalize only transforms the model's INPUT
    representation -- num_mu (the trained OUTPUT head) is never
    un-normalized internally, and is trained via plain MSE directly against
    the raw training target (see typed_mse_ce_loss), which for our
    synthetic SCM prior is already roughly standardized. Real OpenML numeric
    columns are NOT roughly standardized (arbitrary real-world units), so
    num_mu's raw output has to be mapped back with THESE stats -- computed
    the same way the model itself would -- before it's comparable to a raw
    real-world target value. See the conversation this was found in:
    comparing a ~N(0,1)-scale prediction against e.g. blood-transfusion's
    "Monetary" column (hundreds to thousands of cc) gives a garbage MSE
    that reflects a units mismatch, not model quality.
    """
    observed_f = observed_mask.astype(np.float64)
    count = np.clip(observed_f.sum(axis=0), 1.0, None)
    mean = (x_num.astype(np.float64) * observed_f).sum(axis=0) / count
    sq_mean = (x_num.astype(np.float64) ** 2 * observed_f).sum(axis=0) / count
    std = np.sqrt(np.clip(sq_mean - mean**2, 0.0, None) + 1e-6)
    return mean, std


def _row_batched_cell_steps(
    task: CompletionTask, rng: np.random.Generator, group_size: int = 1
) -> list:
    """
    Row-independent alternative to PermARFactorizer(unit="cell")'s global
    cross-row shuffle, used ONLY here for genuine sequential generation --
    never for training, and factorization.py is untouched.

    PermARFactorizer's global shuffle is harmless for training: everything
    happens in one teacher-forced pass regardless of step order, and
    cross-row attention is blocked anyway under inductive_rows conditioning
    (a query row can only read context rows or itself -- see
    model_perm_ar_sparse.py's col/global-axis masking). But for genuine
    sequential generation, "step order" is real wall-clock: two cells in
    different rows can never affect each other's prediction, so serializing
    them into separate forward passes just because a global shuffle happened
    to interleave them is pure waste.

    Instead: give each row its OWN independent random permutation of its own
    missing cells, then group by depth-within-row -- step d = every row's
    d-th cell (whichever rows still have one), one batched forward pass.
    Models exactly the same within-row dependencies as the global-shuffle
    version (nothing crosses rows either way), but sequential pass count
    drops from "total missing cells" to "max missing cells in any one row."
    """
    coords = task.query_coords_local()
    if len(coords) == 0:
        return []

    rows = np.unique(coords[:, 0])
    per_row_order = {}
    max_depth = 0
    for r in rows:
        row_coords = coords[coords[:, 0] == r]
        row_coords = row_coords[rng.permutation(len(row_coords))]
        per_row_order[int(r)] = row_coords
        max_depth = max(max_depth, len(row_coords))

    steps = []
    for d in range(0, max_depth, group_size):
        depth_coords = [
            per_row_order[int(r)][d : d + group_size]
            for r in rows
            if d < len(per_row_order[int(r)])
        ]
        if depth_coords:
            steps.append(np.concatenate(depth_coords, axis=0))
    return steps


@dataclass
class ARGenerationResult:
    rows: np.ndarray          # [K] local row index (within task.row_idx) per query cell
    cols: np.ndarray          # [K] local col index (within task.col_idx) per query cell
    step_index: np.ndarray    # [K] which AR step revealed this cell (0-indexed)
    y_true: np.ndarray        # [K] true value: class id for categorical cols, float for numerical
    y_pred: np.ndarray        # [K] predicted value, same convention
    y_proba: Optional[np.ndarray]  # [K, k_max] softmax probs for categorical cells, NaN rows for numerical
    is_categorical: np.ndarray     # [K] bool


@torch.no_grad()
def generate_ar(
    model,
    full: FullSyntheticTable,
    task: CompletionTask,
    device: torch.device,
    ar_unit: str = "cell",
    group_size: int = 1,
    rng: Optional[np.random.Generator] = None,
    is_sparse: bool = True,
) -> ARGenerationResult:
    """
    Real autoregressive generation: predict, substitute the model's own
    prediction back in as the revealed value, repeat for the next step.

    model: a PermARCompletionModel (model_perm_ar.py, is_sparse=False) or
        PermARCompletionModelSparse (model_perm_ar_sparse.py, is_sparse=True)
        instance, already in eval() mode.
    ar_unit: "cell" (one query cell per step, teacher-forced within a
        group_size>1 group only), "column" (all queried cells of one column
        revealed together per step -- parallel within the column, AR across
        columns), or "row".
    """
    if rng is None:
        rng = np.random.default_rng(0)

    rows_g, cols_g = task.row_idx, task.col_idx
    N, D = len(rows_g), len(cols_g)

    x_num_true = full.x_num[np.ix_(rows_g, cols_g)].astype(np.float32)
    x_cat_true = full.x_cat[np.ix_(rows_g, cols_g)].astype(np.int64)
    col_types_np = full.col_types[cols_g].astype(np.int64)
    cat_cardinalities_np = full.cat_cardinalities[cols_g].astype(np.int64)
    cat_decode_types_np = full.cat_decode_types[cols_g].astype(np.int64)

    # Context-only per-column mean/std, computed once (context rows never
    # change across steps). num_mu is trained via plain MSE against the raw
    # training target (roughly-standardized synthetic data) and is never
    # un-normalized internally -- see numeric_context_stats's docstring for
    # why this is required before num_mu is comparable to a real-world-scale
    # numeric target (e.g. real OpenML columns).
    _ctx_mean, _ctx_std = numeric_context_stats(x_num_true, task.observed_mask)

    # Working copies: ground truth for originally-observed cells, overwritten
    # with the model's own predictions as each AR step reveals more cells.
    # Never touched for still-hidden (RANK_NEVER) cells -- their true value
    # sits here too (dense tensors, fixed shape), but the rank-based
    # attention mask is what keeps them genuinely invisible, matching how
    # PermARTokenizer/the layer masks already work during normal training.
    x_num_work = x_num_true.copy()
    x_cat_work = x_cat_true.copy()

    rank = np.where(task.observed_mask, RANK_OBSERVED, RANK_NEVER).astype(np.int64)

    if ar_unit == "cell":
        # Row-batched scheduling (see _row_batched_cell_steps) -- column/row
        # units already batch optimally via PermARFactorizer as-is (one step
        # = one column/row, naturally covering every row that has it
        # queried), so only "cell" needed this.
        steps = _row_batched_cell_steps(task, rng, group_size=group_size)
    else:
        plan = PermARFactorizer(unit=ar_unit, group_size=group_size).build(task, rng)
        steps = plan.steps

    context_row_mask_np = get_context_row_mask_from_task(task)
    context_row_mask_t = None
    if context_row_mask_np is not None:
        context_row_mask_t = torch.as_tensor(
            context_row_mask_np[None, :], dtype=torch.bool, device=device
        )

    col_types_t = torch.as_tensor(col_types_np, dtype=torch.long, device=device)
    cat_card_t = torch.as_tensor(cat_cardinalities_np, dtype=torch.long, device=device)
    cat_decode_t = torch.as_tensor(cat_decode_types_np, dtype=torch.long, device=device)

    out_rows, out_cols, out_step, out_true, out_pred, out_proba, out_is_cat = (
        [], [], [], [], [], [], [],
    )
    k_max = int(cat_cardinalities_np.max()) if (col_types_np == CATEGORICAL).any() else 1

    for step_idx, coords in enumerate(steps):
        if len(coords) == 0:
            continue

        step_rank = rank.copy()
        step_rank[coords[:, 0], coords[:, 1]] = 1  # the only positive rank present this call

        x_num_t = torch.as_tensor(x_num_work[None], dtype=torch.float32, device=device)
        x_cat_t = torch.as_tensor(x_cat_work[None], dtype=torch.long, device=device)
        batch = TableTensorBatch(
            x_num=x_num_t, x_cat=x_cat_t,
            col_types=col_types_t, cat_cardinalities=cat_card_t, cat_decode_types=cat_decode_t,
        )
        rank_t = torch.as_tensor(step_rank[None], dtype=torch.long, device=device)

        if is_sparse:
            pred_mask_np = np.zeros((N, D), dtype=bool)
            pred_mask_np[coords[:, 0], coords[:, 1]] = True
            pred_mask_t = torch.as_tensor(pred_mask_np[None], dtype=torch.bool, device=device)
            out = model(batch, rank_t, context_row_mask=context_row_mask_t, prediction_mask=pred_mask_t)
        else:
            out = model(batch, rank_t, context_row_mask=context_row_mask_t)

        for r, c in coords:
            r, c = int(r), int(c)
            is_cat = col_types_np[c] == CATEGORICAL
            out_rows.append(r); out_cols.append(c); out_step.append(step_idx)
            out_is_cat.append(bool(is_cat))

            if is_cat:
                n_valid = int(cat_cardinalities_np[c])
                logits = out.cat_logits[0, r, c, :n_valid].float()
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                pred_cls = int(probs.argmax())
                x_cat_work[r, c] = pred_cls  # feed back -- own prediction, not ground truth

                proba_row = np.full(k_max, np.nan, dtype=np.float32)
                proba_row[:n_valid] = probs
                out_proba.append(proba_row)
                out_true.append(int(x_cat_true[r, c]))
                out_pred.append(pred_cls)
            else:
                pred_val_norm = float(out.num_mu[0, r, c].item())
                # num_mu is in the model's (roughly-standardized) training
                # scale -- map back to this column's real-world units before
                # using it for anything, feedback included (x_num_work must
                # stay in the same units as x_num_true for context stats on
                # later steps to mean anything).
                pred_val = pred_val_norm * float(_ctx_std[c]) + float(_ctx_mean[c])
                x_num_work[r, c] = pred_val  # feed back, real-world units
                out_proba.append(np.full(k_max, np.nan, dtype=np.float32))
                out_true.append(float(x_num_true[r, c]))
                out_pred.append(pred_val)

        rank[coords[:, 0], coords[:, 1]] = RANK_OBSERVED  # now revealed for subsequent steps

    return ARGenerationResult(
        rows=np.asarray(out_rows, dtype=np.int64),
        cols=np.asarray(out_cols, dtype=np.int64),
        step_index=np.asarray(out_step, dtype=np.int64),
        y_true=np.asarray(out_true, dtype=np.float64),
        y_pred=np.asarray(out_pred, dtype=np.float64),
        y_proba=np.stack(out_proba) if out_proba else None,
        is_categorical=np.asarray(out_is_cat, dtype=bool),
    )
