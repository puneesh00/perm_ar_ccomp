# scripts/train_synthetic.py

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Allow running without installing package:
#   python scripts/train_synthetic.py ...
# ---------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from tab_completion.sampling import (
    CompletionTask,
    TargetPredictionSampler,
    RandomCellSampler,
    ColumnBlockSampler,
    RowBlockSampler,
    LabelFeatureSampler,
    MixtureSampler,
)
from tab_completion.factorization import (
    ParallelFactorizer,
    PermARFactorizer,
    FactorizationPlan,
)
from tab_completion.synthetic_data import (
    FullSyntheticTable,
    make_synthetic_table,
    SyntheticTableGeneratorConfig,
    SyntheticTableGenerator,
)
from tab_completion.episode_utils import (
    task_to_torch_batch,
    mask_to_torch,
    make_step_query_mask,
)
from tab_completion.model import (
    ModelConfig,
    CellwiseCompletionModel,
    NUMERICAL,
)
from tab_completion.losses import typed_mse_ce_loss
from tab_completion.model_perm_ar import (
    PermARCompletionModel,
    compute_task_loss_onepass,
    compute_task_loss_onepass_batched,
)
from tab_completion.model_perm_ar_sparse import (
    PermARCompletionModel as PermARCompletionModelSparse,
    compute_task_loss_onepass as compute_task_loss_onepass_sparse,
    compute_task_loss_onepass_batched as compute_task_loss_onepass_batched_sparse,
)
from tab_completion.model_single_stream import (
    SingleStreamModel,
    compute_task_loss_single_stream,
    compute_task_loss_single_stream_batched,
)
from tab_completion.synthetic_data_tabpfn import (
    TabPFNSCMConfig,
    TabPFNSCMTableGenerator,
)


def autocast_ctx(args, device: torch.device):
    amp_torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp_dtype)
    if amp_torch_dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_torch_dtype)


def sparse_partial_predict_kwargs(args, np_rng: np.random.Generator) -> Dict:
    if args.architecture != "two_stream_ar_sparse" or args.max_predict_cells is None:
        return {}
    return {"max_predict_cells": args.max_predict_cells, "rng": np_rng}


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def build_sampler(args) -> object:
    if args.sampler == "target":
        return TargetPredictionSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            target_col=args.target_col,
        )

    if args.sampler == "random_cell":
        return RandomCellSampler(
            n_episode_rows=args.n_episode_rows,
            query_frac=args.query_frac,
            max_query_cells=args.max_query_cells,
        )

    if args.sampler == "column_block":
        return ColumnBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            min_query_cols=args.min_query_cols,
            max_query_cols=args.max_query_cols,
            exclude_target=args.exclude_target,
            conditioning_mode=args.column_block_conditioning_mode,
        )

    if args.sampler == "row_block":
        return RowBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            query_frac_cols=args.query_frac_cols,
            conditioning_mode=args.row_block_conditioning_mode,
        )

    if args.sampler == "label_feature":
        return LabelFeatureSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            n_feature_cols=args.n_feature_cols,
            target_col=args.target_col,
            conditioning_mode=args.label_feature_conditioning_mode,
        )

    if args.sampler == "mixture":
        return MixtureSampler(
            samplers=[
                TargetPredictionSampler(
                    n_context=args.n_context,
                    n_query=args.n_query,
                    target_col=args.target_col,
                ),
                RandomCellSampler(
                    n_episode_rows=args.n_episode_rows,
                    query_frac=args.query_frac,
                    max_query_cells=args.max_query_cells,
                ),
                ColumnBlockSampler(
                    n_context=args.n_context,
                    n_query=args.n_query,
                    min_query_cols=args.min_query_cols,
                    max_query_cols=args.max_query_cols,
                    exclude_target=args.exclude_target,
                    conditioning_mode=args.column_block_conditioning_mode,
                ),
                LabelFeatureSampler(
                    n_context=args.n_context,
                    n_query=args.n_query,
                    n_feature_cols=args.n_feature_cols,
                    target_col=args.target_col,
                    conditioning_mode=args.label_feature_conditioning_mode,
                ),
            ],
            weights=[
                args.mix_target,
                args.mix_random_cell,
                args.mix_column_block,
                args.mix_label_feature,
            ],
        )

    raise ValueError(f"Unknown sampler: {args.sampler}")


def build_factorizer(args) -> object:
    if args.factorization == "parallel":
        return ParallelFactorizer()

    if args.factorization == "perm_ar":
        return PermARFactorizer(
            unit=args.ar_unit,
            group_size=args.group_size,
        )

    raise ValueError(f"Unknown factorization: {args.factorization}")


# ---------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------


@dataclass
class StepLossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


def get_context_row_mask_from_task(task: CompletionTask) -> Optional[np.ndarray]:
    """
    Return context-row mask for inductive-row tasks, else None.

    conditioning_mode is set by the sampler:
      - transductive: all observed cells in the episode may be evidence.
      - inductive_rows: query rows may use context rows plus themselves only.

    This must come from task metadata, not from the current AR step, because
    random-cell masking can query cells in many rows but should remain
    transductive.
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


def compute_task_loss(
    model: nn.Module,
    full: FullSyntheticTable,
    task: CompletionTask,
    plan: FactorizationPlan,
    device: torch.device,
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
) -> StepLossOutput:
    """
    Computes loss for either parallel or permutation-AR factorization.

    Parallel:
        one step containing all query cells.

    Perm-AR:
        multiple steps. After each step, the true values from that step are
        revealed through teacher forcing by updating observed_mask.

    In v0, query_mask at each forward pass contains only the current step.
    """
    batch = task_to_torch_batch(full, task, device)

    observed_np = task.observed_mask.copy()
    task_shape = task.observed_mask.shape

    context_row_mask_np = get_context_row_mask_from_task(task)
    if context_row_mask_np is None:
        context_row_mask_t = None
    else:
        context_row_mask_t = torch.as_tensor(
            context_row_mask_np[None, :],
            dtype=torch.bool,
            device=device,
        )

    step_losses: list[torch.Tensor] = []
    metrics_accum: Dict[str, float] = {}
    n_metric_steps = 0

    for step_coords in plan.steps:
        if len(step_coords) == 0:
            continue

        step_query_np = make_step_query_mask(task_shape, step_coords)

        observed_t = mask_to_torch(observed_np, device)
        query_t = mask_to_torch(step_query_np, device)

        out = model(
            batch,
            observed_t,
            query_t,
            context_row_mask=context_row_mask_t,
        )

        loss_out = typed_mse_ce_loss(
            out,
            batch,
            query_t,
            num_weight=num_weight,
            cat_weight=cat_weight,
        )

        step_losses.append(loss_out.loss)

        for key, value in loss_out.metrics.items():
            metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)

        n_metric_steps += 1

        # Teacher forcing reveal for later AR steps.
        observed_np[step_coords[:, 0], step_coords[:, 1]] = True

    if not step_losses:
        raise RuntimeError("No losses computed. Empty factorization plan?")

    loss = torch.stack(step_losses).mean()

    metrics = {
        key: value / max(n_metric_steps, 1)
        for key, value in metrics_accum.items()
    }
    metrics["loss"] = float(loss.detach().cpu())
    metrics["factorization_steps"] = float(plan.num_steps)
    metrics["query_cells"] = float(task.num_query_cells)

    return StepLossOutput(loss=loss, metrics=metrics)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def _eval_override(args, name: str, fallback):
    """Return eval-specific override if provided, otherwise training/default value."""
    value = getattr(args, name)
    return fallback if value is None else value


def _parse_eval_sampler_names(eval_samplers: str) -> list[str]:
    """Parse comma-separated eval sampler names."""
    if eval_samplers == "all":
        return ["target", "random_cell", "column_block", "row_block", "label_feature"]

    names = [x.strip() for x in eval_samplers.split(",") if x.strip()]
    allowed = {"target", "random_cell", "column_block", "row_block", "label_feature"}
    unknown = [x for x in names if x not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown eval samplers: {unknown}. Allowed values are {sorted(allowed)} or 'all'."
        )
    if not names:
        raise ValueError("--eval-samplers produced an empty list.")
    return names


def build_eval_samplers(args) -> Dict[str, object]:
    """
    Build evaluation samplers with eval-specific task-shape overrides.

    Training args such as --n-query are global, but we often want different
    evaluation shapes, e.g. target prediction with many query rows and row
    completion with one partial query row. The --eval-* arguments override
    the corresponding training arguments only for evaluation.
    """
    target_n_context = _eval_override(args, "eval_target_n_context", args.n_context)
    target_n_query = _eval_override(args, "eval_target_n_query", args.n_query)

    random_n_episode_rows = _eval_override(
        args, "eval_random_n_episode_rows", args.n_episode_rows
    )
    random_query_frac = _eval_override(args, "eval_random_query_frac", args.query_frac)
    random_max_query_cells = _eval_override(
        args, "eval_random_max_query_cells", args.max_query_cells
    )

    column_n_context = _eval_override(args, "eval_column_n_context", args.n_context)
    column_n_query = _eval_override(args, "eval_column_n_query", args.n_query)
    column_min_query_cols = _eval_override(
        args, "eval_column_min_query_cols", args.min_query_cols
    )
    column_max_query_cols = _eval_override(
        args, "eval_column_max_query_cols", args.max_query_cols
    )
    column_exclude_target = _eval_override(
        args, "eval_column_exclude_target", args.exclude_target
    )
    column_conditioning_mode = _eval_override(
        args, "eval_column_conditioning_mode", args.column_block_conditioning_mode
    )

    row_n_context = _eval_override(args, "eval_row_n_context", args.n_context)
    row_n_query = _eval_override(args, "eval_row_n_query", args.n_query)
    row_query_frac_cols = _eval_override(
        args, "eval_row_query_frac_cols", args.query_frac_cols
    )
    row_conditioning_mode = _eval_override(
        args, "eval_row_conditioning_mode", args.row_block_conditioning_mode
    )

    label_feature_n_context = _eval_override(
        args, "eval_label_feature_n_context", args.n_context
    )
    label_feature_n_query = _eval_override(
        args, "eval_label_feature_n_query", args.n_query
    )
    label_feature_n_feature_cols = _eval_override(
        args, "eval_label_feature_n_feature_cols", args.n_feature_cols
    )
    label_feature_conditioning_mode = _eval_override(
        args, "eval_label_feature_conditioning_mode",
        args.label_feature_conditioning_mode,
    )

    registry = {
        "target": TargetPredictionSampler(
            n_context=target_n_context,
            n_query=target_n_query,
            target_col=args.target_col,
        ),
        "random_cell": RandomCellSampler(
            n_episode_rows=random_n_episode_rows,
            query_frac=random_query_frac,
            max_query_cells=random_max_query_cells,
        ),
        "column_block": ColumnBlockSampler(
            n_context=column_n_context,
            n_query=column_n_query,
            min_query_cols=column_min_query_cols,
            max_query_cols=column_max_query_cols,
            exclude_target=column_exclude_target,
            conditioning_mode=column_conditioning_mode,
        ),
        "row_block": RowBlockSampler(
            n_context=row_n_context,
            n_query=row_n_query,
            query_frac_cols=row_query_frac_cols,
            conditioning_mode=row_conditioning_mode,
        ),
        "label_feature": LabelFeatureSampler(
            n_context=label_feature_n_context,
            n_query=label_feature_n_query,
            n_feature_cols=label_feature_n_feature_cols,
            target_col=args.target_col,
            conditioning_mode=label_feature_conditioning_mode,
        ),
    }

    names = _parse_eval_sampler_names(args.eval_samplers)
    return {name: registry[name] for name in names}


def _context_query_features(
    full: FullSyntheticTable,
    task: CompletionTask,
) -> Optional[tuple]:
    """
    Shared feature/label extraction for the `target`-sampler context
    baselines below: builds one-hot(categorical) + raw(numerical) features
    from this episode's context/query rows (same rows/columns the model
    sees). Returns (X_ctx, y_ctx, X_qry, y_qry) or None if the task doesn't
    carry the metadata these baselines need (e.g. a non-target sampler).
    """
    try:
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        return None

    target_col = task.meta.get("target_col")
    context_rows_local = task.meta.get("context_rows_local")
    query_rows_local = task.meta.get("query_rows_local")
    if target_col is None or context_rows_local is None or query_rows_local is None:
        return None

    global_rows = task.row_idx
    feature_cols = [c for c in task.col_idx.tolist() if c != target_col]

    parts = []
    for j in feature_cols:
        if full.col_types[j] == NUMERICAL:
            parts.append(full.x_num[global_rows, j][:, None])
        else:
            card = int(full.cat_cardinalities[j])
            enc = OneHotEncoder(sparse_output=False, categories=[list(range(card))])
            parts.append(enc.fit_transform(full.x_cat[global_rows, j][:, None]))
    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(global_rows), 0))
    y = full.x_cat[global_rows, target_col]

    return (
        X[context_rows_local], y[context_rows_local],
        X[query_rows_local], y[query_rows_local],
    )


def logreg_context_baseline_acc(
    full: FullSyntheticTable,
    task: CompletionTask,
) -> Optional[float]:
    """
    Fits a fresh sklearn LogisticRegression on this episode's context rows
    and scores it on the query rows. Gives a finite-context oracle baseline
    for the `target` sampler, so transformer accuracy can be judged against
    "best a correctly-specified linear model can do with the same amount of
    context" rather than an arbitrary number.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    features = _context_query_features(full, task)
    if features is None:
        return None
    X_ctx, y_ctx, X_qry, y_qry = features

    if len(np.unique(y_ctx)) < 2:
        pred = np.full_like(y_qry, y_ctx[0])
    else:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_ctx, y_ctx)
        pred = clf.predict(X_qry)

    return float((pred == y_qry).mean())


def rf_context_baseline_acc(
    full: FullSyntheticTable,
    task: CompletionTask,
) -> Optional[float]:
    """
    Same finite-context-oracle idea as logreg_context_baseline_acc, but a
    RandomForestClassifier: a nonlinear, non-in-context baseline that (unlike
    logreg) can pick up threshold/interaction structure, which the SCM prior
    can easily produce. Same feature encoding as the logreg baseline, so the
    two numbers are directly comparable.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return None

    features = _context_query_features(full, task)
    if features is None:
        return None
    X_ctx, y_ctx, X_qry, y_qry = features

    if len(np.unique(y_ctx)) < 2:
        pred = np.full_like(y_qry, y_ctx[0])
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0)
        clf.fit(X_ctx, y_ctx)
        pred = clf.predict(X_qry)

    return float((pred == y_qry).mean())


def xgb_context_baseline_acc(
    full: FullSyntheticTable,
    task: CompletionTask,
) -> Optional[float]:
    """
    Same finite-context-oracle idea as logreg_context_baseline_acc, but
    gradient-boosted trees (XGBoost) -- the standard strong non-in-context
    tabular baseline, and typically the closest competitor to TabPFN-style
    models in the literature. Same feature encoding as the logreg baseline.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    features = _context_query_features(full, task)
    if features is None:
        return None
    X_ctx, y_ctx, X_qry, y_qry = features

    n_classes = int(np.unique(y_ctx).size)
    if n_classes < 2:
        pred = np.full_like(y_qry, y_ctx[0])
    else:
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            verbosity=0,
            random_state=0,
        )
        clf.fit(X_ctx, y_ctx)
        pred = clf.predict(X_qry)

    return float((pred == y_qry).mean())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    args,
    device: torch.device,
    full_fixed: Optional[FullSyntheticTable] = None,
) -> Dict[str, float]:
    model.eval()

    # Reset eval RNG each time so eval tasks are fixed across checkpoints/runs,
    # assuming args.eval_seed is fixed.
    eval_rng = np.random.default_rng(args.eval_seed)
    eval_samplers = build_eval_samplers(args)
    factorizer = build_factorizer(args)

    eval_table_generator = None
    if args.data_mode == "fresh_table":
        if args.data_prior == "tabpfn":
            eval_table_generator = TabPFNSCMTableGenerator(
                TabPFNSCMConfig(
                    n_rows=args.fresh_n_rows,
                    n_cols=args.n_cols,
                    p_categorical=args.p_categorical,
                    k_max=args.k_max,
                    n_classes=args.n_classes,
                    target_col=args.target_col,
                    base_seed=args.eval_seed,
                    prior_type=args.tabpfn_prior_type,
                    layers_mu_max=args.tabpfn_layers_mu_max,
                    layers_max=args.tabpfn_layers_max,
                    hidden_mu_max=args.tabpfn_hidden_mu_max,
                )
            )
        else:
            eval_table_generator = SyntheticTableGenerator(
                SyntheticTableGeneratorConfig(
                    n_rows=args.fresh_n_rows,
                    n_cols=args.n_cols,
                    p_categorical=args.p_categorical,
                    k_max=args.k_max,
                    n_classes=args.n_classes,
                    target_col=args.target_col,
                    latent_dim=args.latent_dim,
                    noise=args.data_noise,
                    base_seed=args.eval_seed,
                )
            )

    all_metrics: Dict[str, float] = {}

    for task_name, sampler in eval_samplers.items():
        values_by_metric: Dict[str, list[float]] = {}

        for _ in range(args.eval_tasks):
            if args.data_mode == "fixed_table":
                assert full_fixed is not None
                full = full_fixed
            else:
                assert eval_table_generator is not None
                full = eval_table_generator.sample_table()

            info = full.table_info()

            task = sampler.sample(info, eval_rng)
            plan = factorizer.build(task, eval_rng)

            if args.architecture == "two_stream_ar":
                loss_fn = compute_task_loss_onepass
            elif args.architecture == "two_stream_ar_sparse":
                loss_fn = compute_task_loss_onepass_sparse
            elif args.architecture == "single_stream":
                loss_fn = compute_task_loss_single_stream
            else:
                loss_fn = compute_task_loss
            with autocast_ctx(args, device):
                loss_out = loss_fn(
                    model=model,
                    full=full,
                    task=task,
                    plan=plan,
                    device=device,
                    num_weight=args.num_weight,
                    cat_weight=args.cat_weight,
                )

            for key, value in loss_out.metrics.items():
                values_by_metric.setdefault(key, []).append(float(value))

            if task_name == "target":
                baseline_acc = logreg_context_baseline_acc(full, task)
                if baseline_acc is not None:
                    values_by_metric.setdefault("logreg128_acc", []).append(baseline_acc)
                rf_acc = rf_context_baseline_acc(full, task)
                if rf_acc is not None:
                    values_by_metric.setdefault("rf128_acc", []).append(rf_acc)
                xgb_acc = xgb_context_baseline_acc(full, task)
                if xgb_acc is not None:
                    values_by_metric.setdefault("xgb128_acc", []).append(xgb_acc)

        # cat_acc/loss_cat and num_mse/loss_num are set to a sentinel 0.0 on
        # episodes with zero cells of that type (see typed_mse_ce_loss). A
        # plain np.mean over episodes silently averages those sentinel zeros
        # in as if they were real accuracy/loss values, which understates
        # samplers (e.g. column_block) where many episodes land on a column
        # of the other type. Weight by the matching cell count instead, so
        # zero-cell episodes get zero weight rather than counting as a 0.
        cell_weighted = {
            "cat_acc": "cat_cells",
            "loss_cat": "cat_cells",
            "num_mse": "num_cells",
            "loss_num": "num_cells",
        }

        for key, values in values_by_metric.items():
            weight_key = cell_weighted.get(key)
            weights = values_by_metric.get(weight_key) if weight_key else None

            if weights is not None:
                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    all_metrics[f"eval/{task_name}/{key}"] = weighted_sum / total_weight
                else:
                    all_metrics[f"eval/{task_name}/{key}"] = 0.0
            else:
                all_metrics[f"eval/{task_name}/{key}"] = float(np.mean(values))

    model.train()
    return all_metrics


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


class JSONLLogger:
    """
    Simple local logger. Robust to changing keys across train/eval rows.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def maybe_init_wandb(args):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb requested but not installed. Continuing without wandb.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
    )
    return run


def log_metrics(
    logger: JSONLLogger,
    wandb_run,
    row: Dict,
) -> None:
    logger.log(row)

    if wandb_run is not None:
        import wandb

        step = int(row.get("step", 0))
        wandb.log(row, step=step)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--n-cols", type=int, default=32)
    parser.add_argument("--p-categorical", type=float, default=0.3)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--target-col", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--data-noise", type=float, default=0.1)
    parser.add_argument(
        "--data-mode",
        type=str,
        default="fixed_table",
        choices=["fixed_table", "fresh_table"],
        help=(
            "fixed_table: generate one synthetic table and reuse it. "
            "fresh_table: sample a fresh synthetic table per task."
        ),
    )
    parser.add_argument(
        "--data-prior",
        type=str,
        default="simple",
        choices=["simple", "tabpfn"],
        help=(
            "simple: the original latent-factor generator (synthetic_data.py). "
            "tabpfn: the SCM/BNN-mixture prior (synthetic_data_tabpfn.py), "
            "richer and with a real accuracy ceiling by construction -- see "
            "that module's docstring. Only wired up for --data-mode fresh_table."
        ),
    )
    parser.add_argument(
        "--tabpfn-prior-type", type=str, default="mixed",
        choices=["scm", "bnn", "mixed"],
        help="TabPFNSCMConfig.prior_type (default mixed = 50/50 SCM/BNN, "
             "per the paper). Set scm or bnn to use only that branch.",
    )
    parser.add_argument(
        "--tabpfn-layers-mu-max", type=float, default=6.0,
        help="TabPFNSCMConfig.layers_mu_max (default 6.0). Lower = shallower "
             "sampled graphs -- the TNLU mean-depth ceiling, not a hard cap.",
    )
    parser.add_argument(
        "--tabpfn-layers-max", type=int, default=None,
        help="TabPFNSCMConfig.layers_max: hard ceiling on sampled depth. "
             "layers_mu_max alone can't guarantee a max (unbounded-above "
             "truncated normal); set this when you need a real cap.",
    )
    parser.add_argument(
        "--tabpfn-hidden-mu-max", type=float, default=130.0,
        help="TabPFNSCMConfig.hidden_mu_max (default 130.0). Lower = narrower "
             "sampled graphs.",
    )
    parser.add_argument(
        "--fresh-n-rows",
        type=int,
        default=256,
        help="Rows per freshly sampled table when --data-mode=fresh_table.",
    )

    # Sampler
    parser.add_argument(
        "--sampler",
        type=str,
        default="mixture",
        choices=[
            "target",
            "random_cell",
            "column_block",
            "row_block",
            "label_feature",
            "mixture",
        ],
    )
    parser.add_argument("--n-context", type=int, default=128)
    parser.add_argument("--n-query", type=int, default=128)
    parser.add_argument("--n-episode-rows", type=int, default=256)
    parser.add_argument("--query-frac", type=float, default=0.15)
    parser.add_argument("--max-query-cells", type=int, default=2048)
    parser.add_argument("--query-frac-cols", type=float, default=1.0)
    parser.add_argument("--min-query-cols", type=int, default=1)
    parser.add_argument("--max-query-cols", type=int, default=3)
    parser.add_argument("--n-feature-cols", type=int, default=2)
    parser.add_argument("--exclude-target", action="store_true")
    parser.add_argument(
        "--column-block-conditioning-mode",
        type=str,
        default="inductive_rows",
        choices=["inductive_rows", "transductive"],
        help=(
            "conditioning semantics for column_block tasks. "
            "inductive_rows means query rows cannot use other query rows; "
            "transductive means all observed cells are evidence."
        ),
    )
    parser.add_argument(
        "--row-block-conditioning-mode",
        type=str,
        default="inductive_rows",
        choices=["inductive_rows", "transductive"],
    )
    parser.add_argument(
        "--label-feature-conditioning-mode",
        type=str,
        default="inductive_rows",
        choices=["inductive_rows", "transductive"],
    )

    # Mixture weights
    parser.add_argument("--mix-target", type=float, default=0.25)
    parser.add_argument("--mix-random-cell", type=float, default=0.25)
    parser.add_argument("--mix-column-block", type=float, default=0.25)
    parser.add_argument("--mix-label-feature", type=float, default=0.25)

    # Factorization
    parser.add_argument(
        "--factorization",
        type=str,
        default="parallel",
        choices=["parallel", "perm_ar"],
    )
    parser.add_argument(
        "--ar-unit",
        type=str,
        default="column",
        choices=["cell", "column", "row"],
    )
    parser.add_argument("--group-size", type=int, default=1)

    # Model
    parser.add_argument(
        "--architecture",
        type=str,
        default="one_stream",
        choices=["one_stream", "two_stream_ar", "two_stream_ar_sparse", "single_stream"],
        help=(
            "one_stream = original CellwiseCompletionModel: row-encoder + "
            "mean-pooled row-context/column aggregation, AR factorization "
            "trained via a sequential loop (one forward+backward pass per "
            "AR step -- correct but O(num_steps), impractical for "
            "cell-wise AR on tasks with many query cells (e.g. random_cell). "
            "two_stream_ar = PermARCompletionModel (src/tab_completion/"
            "model_perm_ar.py): axial two-stream attention driven by a "
            "per-cell reveal-rank tensor, scores every query cell of an "
            "episode in a single forward pass regardless of factorization "
            "or unit. Not numerically equivalent to one_stream even at "
            "parallel factorization -- it's a different architecture, not "
            "just a faster implementation of the same one. "
            "single_stream = SingleStreamModel (src/tab_completion/"
            "model_single_stream.py): TabPFN-v1-style one-token-per-cell "
            "axial attention with context/query row masking, built from our "
            "own value encoders/decode heads plus the informative "
            "context-derived query placeholder. Parallel-factorization only "
            "-- no reveal-order support, see that file's module docstring "
            "for why. --batched-forward is required (no per-task fallback "
            "path is implemented for this architecture)."
        ),
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-episode-rows", type=int, default=512)
    parser.add_argument("--max-cols", type=int, default=128)
    parser.add_argument("--num-cat-decode-types", type=int, default=128)
    # 8/8 (equal) as of 2026-08-09: matches --tabpfn-style-layers now
    # defaulting on, which requires num_row_layers == num_row_context_layers.
    # Also the depth this codebase's SCM-prior investigation settled on --
    # matches tabpfn_v1_ref's attention depth (16 ops via 8 paired blocks).
    parser.add_argument("--num-row-layers", type=int, default=8)
    parser.add_argument("--num-row-context-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # The six flags below default ON as of 2026-08-09 -- this is the
    # architecture the SCM-prior investigation settled on (see the run
    # log/report). Each still accepts an explicit --no-<flag> to opt back
    # out for a one-off ablation.
    parser.add_argument(
        "--context-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="single_stream/two_stream_ar only: re-standardize x_num (and, "
             "if --unified-cat-encoding is also set, the cast-to-float "
             "categorical values too) per episode using observed(context)-"
             "only mean/std (mirrors TabPFNV1Model's FeatureEncoder). "
             "Ignored by the original architecture. Default on -- pass "
             "--no-context-normalize to opt out.",
    )
    parser.add_argument(
        "--unified-cat-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="single_stream/two_stream_ar only: drop cat_value_emb and "
             "encode categorical cells by casting their raw id to float "
             "through the same num_value_mlp as numerics (TabPFNV1Model's "
             "actual recipe). Ignored by the original architecture. "
             "Default on -- pass --no-unified-cat-encoding to opt out.",
    )
    parser.add_argument(
        "--shared-cat-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one shared [k_max, d] categorical decode matrix instead "
             "of one private slice per column. Cardinality masking is "
             "unaffected either way. Default on -- pass "
             "--no-shared-cat-decoder to opt out. NOTE: also affects the "
             "original one_stream architecture (shared TypedCategoricalHead) "
             "-- that architecture was never part of the ablation that "
             "validated this default.",
    )
    parser.add_argument(
        "--tabpfn-style-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="single_stream/two_stream_ar only: row-attn, col-attn, THEN "
             "one FFN application per paired block (TabPFNV1Layer's "
             "layout) instead of an FFN after every single-axis attention. "
             "Requires --num-row-layers == --num-row-context-layers (both "
             "default to 8). Default on -- pass --no-tabpfn-style-layers "
             "to opt out (remember to also set unequal layer counts back "
             "if you want the old alternating-axis default shape).",
    )
    parser.add_argument(
        "--share-stream-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="two_stream_ar only: share attention (and FFN) weights "
             "between the content and query streams, XLNet-style, instead "
             "of separate content/query parameter sets. Same activations "
             "(attention still runs twice), half the attention+FFN params. "
             "Harmless/inert for single_stream and one_stream. Default on "
             "-- pass --no-share-stream-attn to opt out.",
    )
    parser.add_argument(
        "--drop-type-origin-emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="single_stream/two_stream_ar only: drop type_emb (num vs cat) "
             "and origin_emb (observed vs query-this-episode) entirely -- "
             "no additive signal beyond the value encoding itself, matching "
             "TabPFNV1's lack of any such flag. Default on -- pass "
             "--no-drop-type-origin-emb to opt out.",
    )
    parser.add_argument(
        "--post-ln",
        action="store_true",
        help="single_stream only: post-LN (attn/ffn on raw input, residual "
             "add, THEN LayerNorm) instead of this codebase's default "
             "pre-LN -- matches TabPFNV1Layer's actual layout exactly.",
    )
    parser.add_argument(
        "--global-query-bridge",
        action="store_true",
        help="two_stream_ar_sparse (tabpfn_style_layers only) only: after "
             "row+col axis attention each layer, each sparse query also "
             "attends directly over the full flattened content grid for its "
             "episode (masked to rank(content) < rank(query)), fixing a real "
             "gap where an off-row/off-col earlier-revealed target under "
             "perm_ar has no path to a later query (axial attention alone "
             "can't bridge it -- verified via perturbation test). Moot for "
             "factorization=parallel or single-column target prediction. "
             "Adds an O(Q*N*D)-per-layer attention pass. Default off.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="two_stream_ar_sparse only: checkpoint each paired row+col+FFN "
             "block (rerun its forward during backward instead of retaining "
             "internal activations). Trades ~30-40%% more wall-clock for a "
             "large memory cut across many stacked layers. Pure memory/"
             "compute tradeoff -- does not change the forward computation. "
             "Default off.",
    )

    # Train
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument(
        "--batch-tasks",
        type=int,
        default=1,
        help=(
            "Number of sampled CompletionTasks per optimizer step. "
            "They are looped over separately, so they may have different shapes."
        ),
    )
    parser.add_argument(
        "--batched-forward",
        action="store_true",
        help=(
            "Stack the --batch-tasks sampled episodes into one real [B, N, D] "
            "tensor and call the model once, instead of looping the model "
            "once per task at B=1. Same gradient math (both average B "
            "independently-sampled episode losses before backward), much "
            "better GPU utilization. Only valid for --architecture "
            "two_stream_ar with a sampler/factorization where every episode "
            "has the same (N, D) shape -- true for TargetPredictionSampler + "
            "ParallelFactorizer (every run this flag has been used for), NOT "
            "true for samplers with a per-episode-variable query-cell/column "
            "count (RandomCellSampler, ColumnBlockSampler) or perm-AR plans "
            "with a per-episode-variable step count. Raises if the shapes "
            "in a batch don't actually match, rather than silently falling "
            "back.",
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Linear LR warmup from 0 to --lr over this many steps, then "
             "cosine decay to --lr * --lr-min-ratio over the rest of "
             "--steps (same recipe as the TabPFN paper: 'linear-warmup and "
             "cosine annealing'). 0 disables the schedule -- flat --lr the "
             "whole run, the original behavior.",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.1,
        help="Cosine decay floor, as a fraction of --lr. Only used when "
             "--warmup-steps > 0.",
    )
    parser.add_argument("--num-weight", type=float, default=1.0)
    parser.add_argument("--cat-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # Eval/log
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument(
        "--checkpoint-every", type=int, default=10_000,
        help="Save a model checkpoint every N steps, independent of "
             "--eval-every -- so a long run always has durable, coarse "
             "recovery points even if eval is frequent (many small "
             "eval-triggered checkpoints) or the run is stopped early "
             "between eval checkpoints. Set <=0 to disable.",
    )
    parser.add_argument(
        "--eval-samplers",
        type=str,
        default="all",
        help=(
            "Comma-separated eval sampler names: "
            "target,random_cell,column_block,row_block,label_feature. "
            "Use 'all' to evaluate every task family."
        ),
    )

    # Eval-specific target-prediction overrides.
    parser.add_argument("--eval-target-n-context", type=int, default=None)
    parser.add_argument("--eval-target-n-query", type=int, default=None)

    # Eval-specific random-cell overrides.
    parser.add_argument("--eval-random-n-episode-rows", type=int, default=None)
    parser.add_argument("--eval-random-query-frac", type=float, default=None)
    parser.add_argument("--eval-random-max-query-cells", type=int, default=None)

    # Eval-specific column-completion overrides.
    parser.add_argument("--eval-column-n-context", type=int, default=None)
    parser.add_argument("--eval-column-n-query", type=int, default=None)
    parser.add_argument("--eval-column-min-query-cols", type=int, default=None)
    parser.add_argument("--eval-column-max-query-cols", type=int, default=None)
    parser.add_argument(
        "--eval-column-exclude-target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override exclude_target only for column_block evaluation. "
            "Use --eval-column-exclude-target or --no-eval-column-exclude-target."
        ),
    )
    parser.add_argument(
        "--eval-column-conditioning-mode",
        type=str,
        default=None,
        choices=["inductive_rows", "transductive"],
    )

    # Eval-specific row-completion overrides.
    parser.add_argument("--eval-row-n-context", type=int, default=None)
    parser.add_argument("--eval-row-n-query", type=int, default=None)
    parser.add_argument("--eval-row-query-frac-cols", type=float, default=None)
    parser.add_argument(
        "--eval-row-conditioning-mode",
        type=str,
        default=None,
        choices=["inductive_rows", "transductive"],
    )

    # Eval-specific label+feature overrides.
    parser.add_argument("--eval-label-feature-n-context", type=int, default=None)
    parser.add_argument("--eval-label-feature-n-query", type=int, default=None)
    parser.add_argument("--eval-label-feature-n-feature-cols", type=int, default=None)
    parser.add_argument(
        "--eval-label-feature-conditioning-mode",
        type=str,
        default=None,
        choices=["inductive_rows", "transductive"],
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="results/synthetic_v0")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tab_completion")

    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-predict-cells", type=int, default=None,
        help="two_stream_ar_sparse only: XLNet-style partial prediction. "
             "If set, subsample at most this many query cells per episode "
             "(uniformly at random) and only compute g-states/loss for "
             "those, instead of every cell in task.query_mask. Unbiased "
             "since typed_mse_ce_loss already mean-reduces over queried "
             "cells. Ignored by other architectures.",
    )
    parser.add_argument(
        "--amp-dtype", type=str, default="none", choices=["none", "bf16", "fp16"],
        help="Run the forward pass (model + loss) under torch.autocast in this "
             "dtype. bf16 needs no GradScaler (same dynamic range as fp32); "
             "fp16 is provided for comparison but has no GradScaler wired up "
             "here, so it may underflow -- prefer bf16.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_col is None:
        args.target_col = args.n_cols - 1

    if args.run_name is None:
        args.run_name = f"{args.sampler}_{args.factorization}_{int(time.time())}"

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    logger = JSONLLogger(out_dir / "metrics.jsonl")
    wandb_run = maybe_init_wandb(args)

    print("=== Building synthetic data source ===")

    if args.data_mode == "fixed_table":
        if args.data_prior == "tabpfn":
            raise NotImplementedError(
                "--data-prior tabpfn is only wired up for --data-mode fresh_table."
            )
        full_fixed = make_synthetic_table(
            n_rows=args.n_rows,
            n_cols=args.n_cols,
            p_categorical=args.p_categorical,
            k_max=args.k_max,
            n_classes=args.n_classes,
            seed=args.data_seed,
            target_col=args.target_col,
            latent_dim=args.latent_dim,
            noise=args.data_noise,
        )
        table_generator = None

        info = full_fixed.table_info()
        print("data_mode=fixed_table")
        print(f"n_rows={info.n_rows}, n_cols={info.n_cols}, target_col={info.target_col}")
        print("col_types:", full_fixed.col_types.tolist())
        print("cat_cardinalities:", full_fixed.cat_cardinalities.tolist())

    else:
        full_fixed = None

        if args.data_prior == "tabpfn":
            table_generator = TabPFNSCMTableGenerator(
                TabPFNSCMConfig(
                    n_rows=args.fresh_n_rows,
                    n_cols=args.n_cols,
                    p_categorical=args.p_categorical,
                    k_max=args.k_max,
                    n_classes=args.n_classes,
                    target_col=args.target_col,
                    base_seed=args.data_seed,
                    prior_type=args.tabpfn_prior_type,
                    layers_mu_max=args.tabpfn_layers_mu_max,
                    layers_max=args.tabpfn_layers_max,
                    hidden_mu_max=args.tabpfn_hidden_mu_max,
                )
            )
        else:
            table_generator = SyntheticTableGenerator(
                SyntheticTableGeneratorConfig(
                    n_rows=args.fresh_n_rows,
                    n_cols=args.n_cols,
                    p_categorical=args.p_categorical,
                    k_max=args.k_max,
                    n_classes=args.n_classes,
                    target_col=args.target_col,
                    latent_dim=args.latent_dim,
                    noise=args.data_noise,
                    base_seed=args.data_seed,
                )
            )

        print(f"data_mode=fresh_table data_prior={args.data_prior}")
        print(
            f"fresh table config: n_rows={args.fresh_n_rows}, "
            f"n_cols={args.n_cols}, target_col={args.target_col}"
        )

    print("=== Building sampler / factorizer / model ===")
    sampler = build_sampler(args)
    factorizer = build_factorizer(args)

    model_cfg = ModelConfig(
        d_model=args.d_model,
        max_episode_rows=args.max_episode_rows,
        max_cols=args.max_cols,
        k_max=args.k_max,
        num_cat_decode_types=args.num_cat_decode_types,
        num_row_layers=args.num_row_layers,
        num_row_context_layers=args.num_row_context_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        context_normalize=args.context_normalize,
        unified_cat_encoding=args.unified_cat_encoding,
        shared_cat_decoder=args.shared_cat_decoder,
        tabpfn_style_layers=args.tabpfn_style_layers,
        share_stream_attn=args.share_stream_attn,
        drop_type_origin_emb=args.drop_type_origin_emb,
        post_ln=args.post_ln,
        global_query_bridge=args.global_query_bridge,
        activation_checkpointing=args.activation_checkpointing,
    )

    if args.architecture == "single_stream" and not args.batched_forward:
        raise ValueError(
            "--architecture single_stream requires --batched-forward -- no "
            "per-task fallback path is implemented for this architecture."
        )
    if args.batched_forward and args.architecture not in (
        "two_stream_ar", "two_stream_ar_sparse", "single_stream",
    ):
        raise ValueError(
            "--batched-forward requires --architecture two_stream_ar, "
            "two_stream_ar_sparse, or single_stream."
        )

    if args.architecture == "two_stream_ar":
        model = PermARCompletionModel(model_cfg).to(device)
        loss_fn = compute_task_loss_onepass
    elif args.architecture == "two_stream_ar_sparse":
        model = PermARCompletionModelSparse(model_cfg).to(device)
        loss_fn = compute_task_loss_onepass_sparse
    elif args.architecture == "single_stream":
        model = SingleStreamModel(model_cfg).to(device)
        loss_fn = compute_task_loss_single_stream
    else:
        model = CellwiseCompletionModel(model_cfg).to(device)
        loss_fn = compute_task_loss

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
        total_steps = args.steps
        min_ratio = args.lr_min_ratio

        def lr_lambda(step: int) -> float:
            # step is 0-indexed by LambdaLR (called with last_epoch, starting
            # at 0 before the first optimizer.step()).
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"lr_schedule=warmup({warmup_steps})+cosine(floor={min_ratio}*lr)")
    else:
        print("lr_schedule=none (flat lr)")

    n_params = sum(p.numel() for p in model.parameters())

    print(f"architecture={args.architecture}")
    print(f"device={device}")
    print(f"params={n_params:,}")
    print(f"run_dir={out_dir}")

    # Save model config separately for convenience.
    with (out_dir / "model_config.json").open("w") as f:
        json.dump(model_cfg.__dict__, f, indent=2)

    model.train()

    start_time = time.time()
    last_log_time = time.time()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        task_losses: list[torch.Tensor] = []
        task_metric_rows: list[Dict[str, float]] = []
        task_names: list[str] = []
        plan_modes: list[str] = []

        if args.batched_forward:
            # Sample all B episodes first, then one real [B, N, D] forward
            # pass instead of B sequential B=1 passes -- see
            # compute_task_loss_onepass_batched's docstring for why this is
            # equivalent gradient math, just much better GPU utilization.
            full_batch = []
            task_batch = []
            plan_batch = []
            for _ in range(args.batch_tasks):
                if args.data_mode == "fixed_table":
                    assert full_fixed is not None
                    full = full_fixed
                else:
                    assert table_generator is not None
                    full = table_generator.sample_table()

                info = full.table_info()
                task = sampler.sample(info, np_rng)
                plan = factorizer.build(task, np_rng)

                full_batch.append(full)
                task_batch.append(task)
                plan_batch.append(plan)

            if args.architecture == "single_stream":
                batched_loss_fn = compute_task_loss_single_stream_batched
            elif args.architecture == "two_stream_ar_sparse":
                batched_loss_fn = compute_task_loss_onepass_batched_sparse
            else:
                batched_loss_fn = compute_task_loss_onepass_batched
            with autocast_ctx(args, device):
                loss_out = batched_loss_fn(
                    model=model,
                    full_list=full_batch,
                    task_list=task_batch,
                    plan_list=plan_batch,
                    device=device,
                    num_weight=args.num_weight,
                    cat_weight=args.cat_weight,
                    **sparse_partial_predict_kwargs(args, np_rng),
                )

            task_losses = [loss_out.loss]
            task_metric_rows = [loss_out.metrics]
            task_names = [task_batch[0].task_name]
            plan_modes = [plan_batch[0].mode]

            loss = loss_out.loss
            loss.backward()
        else:
            for _ in range(args.batch_tasks):
                if args.data_mode == "fixed_table":
                    assert full_fixed is not None
                    full = full_fixed
                else:
                    assert table_generator is not None
                    full = table_generator.sample_table()

                info = full.table_info()

                task = sampler.sample(info, np_rng)
                plan = factorizer.build(task, np_rng)

                with autocast_ctx(args, device):
                    loss_out = loss_fn(
                        model=model,
                        full=full,
                        task=task,
                        plan=plan,
                        device=device,
                        num_weight=args.num_weight,
                        cat_weight=args.cat_weight,
                        **sparse_partial_predict_kwargs(args, np_rng),
                    )

                task_losses.append(loss_out.loss)
                task_metric_rows.append(loss_out.metrics)
                task_names.append(task.task_name)
                plan_modes.append(plan.mode)

            loss = torch.stack(task_losses).mean()
            loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            elapsed_since_log = now - last_log_time
            last_log_time = now

            avg_metrics: Dict[str, float] = {}
            for metrics in task_metric_rows:
                for key, value in metrics.items():
                    avg_metrics[key] = avg_metrics.get(key, 0.0) + float(value)

            for key in list(avg_metrics.keys()):
                avg_metrics[key] /= max(len(task_metric_rows), 1)

            row: Dict[str, object] = {
                "step": step,
                "split": "train",
                "sampler": args.sampler,
                "factorization": args.factorization,
                "ar_unit": args.ar_unit,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "steps_per_sec": args.log_every / max(elapsed_since_log, 1e-8),
                "elapsed_sec": time.time() - start_time,
                "task_names": ",".join(task_names),
                "plan_modes": ",".join(plan_modes),
            }

            row.update({f"train/{key}": value for key, value in avg_metrics.items()})

            if device.type == "cuda":
                row["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

            log_metrics(logger, wandb_run, row)

            print(
                f"[step {step:06d}] "
                f"loss={row['loss']:.4f} "
                f"lr={row['lr']:.2e} "
                f"sampler={args.sampler} "
                f"factorization={args.factorization} "
                f"steps/s={row['steps_per_sec']:.2f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate(
                model=model,
                args=args,
                device=device,
                full_fixed=full_fixed,
            )

            eval_loss = next(
                (float(value) for key, value in eval_metrics.items() if key.endswith("/loss")),
                0.0,
            )

            row: Dict[str, object] = {
                "step": step,
                "split": "eval",
                "sampler": args.sampler,
                "factorization": args.factorization,
                "ar_unit": args.ar_unit,
                "loss": eval_loss,
                "grad_norm": 0.0,
                "steps_per_sec": 0.0,
                "elapsed_sec": time.time() - start_time,
            }
            row.update(eval_metrics)

            log_metrics(logger, wandb_run, row)

            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "model_cfg": model_cfg.__dict__,
                },
                ckpt_path,
            )

            print(f"[eval step {step:06d}] saved {ckpt_path}")

            # Print compact eval summary.
            for key, value in sorted(eval_metrics.items()):
                if (
                    key.endswith("/loss")
                    or key.endswith("/cat_acc")
                    or key.endswith("/num_mse")
                    or key.endswith("/logreg128_acc")
                    or key.endswith("/rf128_acc")
                    or key.endswith("/xgb128_acc")
                ):
                    print(f"  {key}: {value:.4f}")

        # Independent periodic checkpoint, decoupled from eval cadence. Skip
        # if this step already got a checkpoint from the eval block above.
        if (
            args.checkpoint_every > 0
            and step % args.checkpoint_every == 0
            and step % args.eval_every != 0
            and step != args.steps
        ):
            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "model_cfg": model_cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"[checkpoint step {step:06d}] saved {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()