# scripts/train_synthetic.py

from __future__ import annotations

import argparse
import json
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
)
from tab_completion.episode_utils import (
    task_to_torch_batch,
    mask_to_torch,
    make_step_query_mask,
)
from tab_completion.model import (
    ModelConfig,
    CellwiseCompletionModel,
)
from tab_completion.losses import typed_mse_ce_loss


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
        )

    if args.sampler == "row_block":
        return RowBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            query_frac_cols=args.query_frac_cols,
        )

    if args.sampler == "label_feature":
        return LabelFeatureSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            n_feature_cols=args.n_feature_cols,
            target_col=args.target_col,
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
                ),
                LabelFeatureSampler(
                    n_context=args.n_context,
                    n_query=args.n_query,
                    n_feature_cols=args.n_feature_cols,
                    target_col=args.target_col,
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

    step_losses: list[torch.Tensor] = []
    metrics_accum: Dict[str, float] = {}
    n_metric_steps = 0

    for step_coords in plan.steps:
        if len(step_coords) == 0:
            continue

        step_query_np = make_step_query_mask(task_shape, step_coords)

        observed_t = mask_to_torch(observed_np, device)
        query_t = mask_to_torch(step_query_np, device)

        out = model(batch, observed_t, query_t)

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


def build_eval_samplers(args) -> Dict[str, object]:
    """
    Every trained model is evaluated on all task families.
    This creates the capability matrix:
      trained objective x evaluated task.
    """
    return {
        "target": TargetPredictionSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            target_col=args.target_col,
        ),
        "random_cell": RandomCellSampler(
            n_episode_rows=args.n_episode_rows,
            query_frac=args.query_frac,
            max_query_cells=args.max_query_cells,
        ),
        "column_block": ColumnBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            min_query_cols=args.min_query_cols,
            max_query_cols=args.max_query_cols,
            exclude_target=args.exclude_target,
        ),
        "row_block": RowBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            query_frac_cols=args.query_frac_cols,
        ),
        "label_feature": LabelFeatureSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            n_feature_cols=args.n_feature_cols,
            target_col=args.target_col,
        ),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    full: FullSyntheticTable,
    args,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    # Reset eval RNG each time so eval tasks are fixed across checkpoints/runs,
    # assuming args.eval_seed is fixed.
    eval_rng = np.random.default_rng(args.eval_seed)

    info = full.table_info()
    eval_samplers = build_eval_samplers(args)
    factorizer = build_factorizer(args)

    all_metrics: Dict[str, float] = {}

    for task_name, sampler in eval_samplers.items():
        values_by_metric: Dict[str, list[float]] = {}

        for _ in range(args.eval_tasks):
            task = sampler.sample(info, eval_rng)
            plan = factorizer.build(task, eval_rng)

            loss_out = compute_task_loss(
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

        for key, values in values_by_metric.items():
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
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-episode-rows", type=int, default=512)
    parser.add_argument("--max-cols", type=int, default=128)
    parser.add_argument("--num-cat-decode-types", type=int, default=128)
    parser.add_argument("--num-row-layers", type=int, default=2)
    parser.add_argument("--num-row-context-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-weight", type=float, default=1.0)
    parser.add_argument("--cat-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # Eval/log
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="results/synthetic_v0")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tab_completion")

    # Device
    parser.add_argument("--device", type=str, default="auto")

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

    print("=== Building synthetic table ===")
    full = make_synthetic_table(
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
    info = full.table_info()

    print(f"n_rows={info.n_rows}, n_cols={info.n_cols}, target_col={info.target_col}")
    print("col_types:", full.col_types.tolist())
    print("cat_cardinalities:", full.cat_cardinalities.tolist())

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
    )

    model = CellwiseCompletionModel(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters())

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

        for _ in range(args.batch_tasks):
            task = sampler.sample(info, np_rng)
            plan = factorizer.build(task, np_rng)

            loss_out = compute_task_loss(
                model=model,
                full=full,
                task=task,
                plan=plan,
                device=device,
                num_weight=args.num_weight,
                cat_weight=args.cat_weight,
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
                f"sampler={args.sampler} "
                f"factorization={args.factorization} "
                f"steps/s={row['steps_per_sec']:.2f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate(
                model=model,
                full=full,
                args=args,
                device=device,
            )

            row: Dict[str, object] = {
                "step": step,
                "split": "eval",
                "sampler": args.sampler,
                "factorization": args.factorization,
                "ar_unit": args.ar_unit,
                "loss": float(eval_metrics.get("eval/target/loss", 0.0)),
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
                ):
                    print(f"  {key}: {value:.4f}")

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()