# scripts/train_tabpfn_v1_baseline_official_prior.py
"""
Same TabPFN-v1-style reference model (tab_completion/model_tabpfn_v1.py),
same optimizer/schedule/logging/eval as train_tabpfn_v1_baseline.py -- the
ONLY thing that changes is where training tables come from: instead of our
own reimplemented SCM generator (synthetic_data_tabpfn.py), tables are
generated live from the REAL released TabPFN-v1 prior code (see
official_v1_prior_gen.py), to test whether training on the actual official
prior (rather than our approximation of it) closes any of the real-data
(OpenML) gap. Eval (the periodic --eval-every synthetic sanity check, using
logreg/rf/xgb context baselines) deliberately stays on OUR OWN generator
unchanged, so its curve is directly comparable to train_tabpfn_v1_baseline.py's
existing runs trained on our prior. The real comparison metric is the
separate OpenML eval (eval_openml_incontext.py) run after training, same as
every other checkpoint in this investigation.

Requires the environment this runs in to have `tabpfn` (the v1 pip package,
tabpfn==0.1.11) importable -- e.g. .venv_tabpfn_v1 in this repo, run with
PYTHONPATH=src:scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

torch.set_num_threads(1)  # see official_v1_prior_gen.py's module docstring

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tab_completion.model_tabpfn_v1 import TabPFNV1Config, TabPFNV1Model
from tab_completion.sampling import TargetPredictionSampler

from train_synthetic import autocast_ctx, resample_variable_table_shape
from train_tabpfn_v1_baseline import build_xy, make_batch, evaluate

from official_v1_prior_gen import OfficialV1LiveTableGenerator


def parse_args():
    p = argparse.ArgumentParser()
    # --- official-prior generator knobs ---
    p.add_argument("--fresh-n-rows", type=int, default=576)
    p.add_argument("--n-cols", type=int, default=64)
    p.add_argument(
        "--n-query-for-check", type=int, default=64,
        help="Rows reserved as 'query' purely for official's own internal "
        "check_is_compatible label-coverage validation at generation time "
        "-- unrelated to this run's actual training --n-context/--n-query, "
        "which TargetPredictionSampler draws independently from the full "
        "generated table.",
    )
    p.add_argument(
        "--variable-table-shape",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mirrors train_tabpfn_v1_baseline.py's flag of the same name.",
    )
    p.add_argument("--min-n-context", type=int, default=None)
    p.add_argument("--max-n-context", type=int, default=None)
    p.add_argument("--min-n-cols", type=int, default=None)
    p.add_argument("--max-n-cols", type=int, default=None)
    p.add_argument("--max-num-classes", type=int, default=10)

    p.add_argument("--n-context", type=int, default=128)
    p.add_argument("--n-query", type=int, default=128)

    # --- eval (stays on OUR OWN generator, unchanged from train_tabpfn_v1_baseline.py) ---
    p.add_argument("--eval-n-context", type=int, default=128)
    p.add_argument("--eval-n-query", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=999)
    p.add_argument("--eval-tasks", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--tabpfn-prior-type", type=str, default="scm")
    p.add_argument("--tabpfn-layers-mu-max", type=float, default=6.0)
    p.add_argument("--tabpfn-layers-max", type=int, default=None)
    p.add_argument("--tabpfn-hidden-mu-max", type=float, default=130.0)
    p.add_argument("--p-categorical", type=float, default=0.3)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument("--target-col", type=int, default=None)

    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-tasks", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=10000)

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)

    p.add_argument("--amp-dtype", type=str, choices=["none", "bf16", "fp16"], default="none")

    p.add_argument("--out-dir", type=str, default="results/synthetic_v2")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.variable_table_shape:
        if None in (args.min_n_context, args.max_n_context, args.min_n_cols, args.max_n_cols):
            raise ValueError(
                "--variable-table-shape requires --min-n-context, --max-n-context, "
                "--min-n-cols, and --max-n-cols to all be set."
            )
        if args.min_n_context > args.max_n_context:
            raise ValueError("--min-n-context must be <= --max-n-context.")
        if args.min_n_cols > args.max_n_cols:
            raise ValueError("--min-n-cols must be <= --max-n-cols.")
        if args.min_n_cols < 4:
            raise ValueError("--min-n-cols must be >= 4.")
        if args.max_n_context + args.n_query > args.fresh_n_rows:
            raise ValueError(
                f"--max-n-context + --n-query ({args.max_n_context} + {args.n_query} "
                f"= {args.max_n_context + args.n_query}) exceeds --fresh-n-rows "
                f"({args.fresh_n_rows})."
            )
        args.target_col = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    table_generator = OfficialV1LiveTableGenerator(
        n_rows=args.fresh_n_rows, n_cols=args.n_cols, base_seed=args.seed,
        n_query_for_check=args.n_query_for_check,
    )
    sampler = TargetPredictionSampler(n_context=args.n_context, n_query=args.n_query, target_col=args.target_col)
    np_rng = np.random.default_rng(args.seed)

    model_cfg = TabPFNV1Config(
        d_model=args.d_model, n_heads=args.n_heads, mlp_hidden=args.mlp_hidden,
        n_layers=args.n_layers, max_num_classes=args.max_num_classes, dropout=args.dropout,
    )
    model = TabPFNV1Model(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
        total_steps = args.steps
        min_ratio = args.lr_min_ratio

        def lr_lambda(step: int) -> float:
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
    tables_per_opt_step = args.batch_tasks * args.grad_accum_steps
    print("=== TabPFN-v1-style reference model, trained on the REAL official v1 prior ===")
    print(f"device={device}")
    print(f"amp_dtype={args.amp_dtype}")
    print(f"params={n_params:,}")
    print(f"run_dir={out_dir}")
    print(
        f"batch_tasks={args.batch_tasks} x grad_accum_steps={args.grad_accum_steps} "
        f"= {tables_per_opt_step} tables/optimizer-step; "
        f"{args.steps} steps -> {tables_per_opt_step * args.steps:,} tables total"
    )

    metrics_path = out_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a")

    model.train()
    start_time = time.time()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        loss_sum_t = torch.zeros((), device=device)
        correct_sum_t = torch.zeros((), device=device)
        total_count = 0
        for _ in range(args.grad_accum_steps):
            if args.variable_table_shape:
                resample_variable_table_shape(args, table_generator, sampler, np_rng)

            x_feat_t, y_ctx_t, y_qry_t, num_valid_classes_t, _ = make_batch(
                table_generator, sampler, np_rng, args.batch_tasks, sampler.n_context, args.n_query, device,
                baselines=False,
            )
            with autocast_ctx(args, device):
                logits = model(
                    x_feat_t, y_ctx_t, sampler.n_context, num_valid_classes=num_valid_classes_t
                )
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, args.max_num_classes), y_qry_t.reshape(-1), ignore_index=-100
                )
            (loss / args.grad_accum_steps).backward()

            loss_sum_t += loss.detach()
            with torch.no_grad():
                correct_sum_t += (logits.argmax(dim=-1) == y_qry_t).float().sum()
                total_count += y_qry_t.numel()

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0 or step == 1:
            avg_loss = (loss_sum_t / args.grad_accum_steps).item()
            cat_acc = (correct_sum_t / total_count).item()
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            print(
                f"[step {step:06d}] loss={avg_loss:.4f} lr={lr_now:.2e} "
                f"train_cat_acc={cat_acc:.4f} steps/s={steps_per_sec:.2f}"
            )
            metrics_file.write(json.dumps({
                "step": step, "split": "train", "loss": avg_loss,
                "train/cat_acc": cat_acc, "lr": lr_now, "grad_norm": float(grad_norm),
            }) + "\n")
            metrics_file.flush()

        if step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate(model, args, device)
            print(f"[eval step {step:06d}]")
            for key, value in sorted(eval_metrics.items()):
                print(f"  {key}: {value:.4f}")
            metrics_file.write(json.dumps({"step": step, "split": "eval", **eval_metrics}) + "\n")
            metrics_file.flush()

        if step % args.checkpoint_every == 0 or step == args.steps:
            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, ckpt_path)
            print(f"[checkpoint step {step:06d}] saved {ckpt_path}")

    metrics_file.close()
    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
