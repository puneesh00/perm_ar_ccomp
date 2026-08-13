# scripts/train_tabpfn_v1_baseline.py
"""
Trains the reference TabPFN-v1-style model (tab_completion/model_tabpfn_v1.py)
on the exact same synthetic SCM prior and target-prediction eval harness used
by scripts/train_synthetic.py's `two_stream_ar` runs, so the two are directly
comparable: same data, same logreg/rf/xgb context baselines, same optimizer
recipe. Only the model differs. See model_tabpfn_v1.py's module docstring for
why this is a useful control.

Only supports the `target` sampler / parallel-equivalent regime (this model
has no notion of arbitrary cell completion) -- that's the one metric
(eval/target/cat_acc) this whole investigation has been tracking anyway.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tab_completion.model import NUMERICAL
from tab_completion.model_tabpfn_v1 import TabPFNV1Config, TabPFNV1Model
from tab_completion.sampling import TargetPredictionSampler
from tab_completion.synthetic_data import FullSyntheticTable
from tab_completion.synthetic_data_tabpfn import TabPFNSCMConfig, TabPFNSCMTableGenerator

from train_synthetic import (
    logreg_context_baseline_acc,
    rf_context_baseline_acc,
    xgb_context_baseline_acc,
)


def build_xy(full: FullSyntheticTable, task, n_context: int, n_query: int):
    """
    task: CompletionTask from TargetPredictionSampler.sample -- row_idx is
    context rows first (0..n_context-1), query rows after.

    Labels are densified to match real TabPFN: each raw class id is remapped
    to its rank among the UNIQUE values observed in the context rows (see
    TabPFNV2's _flatten_multiclass_targets), never the raw generator class
    id. This is what makes a fixed-width nn.Linear(..., max_num_classes)
    output head correct for a prior that samples class count per table --
    the head doesn't need to know the realized count, only an upper bound.

    A query row whose true label never appears in the context split can't be
    predicted correctly by construction (the model never saw that class), so
    it's mapped to -100 (torch's cross_entropy ignore_index) rather than to
    a fake/invalid class slot.

    Returns (x_feat [N, D_feat] float32, y_context [n_context] float32
             (densified), y_query [n_query] int64 (densified, or -100 for
             context-unseen labels), num_valid_classes int -- number of
             unique labels observed in this episode's context).
    """
    target_col = task.meta["target_col"]
    feature_cols = [c for c in task.col_idx.tolist() if c != target_col]
    rows = task.row_idx

    parts = []
    for j in feature_cols:
        if full.col_types[j] == NUMERICAL:
            parts.append(full.x_num[rows, j].astype(np.float32))
        else:
            parts.append(full.x_cat[rows, j].astype(np.float32))
    x_feat = np.stack(parts, axis=1)  # [N, D_feat]

    y_all = full.x_cat[rows, target_col].astype(np.int64)
    y_context_raw = y_all[:n_context]
    y_query_raw = y_all[n_context:]

    unique_context = np.unique(y_context_raw)
    num_valid_classes = int(unique_context.shape[0])
    remap = {int(v): i for i, v in enumerate(unique_context.tolist())}

    y_context = np.array([remap[int(v)] for v in y_context_raw], dtype=np.float32)
    y_query = np.array(
        [remap.get(int(v), -100) for v in y_query_raw], dtype=np.int64
    )

    return x_feat, y_context, y_query, num_valid_classes


def make_batch(
    table_generator: TabPFNSCMTableGenerator,
    sampler: TargetPredictionSampler,
    rng: np.random.Generator,
    batch_size: int,
    n_context: int,
    n_query: int,
    device: torch.device,
    baselines: bool = False,
):
    x_feats, y_ctxs, y_qrys, num_valid_classes_list = [], [], [], []
    baseline_accs = {"logreg": [], "rf": [], "xgb": []}

    for _ in range(batch_size):
        full = table_generator.sample_table()
        info = full.table_info()
        task = sampler.sample(info, rng)

        x_feat, y_context, y_query, num_valid_classes = build_xy(full, task, n_context, n_query)
        x_feats.append(x_feat)
        y_ctxs.append(y_context)
        y_qrys.append(y_query)
        num_valid_classes_list.append(num_valid_classes)

        if baselines:
            lr = logreg_context_baseline_acc(full, task)
            rf = rf_context_baseline_acc(full, task)
            xgb = xgb_context_baseline_acc(full, task)
            if lr is not None:
                baseline_accs["logreg"].append(lr)
            if rf is not None:
                baseline_accs["rf"].append(rf)
            if xgb is not None:
                baseline_accs["xgb"].append(xgb)

    x_feat_t = torch.as_tensor(np.stack(x_feats), dtype=torch.float32, device=device)
    y_ctx_t = torch.as_tensor(np.stack(y_ctxs), dtype=torch.float32, device=device)
    y_qry_t = torch.as_tensor(np.stack(y_qrys), dtype=torch.long, device=device)
    num_valid_classes_t = torch.as_tensor(num_valid_classes_list, dtype=torch.long, device=device)

    return x_feat_t, y_ctx_t, y_qry_t, num_valid_classes_t, baseline_accs


@torch.no_grad()
def evaluate(model, args, device) -> Dict[str, float]:
    model.eval()
    eval_rng = np.random.default_rng(args.eval_seed)
    eval_gen = TabPFNSCMTableGenerator(
        TabPFNSCMConfig(
            n_rows=args.fresh_n_rows,
            n_cols=args.n_cols,
            p_categorical=args.p_categorical,
            k_max=args.k_max,
            n_classes=None,
            target_col=args.target_col,
            base_seed=args.eval_seed,
            prior_type=args.tabpfn_prior_type,
            layers_mu_max=args.tabpfn_layers_mu_max,
            layers_max=args.tabpfn_layers_max,
            hidden_mu_max=args.tabpfn_hidden_mu_max,
        )
    )
    eval_sampler = TargetPredictionSampler(
        n_context=args.eval_n_context, n_query=args.eval_n_query, target_col=args.target_col
    )

    accs = []
    baseline_accs = {"logreg": [], "rf": [], "xgb": []}
    for _ in range(args.eval_tasks):
        x_feat_t, y_ctx_t, y_qry_t, num_valid_classes_t, b = make_batch(
            eval_gen, eval_sampler, eval_rng, 1, args.eval_n_context, args.eval_n_query, device,
            baselines=True,
        )
        logits = model(x_feat_t, y_ctx_t, args.eval_n_context, num_valid_classes=num_valid_classes_t)
        pred = logits.argmax(dim=-1)
        accs.append((pred == y_qry_t).float().mean().item())
        for k in baseline_accs:
            baseline_accs[k].extend(b[k])

    model.train()
    out = {"eval/target/cat_acc": float(np.mean(accs))}
    for k, key in (("logreg", "logreg128_acc"), ("rf", "rf128_acc"), ("xgb", "xgb128_acc")):
        if baseline_accs[k]:
            out[f"eval/target/{key}"] = float(np.mean(baseline_accs[k]))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tabpfn-prior-type", type=str, default="scm")
    p.add_argument("--tabpfn-layers-mu-max", type=float, default=3.0)
    p.add_argument("--tabpfn-layers-max", type=int, default=4)
    p.add_argument("--tabpfn-hidden-mu-max", type=float, default=40.0)
    p.add_argument("--fresh-n-rows", type=int, default=256)
    p.add_argument("--n-cols", type=int, default=16)
    p.add_argument("--target-col", type=int, default=None)
    p.add_argument("--p-categorical", type=float, default=0.3)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument(
        "--max-num-classes", type=int, default=10,
        help="Fixed output-head width (upper bound on per-table sampled class "
        "count, matching TabPFNSCMConfig.n_classes_max_max). Not a per-run "
        "constant class count -- see model_tabpfn_v1.py's TabPFNV1Config docstring.",
    )

    p.add_argument("--n-context", type=int, default=128)
    p.add_argument("--n-query", type=int, default=128)
    p.add_argument("--eval-n-context", type=int, default=128)
    p.add_argument("--eval-n-query", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=999)
    p.add_argument("--eval-tasks", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)

    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-tasks", type=int, default=8)
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

    p.add_argument("--out-dir", type=str, default="results/synthetic_v2")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    table_generator = TabPFNSCMTableGenerator(
        TabPFNSCMConfig(
            n_rows=args.fresh_n_rows,
            n_cols=args.n_cols,
            p_categorical=args.p_categorical,
            k_max=args.k_max,
            n_classes=None,
            target_col=args.target_col,
            base_seed=args.seed,
            prior_type=args.tabpfn_prior_type,
            layers_mu_max=args.tabpfn_layers_mu_max,
            layers_max=args.tabpfn_layers_max,
            hidden_mu_max=args.tabpfn_hidden_mu_max,
        )
    )
    sampler = TargetPredictionSampler(n_context=args.n_context, n_query=args.n_query, target_col=args.target_col)
    np_rng = np.random.default_rng(args.seed)

    model_cfg = TabPFNV1Config(
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden=args.mlp_hidden,
        n_layers=args.n_layers,
        max_num_classes=args.max_num_classes,
        dropout=args.dropout,
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
    print("=== TabPFN-v1-style reference model ===")
    print(f"device={device}")
    print(f"params={n_params:,}")
    print(f"run_dir={out_dir}")

    metrics_path = out_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a")

    model.train()
    start_time = time.time()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        x_feat_t, y_ctx_t, y_qry_t, num_valid_classes_t, _ = make_batch(
            table_generator, sampler, np_rng, args.batch_tasks, args.n_context, args.n_query, device,
            baselines=False,
        )
        logits = model(
            x_feat_t, y_ctx_t, args.n_context, num_valid_classes=num_valid_classes_t
        )  # [B, n_query, max_num_classes]

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, args.max_num_classes), y_qry_t.reshape(-1), ignore_index=-100
        )
        loss.backward()

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                cat_acc = (logits.argmax(dim=-1) == y_qry_t).float().mean().item()
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            print(
                f"[step {step:06d}] loss={loss.item():.4f} lr={lr_now:.2e} "
                f"train_cat_acc={cat_acc:.4f} steps/s={steps_per_sec:.2f}"
            )
            metrics_file.write(json.dumps({
                "step": step, "split": "train", "loss": loss.item(),
                "train/cat_acc": cat_acc, "lr": lr_now, "grad_norm": float(grad_norm),
            }) + "\n")
            metrics_file.flush()

        if step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate(model, args, device)
            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, ckpt_path)
            print(f"[eval step {step:06d}] saved {ckpt_path}")
            for key, value in sorted(eval_metrics.items()):
                print(f"  {key}: {value:.4f}")
            metrics_file.write(json.dumps({"step": step, "split": "eval", **eval_metrics}) + "\n")
            metrics_file.flush()

    metrics_file.close()
    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
