# scripts/eval_synthetic_incontext.py
"""
Evaluates trained checkpoints against FRESH tables drawn from the SAME
synthetic prior used during training -- in-distribution eval, complementary
to eval_openml_incontext.py's real-data comparison.

Reuses the exact prediction/metrics/baseline machinery from
eval_openml_incontext.py (predict_stream_family_parallel_multi,
predict_stream_family_ar_multi, multi_cell_metrics,
build_tuned_multi_col_families, predict_baseline_family_multi,
predict_baseline_family_chained) -- only the table SOURCE differs
(TabPFNSCMTableGenerator + a hand-built CompletionTask, mirroring
eval_openml_incontext.py's convert_openml_table_multi, instead of an
OpenML DataFrame).

Portable across machines: fixing --eval-seed and the generative-prior
config (--p-categorical, --k-max, --fresh-n-rows, --n-cols, --tabpfn-*)
reproduces the exact same sequence of fresh tables regardless of which
machine runs it (numpy's PCG64 seeding is deterministic and platform-
independent) -- run this with the same flags on a different machine
against a different checkpoint, and both are directly comparable.

k=1 always queries that table's own designated target column (true label
prediction, not a random single column). k>1 draws k columns per episode,
either from all columns (default -- matches what column_block-trained
checkpoints actually trained on) or, with --categorical-only, from only
that table's categorical columns (needed for checkpoints whose num_head
never trained on a numeric query cell, e.g. label-only runs) -- NOT
restricted by default, so results without the flag mix categorical and
numeric columns proportional to --p-categorical, same "effective k varies
per episode" caveat as the OpenML multi_target eval.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from run_openml_baselines import JSONLLogger, write_csv_from_jsonl  # noqa: E402
from eval_openml_incontext import (  # noqa: E402
    load_checkpoint,
    parse_checkpoints_arg,
    eligible_for_checkpoint,
    predict_stream_family_parallel_multi,
    predict_stream_family_ar_multi,
    multi_cell_metrics,
    build_tuned_multi_col_families,
    predict_baseline_family_multi,
    predict_baseline_family_chained,
    ConvertedTableMulti,
    MultiCellPrediction,
)
from tab_completion.model import CATEGORICAL  # noqa: E402
from tab_completion.synthetic_data_tabpfn import TabPFNSCMConfig, TabPFNSCMTableGenerator  # noqa: E402
from tab_completion.sampling import CompletionTask, sample_unique_rows_fast  # noqa: E402


def build_table_generator(args) -> TabPFNSCMTableGenerator:
    return TabPFNSCMTableGenerator(TabPFNSCMConfig(
        n_rows=args.fresh_n_rows, n_cols=args.n_cols, p_categorical=args.p_categorical,
        k_max=args.k_max, n_classes=None, target_col=None, base_seed=args.eval_seed,
        prior_type=args.tabpfn_prior_type, layers_mu_max=args.tabpfn_layers_mu_max,
        layers_max=args.tabpfn_layers_max, hidden_mu_max=args.tabpfn_hidden_mu_max,
    ))


def build_task(full, k, n_context, n_query, row_rng, col_rng, categorical_only, force_target):
    """Mirrors eval_openml_incontext.py's convert_openml_table_multi, sourcing
    rows/columns from a fresh synthetic table instead of an OpenML DataFrame.
    Returns None if this table doesn't have enough eligible columns for k
    (only possible with categorical_only, since p_categorical<1)."""
    n_cols = full.x_num.shape[1]
    n_rows = full.x_num.shape[0]
    n_ep = n_context + n_query
    row_idx = sample_unique_rows_fast(row_rng, n_rows, n_ep, replace=False)
    col_idx = np.arange(n_cols, dtype=np.int64)

    if force_target:
        query_cols = np.array([full.target_col], dtype=np.int64)
    else:
        eligible = col_idx[full.col_types[col_idx] == CATEGORICAL] if categorical_only else col_idx
        if len(eligible) < k:
            return None
        query_cols = np.sort(col_rng.choice(eligible, size=k, replace=False)).astype(np.int64)

    observed_mask = np.ones((n_ep, n_cols), dtype=bool)
    query_mask = np.zeros((n_ep, n_cols), dtype=bool)
    query_rows = np.arange(n_context, n_ep)
    observed_mask[np.ix_(query_rows, query_cols)] = False
    query_mask[np.ix_(query_rows, query_cols)] = True

    return CompletionTask(
        row_idx=row_idx, col_idx=col_idx, observed_mask=observed_mask, query_mask=query_mask,
        task_name="synthetic_multi_target",
        meta={
            "query_cols": query_cols.tolist(),
            "n_context": int(n_context), "n_query": int(n_query),
            "conditioning_mode": "inductive_rows",
            "context_rows_local": np.arange(n_context, dtype=np.int64),
            "query_rows_local": np.arange(n_context, n_ep, dtype=np.int64),
        },
    )


def concat_predictions(preds: List[MultiCellPrediction]) -> Optional[MultiCellPrediction]:
    if not preds:
        return None
    proba_list = [p.y_proba for p in preds if p.y_proba is not None]
    y_proba = None
    if proba_list:
        # k_max (max categorical cardinality) is resampled fresh per table, so
        # it varies episode to episode -- pad to the global max before concat.
        global_k_max = max(arr.shape[1] for arr in proba_list)
        padded = []
        for p in preds:
            arr = p.y_proba
            if arr is None:
                continue
            if arr.shape[1] < global_k_max:
                pad = np.full((arr.shape[0], global_k_max - arr.shape[1]), np.nan, dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            padded.append(arr)
        y_proba = np.concatenate(padded)
    return MultiCellPrediction(
        rows=np.concatenate([p.rows for p in preds]),
        cols=np.concatenate([p.cols for p in preds]),
        y_true=np.concatenate([p.y_true for p in preds]),
        y_pred=np.concatenate([p.y_pred for p in preds]),
        y_proba=y_proba,
        is_categorical=np.concatenate([p.is_categorical for p in preds]),
        step_index=np.concatenate([p.step_index for p in preds]),
    )


def run_k(k: int, args, checkpoints, device, logger, table_gen) -> None:
    force_target = (k == 1)
    row_rng = np.random.default_rng(args.eval_seed)  # reset -- identical fresh tables for every k
    col_rng = np.random.default_rng([k, args.eval_seed])

    model_preds: Dict[str, List[MultiCellPrediction]] = {}
    model_ar_preds: Dict[str, List[MultiCellPrediction]] = {}
    baseline_preds: Dict[str, List[MultiCellPrediction]] = {}
    baseline_chained_preds: Dict[str, List[MultiCellPrediction]] = {}
    row_offset = 0
    n_skipped = 0

    families = build_tuned_multi_col_families(
        args.seed, args.n_jobs, args.baseline_tuning_iters, args.baseline_cv_folds
    ) if args.baseline_modes else {}

    for ep in range(args.eval_tasks):
        full = table_gen.sample_table()
        task = build_task(
            full, k, args.n_context, args.n_query, row_rng, col_rng,
            args.categorical_only, force_target,
        )
        if task is None:
            n_skipped += 1
            continue

        table = ConvertedTableMulti(
            full=full, task=task, n_context=args.n_context, n_query=args.n_query,
            query_cols=np.array(task.meta["query_cols"]), n_classes=0,
            max_feature_cardinality=0, conditioning_mode="inductive_rows",
        )

        for ckpt in checkpoints:
            if eligible_for_checkpoint(table, ckpt) is not None:
                continue
            pred = predict_stream_family_parallel_multi(ckpt, table, device)
            pred.rows = pred.rows + row_offset
            model_preds.setdefault(ckpt.tag, []).append(pred)
            if ckpt.family in ("two_stream_ar", "two_stream_ar_sparse"):
                ar_pred = predict_stream_family_ar_multi(ckpt, table, device)
                ar_pred.rows = ar_pred.rows + row_offset
                model_ar_preds.setdefault(ckpt.tag, []).append(ar_pred)

        if families and ep < args.baseline_eval_tasks:
            for name, (clf_factory, reg_factory) in families.items():
                try:
                    with threadpool_limits(limits=args.n_jobs):
                        if "independent" in args.baseline_modes:
                            bpred = predict_baseline_family_multi(full, task, clf_factory, reg_factory)
                            if bpred is not None:
                                bpred.rows = bpred.rows + row_offset
                                baseline_preds.setdefault(name, []).append(bpred)
                        if "chained" in args.baseline_modes:
                            cpred = predict_baseline_family_chained(full, task, clf_factory, reg_factory)
                            if cpred is not None:
                                cpred.rows = cpred.rows + row_offset
                                baseline_chained_preds.setdefault(name, []).append(cpred)
                except Exception as e:
                    print(f"  [error] baseline {name} k={k} ep={ep}: {e}")

        row_offset += args.n_query

    if n_skipped:
        print(f"  [k={k}] skipped {n_skipped}/{args.eval_tasks} episodes (not enough categorical columns)")

    def log_row(model_name: str, ar_mode: str, metrics: Dict, n_episodes: int) -> None:
        row = {
            "k_cols": k, "model": model_name, "ar_mode": ar_mode,
            "n_episodes": n_episodes, "n_context": args.n_context, "n_query": args.n_query,
            "categorical_only": args.categorical_only, "status": "ok",
        }
        row.update(metrics)
        logger.log(row)

    for tag, preds in model_preds.items():
        combined = concat_predictions(preds)
        metrics = multi_cell_metrics(combined)
        log_row(tag, "parallel", metrics, len(preds))
        print(f"  {tag:24s} k={k:2d} parallel  marginal_cat_acc={metrics.get('marginal_cat_acc', float('nan')):.4f}  "
              f"joint_exact_match={metrics.get('joint_exact_match', float('nan')):.4f}  "
              f"joint_nll={metrics.get('joint_nll', float('nan')):.4f}")

    for tag, preds in model_ar_preds.items():
        combined = concat_predictions(preds)
        metrics = multi_cell_metrics(combined)
        log_row(tag, "perm_ar", metrics, len(preds))
        print(f"  {tag:24s} k={k:2d} perm_ar   marginal_cat_acc={metrics.get('marginal_cat_acc', float('nan')):.4f}  "
              f"joint_exact_match={metrics.get('joint_exact_match', float('nan')):.4f}  "
              f"joint_nll={metrics.get('joint_nll', float('nan')):.4f}")

    for name, preds in baseline_preds.items():
        combined = concat_predictions(preds)
        metrics = multi_cell_metrics(combined)
        log_row(name, "n/a", metrics, len(preds))
        print(f"  {name:24s} k={k:2d} indep.    marginal_cat_acc={metrics.get('marginal_cat_acc', float('nan')):.4f}  "
              f"joint_exact_match={metrics.get('joint_exact_match', float('nan')):.4f}  "
              f"joint_nll={metrics.get('joint_nll', float('nan')):.4f}")

    for name, preds in baseline_chained_preds.items():
        combined = concat_predictions(preds)
        metrics = multi_cell_metrics(combined)
        log_row(f"{name}_chained", "n/a", metrics, len(preds))
        print(f"  {name+'_chained':24s} k={k:2d} chained   marginal_cat_acc={metrics.get('marginal_cat_acc', float('nan')):.4f}  "
              f"joint_exact_match={metrics.get('joint_exact_match', float('nan')):.4f}  "
              f"joint_nll={metrics.get('joint_nll', float('nan')):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="'path1=tag1,path2=tag2,...'")
    parser.add_argument("--k-cols", type=str, default="1,2,4,8")
    parser.add_argument("--categorical-only", action="store_true")

    parser.add_argument("--eval-tasks", type=int, default=100, help="episodes for model predictions")
    parser.add_argument("--baseline-eval-tasks", type=int, default=15,
                         help="episodes for baseline fitting (much slower than model forward passes -- "
                              "kept separate and smaller by default). Uses the same first N episodes as "
                              "the model predictions, so it's an apples-to-apples subset, not a separate draw.")
    parser.add_argument("--eval-seed", type=int, default=999, help="must match across machines/runs to compare")
    parser.add_argument("--n-context", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=64)

    # Generative-prior config -- must match the training run(s) being evaluated.
    parser.add_argument("--fresh-n-rows", type=int, default=576)
    parser.add_argument("--n-cols", type=int, default=64)
    parser.add_argument("--p-categorical", type=float, default=0.3)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--tabpfn-prior-type", type=str, default="scm")
    parser.add_argument("--tabpfn-layers-mu-max", type=float, default=6.0)
    parser.add_argument("--tabpfn-layers-max", type=int, default=None)
    parser.add_argument("--tabpfn-hidden-mu-max", type=float, default=130.0)

    parser.add_argument("--baseline-modes", type=str, default="independent,chained",
                         help="comma-separated: independent, chained. Empty string disables baselines entirely.")
    parser.add_argument("--baseline-tuning-iters", type=int, default=25)
    parser.add_argument("--baseline-cv-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=4)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0, help="baseline RandomizedSearchCV seed")
    parser.add_argument("--out-dir", type=str, default="results/synthetic_incontext")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()
    args.baseline_modes = [m.strip() for m in args.baseline_modes.split(",") if m.strip()]

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if device.type == "cpu" and args.device == "auto":
        device = torch.device("cpu")

    checkpoint_specs = parse_checkpoints_arg(args.checkpoints)
    checkpoints = [load_checkpoint(path, tag, device) for path, tag in checkpoint_specs]

    if args.run_name is None:
        args.run_name = f"synthetic_{'-'.join(c.tag for c in checkpoints)}_{int(time.time())}"
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"
    logger = JSONLLogger(jsonl_path)

    print("=== Synthetic-prior in-context checkpoint evaluation ===")
    print(f"device={device}")
    print(f"checkpoints={[(c.tag, c.family, c.step) for c in checkpoints]}")
    print(f"eval_seed={args.eval_seed}  eval_tasks={args.eval_tasks}  baseline_eval_tasks={args.baseline_eval_tasks}")
    print(f"categorical_only={args.categorical_only}")
    print(f"out_dir={out_dir}")

    table_gen = build_table_generator(args)
    k_list = [int(k) for k in args.k_cols.split(",")]
    for k in k_list:
        print(f"\n=== k={k} ===")
        try:
            run_k(k, args, checkpoints, device, logger, table_gen)
        except Exception as e:
            print(f"[error] k={k}: {e}")
            logger.log({"k_cols": k, "model": None, "status": "error", "error": repr(e),
                        "traceback": traceback.format_exc()})
        write_csv_from_jsonl(jsonl_path, csv_path)

    print("\nDone.")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
