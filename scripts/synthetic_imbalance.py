# scripts/synthetic_imbalance.py
"""
Per-episode class balance + per-checkpoint accuracy/balanced_accuracy on
fresh synthetic (k=1, true label prediction) episodes -- the per-episode
sibling of eval_synthetic_incontext.py, which only logs metrics aggregated
across all --eval-tasks episodes. This script keeps one row per (episode,
checkpoint) specifically so results can be bucketed by how imbalanced that
episode's context was, instead of only seeing one pooled average.

Fixing --eval-seed and the generative-prior flags (--fresh-n-rows, --n-cols,
--p-categorical, --k-max, --tabpfn-*) reproduces the exact same sequence of
fresh tables regardless of machine -- see eval_synthetic_incontext.py's
module docstring. Run this with the same flags (and the same n_context/
n_query) as whatever you're comparing against, on any machine, and the
per-episode rows line up 1:1 by `episode`.

Two balance metrics per episode's CONTEXT, both over REALIZED classes only:
  minority_frac:      (count of the rarest realized class) / n_context.
                       Not comparable across tables with different realized
                       class counts -- e.g. episodes here span n_classes
                       2-10 (see sample_n_classes in synthetic_data_tabpfn.py),
                       and a perfectly balanced 10-class table caps out at
                       10%, same range as a badly skewed BINARY table.
  normalized_balance:  minority_frac * n_realized_classes. 1.0 for a
                       perfectly balanced table regardless of class count,
                       degrading toward 0 as it skews -- bucket by this one
                       when comparing across the mixed class-count episodes
                       this script produces.
  A realized class count of 1 (every context row the same class -- see
  bin_by_realized_values in synthetic_data_tabpfn.py: boundaries are drawn
  as random DATA POINTS, not balanced quantiles, so this happens by chance
  even when the table's designated n_classes is > 1) gets minority_frac=0
  and normalized_balance=0 either way -- worth keeping as its own bucket
  rather than folding into the lowest numeric one, since it's a genuinely
  different case (literally zero within-context value signal for that
  column once context_normalize z-scores it -- see the write-up this script
  was built to check).

CHECKPOINT COMPATIBILITY: --checkpoints uses eval_openml_incontext.py's
load_checkpoint / predict_stream_family_parallel_multi, which auto-detects
single_stream / two_stream_ar / two_stream_ar_sparse / this-repo's own
TabPFNV1Model family (train_tabpfn_v1_baseline.py) from the checkpoint's own
contents. It does NOT load the officially pip-installed `tabpfn` package
(tabpfn==0.1.11 / tabpfn>=2.0) -- that's a different prediction interface
entirely (sklearn .fit()/.predict_proba(), no torch checkpoint file), see
eval_openml_official_tabpfn.py's build_model for how that repo's harness
calls it. Comparing official TabPFN checkpoints against this script's output
means adapting a small predict_official_tabpfn(...) analogous to that
file's build_model, feeding it the same (full, task) this script already
builds each episode -- not wired up here since this repo never trains an
official-TabPFN-family checkpoint itself.

Usage:
    python scripts/synthetic_imbalance.py \\
        --checkpoints "path1.pt=tag1,path2.pt=tag2" \\
        --eval-tasks 100 --eval-seed 999 \\
        --n-context 512 --n-query 64 \\
        --out-csv results/synthetic_incontext/synthetic_context_imbalance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from eval_synthetic_incontext import build_table_generator, build_task  # noqa: E402
from eval_openml_incontext import (  # noqa: E402
    load_checkpoint,
    parse_checkpoints_arg,
    ConvertedTableMulti,
    predict_stream_family_parallel_multi,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="'path1=tag1,path2=tag2,...'")
    parser.add_argument("--eval-tasks", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=999)
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

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if device.type == "cpu" and args.device == "auto":
        device = torch.device("cpu")

    checkpoint_specs = parse_checkpoints_arg(args.checkpoints)
    checkpoints = [load_checkpoint(p, t, device) for p, t in checkpoint_specs]

    table_gen = build_table_generator(args)
    row_rng = np.random.default_rng(args.eval_seed)
    col_rng = np.random.default_rng([1, args.eval_seed])

    n_context, n_query = args.n_context, args.n_query
    rows = []

    for ep in range(args.eval_tasks):
        full = table_gen.sample_table()
        task = build_task(full, 1, n_context, n_query, row_rng, col_rng, False, True)

        y_context = full.x_cat[task.row_idx[:n_context], full.target_col]
        counts = np.bincount(y_context)
        counts = counts[counts > 0]  # realized classes only, see module docstring
        n_realized = len(counts)
        minority_frac = 0.0 if n_realized == 1 else counts.min() / counts.sum()
        normalized_balance = minority_frac * n_realized

        table = ConvertedTableMulti(
            full=full, task=task, n_context=n_context, n_query=n_query,
            query_cols=np.array(task.meta["query_cols"]), n_classes=0,
            max_feature_cardinality=0, conditioning_mode="inductive_rows",
        )
        for ckpt in checkpoints:
            pred = predict_stream_family_parallel_multi(ckpt, table, device)
            acc = float((pred.y_true == pred.y_pred).mean())
            try:
                bacc = float(balanced_accuracy_score(pred.y_true, pred.y_pred))
            except Exception:
                bacc = float("nan")
            rows.append({
                "episode": ep, "checkpoint": ckpt.tag,
                "n_realized_classes": n_realized,
                "minority_frac": minority_frac,
                "normalized_balance": normalized_balance,
                "accuracy": acc, "balanced_accuracy": bacc,
            })

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.head(20).to_string(index=False))
    print(f"...\nWrote {len(df)} rows ({args.eval_tasks} episodes x {len(checkpoints)} checkpoints) to {out_path}")


if __name__ == "__main__":
    main()
