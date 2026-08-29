# scripts/cc18_imbalance.py
"""
Computes, per (task_id, seed), the class balance of the CONTEXT split that
eval_openml_incontext.py's --regime target would build for that exact
(task_id, seed) pair -- same convert_openml_table call, same rng formula
(np.random.default_rng(seed + task_id)), so this is checkpoint-agnostic and
joins directly onto any --regime target run's metrics.csv (join key:
task_id, seed) regardless of which checkpoint produced it, official
TabPFN v1/v2 included.

Two balance metrics, both computed over REALIZED classes only (a class with
zero rows in this context split is dropped before computing either, rather
than counted as a phantom zero-count "minority" -- see the comment on
sample_n_classes-adjacent binning in synthetic_imbalance.py for the sibling
issue on the synthetic side):

  minority_frac:      (count of the rarest realized class) / n_context.
                       Not comparable across different n_classes -- a
                       perfectly balanced k-class table caps out at 1/k.
  normalized_balance:  minority_frac * n_realized_classes. 1.0 for a
                       perfectly balanced table regardless of k (binary or
                       10-way), degrading toward 0 as it skews -- the
                       metric to bucket by when mixing datasets of
                       different class counts.

Usage:
    python scripts/cc18_imbalance.py \\
        --task-ids 11,31,37,49,53,3560,3913,9946,9971,10101,146819 \\
        --seeds 0,1,2,3,4 \\
        --out-csv results/openml_incontext/cc18_context_imbalance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from run_openml_baselines import load_openml_task  # noqa: E402
from eval_openml_incontext import convert_openml_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-ids", type=str, required=True,
        help="Comma-separated OpenML task ids. Must match whatever --task-ids "
             "(or the implied suite selection) the eval run(s) you're joining "
             "against used, or the (task_id, seed) join key won't line up.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--max-query", type=int, default=512)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for task_id in task_ids:
        try:
            _task, X, y_raw = load_openml_task(task_id)
        except Exception as e:
            print(f"[skip] task={task_id}: failed to load ({e})")
            continue
        if X.isna().any().any() or y_raw.isna().any():
            print(f"[skip] task={task_id}: contains missing values "
                  f"(eval_openml_incontext.py skips these too)")
            continue

        for seed in seeds:
            rng = np.random.default_rng(seed + task_id)
            table = convert_openml_table(X, y_raw, rng, args.max_context, args.max_query)
            counts = np.bincount(table.y_train)
            counts = counts[counts > 0]  # realized classes only, see module docstring
            n_realized = len(counts)
            minority_frac = counts.min() / counts.sum()
            rows.append({
                "task_id": task_id, "seed": seed,
                "n_context": table.n_context, "n_query": table.n_query,
                "n_realized_classes": n_realized,
                "minority_frac": minority_frac,
                "normalized_balance": minority_frac * n_realized,
            })

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
