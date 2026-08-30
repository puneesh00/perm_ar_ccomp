# scripts/list_official_eval_tasks.py
"""
Lists the OpenML-CC18 task ids that satisfy the same size/shape filter
eval_openml_official_tabpfn.py and eval_openml_incontext.py apply at
runtime (n_rows in [min_rows, max_total_rows], n_features <= max_features,
no missing values), plus an n_classes <= max_classes cut matching TabPFN-v1's
hard class-count limit. Run standalone before an official-checkpoint eval
sweep to fix the exact task_id list up front.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_openml_baselines import get_suite_task_ids, load_openml_task  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="cc18")
    parser.add_argument("--max-total-rows", type=int, default=1024)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=63)
    parser.add_argument("--max-classes", type=int, default=10)
    args = parser.parse_args()

    task_ids = get_suite_task_ids(args.suite)
    print(f"suite={args.suite} total_tasks={len(task_ids)}", file=sys.stderr)

    kept = []
    for task_id in task_ids:
        try:
            task, X, y = load_openml_task(task_id)
        except Exception as e:
            print(f"[error] task={task_id}: failed to load ({e!r})", file=sys.stderr)
            continue

        task_name = getattr(task, "name", None) or str(task_id)
        n_rows, n_features = X.shape
        n_classes = int(y.nunique(dropna=True))
        has_na = bool(X.isna().any().any() or y.isna().any())

        reasons = []
        if not (args.min_rows <= n_rows <= args.max_total_rows):
            reasons.append(f"n_rows={n_rows} outside [{args.min_rows},{args.max_total_rows}]")
        if n_features > args.max_features:
            reasons.append(f"n_features={n_features} > {args.max_features}")
        if has_na:
            reasons.append("has missing values")
        if n_classes > args.max_classes:
            reasons.append(f"n_classes={n_classes} > {args.max_classes}")

        status = "KEEP" if not reasons else "skip"
        print(
            f"[{status}] task_id={task_id} name={task_name} n_rows={n_rows} "
            f"n_features={n_features} n_classes={n_classes}"
            + (f" -- {'; '.join(reasons)}" if reasons else "")
        )
        if not reasons:
            kept.append(task_id)

    print(f"\n=== kept {len(kept)}/{len(task_ids)} tasks ===")
    print(",".join(str(t) for t in kept))


if __name__ == "__main__":
    main()
