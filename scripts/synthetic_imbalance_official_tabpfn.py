# scripts/synthetic_imbalance_official_tabpfn.py
"""
Official-TabPFN sibling of scripts/synthetic_imbalance.py -- that script's
own docstring says it doesn't cover the officially pip-installed `tabpfn`
package (sklearn .fit()/.predict_proba(), not this repo's checkpoint
loading), and points at eval_openml_official_tabpfn.py's build_model as the
adapter to write. This is that adapter, reusing the exact same per-episode
table/task construction (build_table_generator/build_task) and the same
context-balance computation as synthetic_imbalance.py, so its output rows
line up 1:1 by `episode` with that script's and with
eval_synthetic_official_tabpfn.py's (X_train/y_train/X_test/y_test
flattening also reused from that module).

Meant to be invoked twice, once from each dedicated venv
(.venv_tabpfn_v1, .venv_tabpfn_v2), via --version v1|v2.

Usage:
    python scripts/synthetic_imbalance_official_tabpfn.py \\
        --version v1 --eval-tasks 100 --eval-seed 999 \\
        --n-context 512 --n-query 64 \\
        --out-csv results/synthetic_incontext/synthetic_context_imbalance_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from eval_synthetic_incontext import build_table_generator, build_task  # noqa: E402
from eval_synthetic_official_tabpfn import build_model, flatten_xy, padded_proba  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2"])
    parser.add_argument("--eval-tasks", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--n-context", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=64)

    parser.add_argument("--fresh-n-rows", type=int, default=576)
    parser.add_argument("--n-cols", type=int, default=64)
    parser.add_argument("--p-categorical", type=float, default=0.3)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--tabpfn-prior-type", type=str, default="scm")
    parser.add_argument("--tabpfn-layers-mu-max", type=float, default=6.0)
    parser.add_argument("--tabpfn-layers-max", type=int, default=None)
    parser.add_argument("--tabpfn-hidden-mu-max", type=float, default=130.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    table_gen = build_table_generator(args)
    row_rng = np.random.default_rng(args.eval_seed)
    col_rng = np.random.default_rng([1, args.eval_seed])  # k=1, matches synthetic_imbalance.py

    checkpoint_tag = f"tabpfn_{args.version}_official"
    rows = []

    for ep in range(args.eval_tasks):
        full = table_gen.sample_table()
        task = build_task(
            full, 1, args.n_context, args.n_query, row_rng, col_rng,
            categorical_only=False, force_target=True,
        )

        y_context = full.x_cat[task.row_idx[: args.n_context], full.target_col]
        counts = np.bincount(y_context)
        counts = counts[counts > 0]
        n_realized = len(counts)
        minority_frac = 0.0 if n_realized == 1 else counts.min() / counts.sum()
        normalized_balance = minority_frac * n_realized

        X_train, y_train, X_test, y_test = flatten_xy(full, task)

        try:
            clf = build_model(args.version, args.device, args.seed)
            clf.fit(X_train, y_train)
            proba_raw = clf.predict_proba(X_test)
            classes_ = np.asarray(clf.classes_)
            n_classes = int(full.cat_cardinalities[int(task.meta["query_cols"][0])])
            proba = padded_proba(proba_raw, classes_, n_classes)
            y_pred = proba.argmax(axis=1)

            acc = float((y_pred == y_test).mean())
            try:
                bacc = float(balanced_accuracy_score(y_test, y_pred))
            except Exception:
                bacc = float("nan")

            rows.append({
                "episode": ep, "checkpoint": checkpoint_tag,
                "n_realized_classes": n_realized,
                "minority_frac": minority_frac,
                "normalized_balance": normalized_balance,
                "accuracy": acc, "balanced_accuracy": bacc,
                "status": "ok",
            })
        except Exception as e:
            print(f"  [error] ep={ep}: {e}")
            rows.append({
                "episode": ep, "checkpoint": checkpoint_tag,
                "n_realized_classes": n_realized,
                "minority_frac": minority_frac,
                "normalized_balance": normalized_balance,
                "accuracy": float("nan"), "balanced_accuracy": float("nan"),
                "status": "error", "error": repr(e),
            })

        if (ep + 1) % 20 == 0:
            print(f"  [{ep + 1}/{args.eval_tasks}] done")

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.head(20).to_string(index=False))
    print(f"...\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
