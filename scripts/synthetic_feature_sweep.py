# scripts/synthetic_feature_sweep.py
"""
Structural sweep: for a fixed n_context (well inside the trained
--variable-table-shape range of 64-512, per the write-up this script was
built to check), vary n_cols across an explicit grid spanning both sides of
training's n_cols floor (16, i.e. 15 features), and record per-episode
context class-balance alongside accuracy/balanced_accuracy -- so results can
be sliced two ways at once: fixed imbalance bracket, features swept within
it (or the reverse).

One backend per invocation (--backend ours|v1|v2):
  ours: this repo's own checkpoint family, via eval_openml_incontext.py's
        load_checkpoint / predict_stream_family_parallel_multi (run from the
        main .venv).
  v1/v2: the officially pip-installed tabpfn package, via
        eval_synthetic_official_tabpfn.py's build_model / flatten_xy /
        padded_proba (run from .venv_tabpfn_v1 / .venv_tabpfn_v2
        respectively -- see that module's docstring for why they can't
        share an environment).

A fresh TabPFNSCMTableGenerator is constructed per n_cols grid point (same
--eval-seed each time), and row_rng/col_rng are reset per grid point too --
so for a given n_cols, the sequence of episodes is identical across
--backend ours/v1/v2 invocations, same reproducibility contract as
eval_synthetic_official_tabpfn.py / synthetic_imbalance.py.

Usage:
    python scripts/synthetic_feature_sweep.py \\
        --backend ours --checkpoint path.pt \\
        --n-cols-grid 5,9,17,25,33,49,64 --episodes-per-point 80 \\
        --n-context 512 --eval-seed 999 --device cuda \\
        --out-csv results/synthetic_incontext/feature_sweep_ours.csv
"""

from __future__ import annotations

import argparse
import copy
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
    ConvertedTableMulti,
    predict_stream_family_parallel_multi,
)
from eval_synthetic_official_tabpfn import build_model, flatten_xy, padded_proba  # noqa: E402


def context_balance(full, task, n_context):
    y_context = full.x_cat[task.row_idx[:n_context], full.target_col]
    counts = np.bincount(y_context)
    counts = counts[counts > 0]
    n_realized = len(counts)
    minority_frac = 0.0 if n_realized == 1 else counts.min() / counts.sum()
    return n_realized, minority_frac, minority_frac * n_realized


def run_ours(args, n_cols_grid, device):
    ckpt = load_checkpoint(args.checkpoint, "ours", device)
    rows = []
    for n_cols in n_cols_grid:
        cfg_args = copy.copy(args)
        cfg_args.n_cols = n_cols
        table_gen = build_table_generator(cfg_args)
        row_rng = np.random.default_rng(args.eval_seed)
        col_rng = np.random.default_rng([1, args.eval_seed])

        for ep in range(args.episodes_per_point):
            full = table_gen.sample_table()
            task = build_task(full, 1, args.n_context, args.n_query, row_rng, col_rng, False, True)
            n_realized, minority_frac, normalized_balance = context_balance(full, task, args.n_context)

            table = ConvertedTableMulti(
                full=full, task=task, n_context=args.n_context, n_query=args.n_query,
                query_cols=np.array(task.meta["query_cols"]), n_classes=0,
                max_feature_cardinality=0, conditioning_mode="inductive_rows",
            )
            pred = predict_stream_family_parallel_multi(ckpt, table, device)
            acc = float((pred.y_true == pred.y_pred).mean())
            try:
                bacc = float(balanced_accuracy_score(pred.y_true, pred.y_pred))
            except Exception:
                bacc = float("nan")

            rows.append({
                "n_cols": n_cols, "n_features": n_cols - 1, "episode": ep,
                "n_realized_classes": n_realized, "minority_frac": minority_frac,
                "normalized_balance": normalized_balance, "accuracy": acc, "balanced_accuracy": bacc,
            })
        print(f"  [n_cols={n_cols}] done ({args.episodes_per_point} episodes)")
    return rows


def run_official(args, n_cols_grid, version, device):
    rows = []
    for n_cols in n_cols_grid:
        cfg_args = copy.copy(args)
        cfg_args.n_cols = n_cols
        table_gen = build_table_generator(cfg_args)
        row_rng = np.random.default_rng(args.eval_seed)
        col_rng = np.random.default_rng([1, args.eval_seed])

        for ep in range(args.episodes_per_point):
            full = table_gen.sample_table()
            task = build_task(full, 1, args.n_context, args.n_query, row_rng, col_rng, False, True)
            n_realized, minority_frac, normalized_balance = context_balance(full, task, args.n_context)

            X_train, y_train, X_test, y_test = flatten_xy(full, task)
            try:
                clf = build_model(version, args.device, args.seed, args.n_estimators, args.no_tricks)
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
                    "n_cols": n_cols, "n_features": n_cols - 1, "episode": ep,
                    "n_realized_classes": n_realized, "minority_frac": minority_frac,
                    "normalized_balance": normalized_balance, "accuracy": acc, "balanced_accuracy": bacc,
                    "status": "ok",
                })
            except Exception as e:
                rows.append({
                    "n_cols": n_cols, "n_features": n_cols - 1, "episode": ep,
                    "n_realized_classes": n_realized, "minority_frac": minority_frac,
                    "normalized_balance": normalized_balance, "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"), "status": "error", "error": repr(e),
                })
        print(f"  [n_cols={n_cols}] done ({args.episodes_per_point} episodes)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["ours", "v1", "v2"])
    parser.add_argument("--checkpoint", type=str, default=None, help="required for --backend ours")
    parser.add_argument("--n-cols-grid", type=str, required=True, help="comma-separated total column counts (features+1)")
    parser.add_argument("--episodes-per-point", type=int, default=80)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--n-context", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=64)

    parser.add_argument("--fresh-n-rows", type=int, default=576)
    parser.add_argument("--n-cols", type=int, default=64)  # overwritten per grid point
    parser.add_argument("--p-categorical", type=float, default=0.3)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--tabpfn-prior-type", type=str, default="scm")
    parser.add_argument("--tabpfn-layers-mu-max", type=float, default=6.0)
    parser.add_argument("--tabpfn-layers-max", type=int, default=None)
    parser.add_argument("--tabpfn-hidden-mu-max", type=float, default=130.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=1,
                         help="official-checkpoint (v1/v2) ensemble size / N_ensemble_configurations. "
                              "Default 1 (single forward pass, matching --backend ours). Ignored for "
                              "--backend ours or when --no-tricks is set.")
    parser.add_argument("--no-tricks", action="store_true",
                         help="v1/v2 only: single forward pass with every inference-time augmentation "
                              "disabled (no feature/class shift, no preprocessing-transform ensemble, "
                              "no fingerprint feature, no outlier clipping, no probability balancing, "
                              "no temperature sharpening) -- see eval_synthetic_official_tabpfn."
                              "build_model's docstring. Ignored for --backend ours.")
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    n_cols_grid = [int(x) for x in args.n_cols_grid.split(",")]
    device = torch.device(args.device)

    print(f"=== Synthetic feature-count sweep: backend={args.backend} ===")
    print(f"n_cols_grid={n_cols_grid} (features={[c-1 for c in n_cols_grid]})")
    print(f"n_context={args.n_context} (fixed) episodes_per_point={args.episodes_per_point}")

    if args.backend == "ours":
        assert args.checkpoint, "--checkpoint required for --backend ours"
        rows = run_ours(args, n_cols_grid, device)
    else:
        rows = run_official(args, n_cols_grid, args.backend, device)

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
