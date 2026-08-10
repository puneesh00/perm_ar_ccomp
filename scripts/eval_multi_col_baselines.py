# scripts/eval_multi_col_baselines.py
"""
Standalone finite-context-oracle baselines (logistic/ridge, random forest,
XGBoost) for the multi-queried-column samplers: column_block, row_block,
label_feature. Deliberately NOT wired into train_synthetic.py's evaluate()
loop -- run this separately, once, and compare its numbers against the
eval/{sampler}/cat_acc and eval/{sampler}/num_mse already logged for a
trained model run.

Generalizes the existing target-only baselines (logreg128_acc etc. in
train_synthetic.py) to episodes with MULTIPLE queried columns sharing one
context/query row split. That split is real for column_block, row_block,
and label_feature (see sampling.py: each samples a fixed row_idx/col_idx
grid, then builds query_coords as make_grid_coords(query_rows, query_cols)
-- the full cartesian product of one fixed row set against one fixed column
set, so every query row in the episode shares the identical queried-column
set). random_cell has no such split (conditioning_mode="transductive",
cells scattered anywhere) and is deliberately not supported here -- see
_multi_context_query_features's docstring.

For each queried column: ALL queried columns (not just the one being
predicted) are excluded from that column's own feature set, so no queried
column leaks into predicting another. Classifier + accuracy for categorical
queried columns, regressor + MSE for numerical ones -- mirroring exactly how
the model's own typed_mse_ce_loss splits by type.

Usage (matches the standard eval config used throughout this project's runs):

  PYTHONPATH=src python scripts/eval_multi_col_baselines.py \
    --tabpfn-prior-type scm --tabpfn-layers-mu-max 1.0 --tabpfn-layers-max 2 \
    --tabpfn-hidden-mu-max 10.0 --fresh-n-rows 512 --n-cols 16 \
    --samplers column_block,row_block,label_feature \
    --eval-tasks 50 --eval-seed 999
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tab_completion.sampling import (
    CompletionTask,
    ColumnBlockSampler,
    RowBlockSampler,
    LabelFeatureSampler,
)
from tab_completion.model import NUMERICAL, CATEGORICAL
from tab_completion.synthetic_data import FullSyntheticTable
from tab_completion.synthetic_data_tabpfn import TabPFNSCMConfig, TabPFNSCMTableGenerator


def _multi_context_query_features(
    full: FullSyntheticTable,
    task: CompletionTask,
) -> Optional[List[dict]]:
    """
    One dict per queried column: {"col", "is_cat", X_ctx, y_ctx, X_qry, y_qry}.
    Returns None if the task has no context_rows_local/query_rows_local
    (no row-level context/query split to fit a classical model on) or zero
    queried columns.
    """
    try:
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        return None

    context_rows_local = task.meta.get("context_rows_local")
    query_rows_local = task.meta.get("query_rows_local")
    if context_rows_local is None or query_rows_local is None:
        return None

    global_rows = task.row_idx
    local_cols = task.col_idx.tolist()

    queried_local = np.where(task.query_mask.any(axis=0))[0]
    if len(queried_local) == 0:
        return None
    queried_global = sorted({int(local_cols[i]) for i in queried_local})
    queried_set = set(queried_global)

    feature_cols = [c for c in local_cols if c not in queried_set]

    parts = []
    for j in feature_cols:
        if full.col_types[j] == NUMERICAL:
            parts.append(full.x_num[global_rows, j][:, None])
        else:
            card = int(full.cat_cardinalities[j])
            enc = OneHotEncoder(sparse_output=False, categories=[list(range(card))])
            parts.append(enc.fit_transform(full.x_cat[global_rows, j][:, None]))
    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(global_rows), 0))

    out = []
    for c in queried_global:
        is_cat = full.col_types[c] == CATEGORICAL
        y = full.x_cat[global_rows, c] if is_cat else full.x_num[global_rows, c]
        out.append({
            "col": c,
            "is_cat": bool(is_cat),
            "X_ctx": X[context_rows_local], "y_ctx": y[context_rows_local],
            "X_qry": X[query_rows_local], "y_qry": y[query_rows_local],
        })
    return out


def _build_families():
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    families = {
        "logreg128": (
            lambda: LogisticRegression(max_iter=1000),
            lambda: Ridge(),
        ),
        "rf128": (
            lambda: RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0),
            lambda: RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0),
        ),
    }
    try:
        from xgboost import XGBClassifier, XGBRegressor
        families["xgb128"] = (
            lambda: XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                n_jobs=-1, verbosity=0, random_state=0,
            ),
            lambda: XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                n_jobs=-1, verbosity=0, random_state=0,
            ),
        )
    except ImportError:
        pass
    return families


def multi_col_baseline_metrics(full: FullSyntheticTable, task: CompletionTask, families) -> Dict[str, float]:
    cols = _multi_context_query_features(full, task)
    if not cols:
        return {}

    out: Dict[str, float] = {}
    for name, (clf_factory, reg_factory) in families.items():
        cat_correct, cat_total = 0.0, 0
        num_sq_err, num_total = 0.0, 0

        for col in cols:
            X_ctx, y_ctx, X_qry, y_qry = col["X_ctx"], col["y_ctx"], col["X_qry"], col["y_qry"]
            if col["is_cat"]:
                if len(np.unique(y_ctx)) < 2:
                    pred = np.full_like(y_qry, y_ctx[0])
                else:
                    # Remap to dense 0..K'-1 codes before fitting: with only
                    # ~128 context rows, a high-cardinality column can easily
                    # be missing some category values in context, and
                    # XGBClassifier (unlike sklearn's classifiers) requires
                    # contiguous 0..K-1 labels with no gaps -- errors
                    # otherwise. Harmless no-op for logreg/rf, which handle
                    # arbitrary label sets fine either way.
                    classes, y_ctx_codes = np.unique(y_ctx, return_inverse=True)
                    clf = clf_factory()
                    clf.fit(X_ctx, y_ctx_codes)
                    pred_codes = clf.predict(X_qry)
                    pred = classes[pred_codes]
                cat_correct += float((pred == y_qry).sum())
                cat_total += len(y_qry)
            else:
                reg = reg_factory()
                reg.fit(X_ctx, y_ctx)
                pred = reg.predict(X_qry)
                num_sq_err += float(((pred - y_qry) ** 2).sum())
                num_total += len(y_qry)

        if cat_total > 0:
            out[f"{name}_cat_acc"] = cat_correct / cat_total
            out[f"{name}_cat_cells"] = float(cat_total)
        if num_total > 0:
            out[f"{name}_num_mse"] = num_sq_err / num_total
            out[f"{name}_num_cells"] = float(num_total)

    return out


def run_sampler(name: str, sampler, args, families) -> Dict[str, float]:
    table_gen = TabPFNSCMTableGenerator(
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
    rng = np.random.default_rng(args.eval_seed)

    values_by_metric: Dict[str, list] = {}
    n_skipped = 0
    for _ in range(args.eval_tasks):
        full = table_gen.sample_table()
        info = full.table_info()
        task = sampler.sample(info, rng)
        metrics = multi_col_baseline_metrics(full, task, families)
        if not metrics:
            n_skipped += 1
            continue
        for key, value in metrics.items():
            values_by_metric.setdefault(key, []).append(value)

    cell_weighted = {}
    for fam in families:
        cell_weighted[f"{fam}_cat_acc"] = f"{fam}_cat_cells"
        cell_weighted[f"{fam}_num_mse"] = f"{fam}_num_cells"

    out: Dict[str, float] = {}
    for key, values in values_by_metric.items():
        weight_key = cell_weighted.get(key)
        weights = values_by_metric.get(weight_key) if weight_key else None
        if weights is not None:
            total_weight = sum(weights)
            out[key] = (sum(v * w for v, w in zip(values, weights)) / total_weight) if total_weight > 0 else 0.0
        elif not key.endswith("_cells"):
            out[key] = float(np.mean(values))
    if n_skipped:
        print(f"  [{name}] skipped {n_skipped}/{args.eval_tasks} episodes (no queried columns)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tabpfn-prior-type", type=str, default="scm", choices=["scm", "bnn", "mixed"])
    p.add_argument("--tabpfn-layers-mu-max", type=float, default=6.0)
    p.add_argument("--tabpfn-layers-max", type=int, default=None)
    p.add_argument("--tabpfn-hidden-mu-max", type=float, default=130.0)
    p.add_argument("--fresh-n-rows", type=int, default=512)
    p.add_argument("--n-cols", type=int, default=16)
    p.add_argument("--p-categorical", type=float, default=0.2)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--target-col", type=int, default=None)

    p.add_argument("--samplers", type=str, default="column_block,row_block,label_feature")
    p.add_argument("--eval-tasks", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=999)

    p.add_argument("--column-n-context", type=int, default=128)
    p.add_argument("--column-n-query", type=int, default=128)
    p.add_argument("--column-min-query-cols", type=int, default=1)
    p.add_argument("--column-max-query-cols", type=int, default=1)

    p.add_argument("--row-n-context", type=int, default=128)
    p.add_argument("--row-n-query", type=int, default=1)
    p.add_argument("--row-query-frac-cols", type=float, default=0.5)

    p.add_argument("--label-feature-n-context", type=int, default=128)
    p.add_argument("--label-feature-n-query", type=int, default=1)
    p.add_argument("--label-feature-n-feature-cols", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    names = [s.strip() for s in args.samplers.split(",") if s.strip()]

    registry = {
        "column_block": ColumnBlockSampler(
            n_context=args.column_n_context,
            n_query=args.column_n_query,
            min_query_cols=args.column_min_query_cols,
            max_query_cols=args.column_max_query_cols,
        ),
        "row_block": RowBlockSampler(
            n_context=args.row_n_context,
            n_query=args.row_n_query,
            query_frac_cols=args.row_query_frac_cols,
        ),
        "label_feature": LabelFeatureSampler(
            n_context=args.label_feature_n_context,
            n_query=args.label_feature_n_query,
            n_feature_cols=args.label_feature_n_feature_cols,
            target_col=args.target_col,
        ),
    }

    families = _build_families()
    print(f"baseline families: {list(families.keys())}")
    print(f"eval_tasks={args.eval_tasks} eval_seed={args.eval_seed}")
    print()

    for name in names:
        if name not in registry:
            print(f"skipping unknown sampler {name!r} (supported: {list(registry.keys())})")
            continue
        print(f"=== {name} ===")
        results = run_sampler(name, registry[name], args, families)
        for key in sorted(results):
            print(f"  {key}: {results[key]:.4f}")
        print()


if __name__ == "__main__":
    main()
