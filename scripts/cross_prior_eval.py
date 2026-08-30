# scripts/cross_prior_eval.py
"""
Cross-prior diagnostic: evaluate one or more of our own checkpoints (and
optionally the official pretrained TabPFN v1 checkpoint) on tables drawn
from the REAL, released TabPFN-v1 causal-MLP prior generator
(tabpfn.priors.mlp + .flexible_categorical, shipped inside the pip
tabpfn==0.1.11 package -- not a reimplementation), to compare against the
same checkpoints' scores on our own synthetic prior (see
results/synthetic_incontext/synthetic_*_k1 for that half of the matrix).

Simplification (matches the review that motivated this script's own
suggested fallback): official-v1 categorical FEATURE conversion is disabled
(categorical_feature_p=0.0) so every generated table is all-numeric
features + one categorical target -- avoids needing to recover
column-type/cardinality metadata that official's own FlexibleCategorical
wrapper discards internally. Per-table hyperparameters (num_classes,
num_layers, hidden width, num_causes, noise_std, dropout) are jittered
across a plausible range each draw, approximating genuine prior diversity
rather than one fixed hyperparameter point -- NOT the real ConfigSpace-
sampled ihyperparameter distribution (ConfigSpace isn't installed; ours is
a hand-rolled approximation).

Requirements to run:
  - `tabpfn` (the v1 pip package, tabpfn==0.1.11) importable on this
    interpreter's path -- ANY environment with it pip-installed works, this
    script locates it dynamically rather than hardcoding a venv path.
  - our own `tab_completion` src package importable (PYTHONPATH=<repo>/src,
    or run from a venv that already has it on the path).
  - torch, numpy, pandas.
  - the official TabPFNClassifier eval leg (--eval-official-v1) additionally
    needs whatever GPU/CPU device you point --official-device at; skip it
    entirely if you only want to score your own checkpoints.

Example (score two of our own checkpoints, skip official v1):
    PYTHONPATH=src python scripts/cross_prior_eval.py \\
        --checkpoints "path/to/ckpt_a.pt=ckpt_a,path/to/ckpt_b.pt=ckpt_b" \\
        --n-tables 30 --out-csv results/cross_prior_mycheckpoints.csv

Example (also score official v1, needs the real tabpfn package + GPU):
    PYTHONPATH=src python scripts/cross_prior_eval.py \\
        --checkpoints "path/to/ckpt_a.pt=ckpt_a" \\
        --eval-official-v1 --official-device cuda \\
        --n-tables 30 --out-csv results/cross_prior_with_official.csv
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from eval_openml_incontext import load_checkpoint, predict_stream_family, ConvertedTable  # noqa: E402
from tab_completion.synthetic_data import FullSyntheticTable  # noqa: E402
from tab_completion.model import NUMERICAL, CATEGORICAL  # noqa: E402
from tab_completion.sampling import TargetPredictionSampler  # noqa: E402


def bypass_import_priors():
    """tabpfn/priors/__init__.py eagerly imports fast_gp -> gpytorch (often
    not installed, since it's only needed for the GP prior branch we don't
    use here). Register a stub package module so mlp.py/flexible_categorical.py's
    relative imports resolve without executing the real __init__.py.

    Locates the installed `tabpfn` package dynamically (importlib.util.find_spec)
    instead of hardcoding a venv path, so this works on any machine/environment
    that has tabpfn==0.1.11 pip-installed, not just this one."""
    tabpfn_spec = importlib.util.find_spec("tabpfn")
    if tabpfn_spec is None or not tabpfn_spec.submodule_search_locations:
        raise ImportError(
            "Could not locate an installed 'tabpfn' package on this "
            "interpreter's path. Install tabpfn==0.1.11 (pip install "
            "tabpfn==0.1.11) in whatever environment you run this script "
            "from -- it's needed to read the real v1 prior-generation code, "
            "even if you skip --eval-official-v1."
        )
    tabpfn_root = Path(list(tabpfn_spec.submodule_search_locations)[0])
    priors_dir = tabpfn_root / "priors"

    pkg_name = "tabpfn.priors"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(priors_dir / "__init__.py"),
        submodule_search_locations=[str(priors_dir)],
    )
    stub = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = stub
    mlp = importlib.import_module("tabpfn.priors.mlp")
    flexible_categorical = importlib.import_module("tabpfn.priors.flexible_categorical")
    return mlp, flexible_categorical


def generate_official_v1_table(mlp, flexible_categorical, rng, num_features, n_context, n_query):
    num_classes = int(rng.integers(2, 11))
    num_layers = int(rng.integers(2, 9))
    prior_mlp_hidden_dim = int(rng.integers(64, 200))
    num_causes = int(rng.integers(1, min(10, num_features) + 1))
    noise_std = float(rng.uniform(0.05, 0.3))
    dropout_prob = float(rng.uniform(0.0, 0.3))
    seq_len = n_context + n_query

    hp = dict(
        num_classes=num_classes, balanced=False, multiclass_type="rank", output_multiclass_ordered_p=0.5,
        nan_prob_no_reason=0.0, nan_prob_a_reason=0.0, nan_prob_unknown_reason=0.0,
        nan_prob_unknown_reason_reason_prior=0.5, set_value_to_nan=0.5, categorical_feature_p=0.0,
        normalize_to_ranking=False, normalize_by_used_features=True, num_features_used=num_features,
        check_is_compatible=True, normalize_labels=True, normalize_ignore_label_too=False,
        rotate_normalized_labels=True, seq_len_used=seq_len,
        num_layers=num_layers, is_causal=True, num_causes=num_causes, prior_mlp_hidden_dim=prior_mlp_hidden_dim,
        pre_sample_causes=True, noise_std=noise_std, pre_sample_weights=True,
        prior_mlp_activations=lambda: (lambda: torch.nn.Tanh()),
        block_wise_dropout=False, prior_mlp_dropout_prob=dropout_prob, prior_mlp_scale_weights_sqrt=True,
        init_std=1.0, new_mlp_per_example=True, mix_activations=False, sampling="normal",
        y_is_effect=True, in_clique=False, sort_features=False, random_feature_rotation=True, verbose=False,
    )
    x, y, _ = flexible_categorical.get_batch(
        batch_size=1, seq_len=seq_len, num_features=num_features,
        get_batch=mlp.get_batch, device="cpu", hyperparameters=hp, single_eval_pos=n_context,
    )
    x_np = x[:, 0, :].detach().numpy().astype(np.float32)
    y_np = y[:, 0].detach().numpy()
    return x_np, y_np, num_classes


def build_full_synthetic_table(x_np, y_np, num_classes):
    n_rows, num_features = x_np.shape
    n_cols = num_features + 1
    target_col = num_features
    x_num = np.zeros((n_rows, n_cols), dtype=np.float32)
    x_num[:, :num_features] = x_np
    x_cat = np.zeros((n_rows, n_cols), dtype=np.int64)
    x_cat[:, target_col] = y_np.astype(np.int64)
    col_types = np.full(n_cols, NUMERICAL, dtype=np.int64)
    col_types[target_col] = CATEGORICAL
    cat_cardinalities = np.ones(n_cols, dtype=np.int64)
    cat_cardinalities[target_col] = num_classes
    cat_decode_types = np.arange(n_cols, dtype=np.int64)
    full = FullSyntheticTable(
        x_num=x_num, x_cat=x_cat, col_types=col_types,
        cat_cardinalities=cat_cardinalities, cat_decode_types=cat_decode_types,
        target_col=target_col,
    )
    return full, target_col


def context_balance_metrics(y_np, n_context, num_classes):
    """Mirrors the normalized_balance convention used in
    results/synthetic_incontext/synthetic_context_imbalance_*.csv (0 =
    maximally imbalanced realized-class split, 1 = perfectly balanced)."""
    ctx = y_np[:n_context].astype(np.int64)
    counts = np.bincount(ctx[ctx >= 0], minlength=num_classes).astype(np.float64)
    counts = counts[counts > 0]
    n_realized = len(counts)
    minority_frac = float(counts.min() / counts.sum())
    if n_realized <= 1:
        normalized_balance = 0.0
    else:
        uniform_minority = 1.0 / n_realized
        normalized_balance = float(minority_frac / uniform_minority)
    return n_realized, minority_frac, normalized_balance


def eval_ours(ckpt, full, target_col, n_context, n_query, num_classes, device, seed):
    info = full.table_info()
    task = TargetPredictionSampler(
        n_context=n_context, n_query=n_query, target_col=target_col
    ).sample(info, np.random.default_rng(seed))
    query_rows_global = task.row_idx[n_context:n_context + n_query]
    y_query_true = full.x_cat[query_rows_global, target_col]

    table = ConvertedTable(
        full=full, task=task, n_context=n_context, n_query=n_query, target_col=target_col,
        y_query_true=y_query_true, n_classes=num_classes, max_feature_cardinality=1,
        X_train_df=pd.DataFrame(), X_test_df=pd.DataFrame(),
        y_train=np.array([]), y_test=np.array([]),
    )
    y_true, preds, _ = predict_stream_family(ckpt, table, device)
    acc = float((preds == y_true).mean())
    bal_acc = float(np.mean([
        (preds[y_true == c] == c).mean() if np.any(y_true == c) else np.nan
        for c in np.unique(y_true)
    ]))
    return acc, bal_acc


def eval_official_v1(clf, x_np, y_np, n_context):
    X_context = x_np[:n_context].astype(np.float64)
    y_context = y_np[:n_context].astype(np.int64)
    X_query = x_np[n_context:].astype(np.float64)
    y_query = y_np[n_context:].astype(np.int64)
    if len(np.unique(y_context)) < 2:
        return None, None
    clf.fit(X_context, y_context)
    proba = clf.predict_proba(X_query)
    preds = clf.classes_[proba.argmax(axis=1)]
    acc = float((preds == y_query).mean())
    bal_acc = float(np.mean([
        (preds[y_query == c] == c).mean() if np.any(y_query == c) else np.nan
        for c in np.unique(y_query)
    ]))
    return acc, bal_acc


def parse_checkpoints(spec: str):
    """'path1=tag1,path2=tag2' -> [(path1, tag1), (path2, tag2)]. A bare
    'path' with no '=tag' uses the checkpoint's filename stem as the tag."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            path, tag = part.split("=", 1)
        else:
            path, tag = part, Path(part).stem
        out.append((path.strip(), tag.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tables", type=int, default=30)
    ap.add_argument("--n-context", type=int, default=512)
    ap.add_argument("--n-query", type=int, default=64)
    ap.add_argument("--num-features", type=int, default=64)
    ap.add_argument(
        "--checkpoints", type=str, required=True,
        help="comma-separated path=tag pairs, e.g. 'ckpt_a.pt=ours_a,ckpt_b.pt=ours_b'",
    )
    ap.add_argument("--device", type=str, default="cpu", help="device for our own checkpoints")
    ap.add_argument(
        "--eval-official-v1", action="store_true",
        help="also fit/score the official pretrained TabPFNClassifier v1 on every table "
             "(needs the real tabpfn package's sklearn-style interface -- same package "
             "already required for prior generation, so no extra install if that already works)",
    )
    ap.add_argument("--official-device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out-csv", type=str, required=True)
    args = ap.parse_args()

    mlp, flexible_categorical = bypass_import_priors()

    device = torch.device(args.device)
    checkpoints = parse_checkpoints(args.checkpoints)
    ckpts = [(tag, load_checkpoint(path, tag, device)) for path, tag in checkpoints]
    print(f"loaded {len(ckpts)} checkpoint(s): {[t for t, _ in ckpts]}")

    clf = None
    if args.eval_official_v1:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(
            device=args.official_device, seed=args.seed, N_ensemble_configurations=1,
            no_preprocess_mode=True, feature_shift_decoder=False, multiclass_decoder="none",
        )

    rng = np.random.default_rng(args.seed)
    rows = []
    t_start = time.time()
    for i in range(args.n_tables):
        t0 = time.time()
        x_np, y_np, num_classes = generate_official_v1_table(
            mlp, flexible_categorical, rng, args.num_features, args.n_context, args.n_query
        )
        t_gen = time.time() - t0

        if len(np.unique(y_np[: args.n_context])) < 2:
            print(f"[{i}] skip: <2 classes realized in context")
            continue

        n_realized, minority_frac, normalized_balance = context_balance_metrics(
            y_np, args.n_context, num_classes
        )
        full, target_col = build_full_synthetic_table(x_np, y_np, num_classes)

        row = dict(
            i=i, num_classes=num_classes, gen_seconds=t_gen,
            n_realized_classes=n_realized, minority_frac=minority_frac,
            normalized_balance=normalized_balance,
        )
        msg = f"[{i+1}/{args.n_tables}] num_classes={num_classes} gen={t_gen:.1f}s"

        for tag, ckpt in ckpts:
            acc, bal_acc = eval_ours(
                ckpt, full, target_col, args.n_context, args.n_query, num_classes,
                device, seed=args.seed + i,
            )
            row[f"acc_{tag}"] = acc
            row[f"bal_acc_{tag}"] = bal_acc
            msg += f" acc_{tag}={acc:.4f}"

        if clf is not None:
            acc_o, bal_acc_o = eval_official_v1(clf, x_np, y_np, args.n_context)
            row["acc_official_v1"] = acc_o
            row["bal_acc_official_v1"] = bal_acc_o
            msg += f" acc_official_v1={acc_o}"

        elapsed = time.time() - t_start
        print(msg + f" (elapsed={elapsed:.0f}s)")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\n=== AGGREGATE (p_official-v1 prior) ===")
    print(f"n_tables_scored={len(df)} / {args.n_tables}")
    for tag, _ in ckpts:
        print(f"acc_{tag:30s} mean={df[f'acc_{tag}'].mean():.4f}  bal_acc mean={df[f'bal_acc_{tag}'].mean():.4f}")
    if clf is not None:
        ok = df.dropna(subset=["acc_official_v1"])
        print(f"acc_official_v1{'':16s} mean={ok['acc_official_v1'].mean():.4f}  bal_acc mean={ok['bal_acc_official_v1'].mean():.4f}  (n={len(ok)})")
    print(f"total wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
