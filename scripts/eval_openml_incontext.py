# scripts/eval_openml_incontext.py
"""
Evaluates trained checkpoints (TabPFN-v1 reference baseline, single_stream,
two_stream_ar, two_stream_ar_sparse) as in-context classifiers on real
OpenML-CC18 datasets, TabPFN-v1-paper style: predict one held-out target
column from a context of other rows, no gradient updates, compared against
tree-based baselines (RF/HGB/XGB) given a light hyperparameter search (our
model gets none -- that's the whole point of the comparison).

Reuses scripts/run_openml_baselines.py's dataset loading, preprocessing, and
baseline-model builders, and scripts/train_tabpfn_v1_baseline.py's build_xy
for the TabPFN-v1 model family -- see those files for the underlying
mechanics. This script's only new piece is the OpenML-DataFrame ->
FullSyntheticTable/CompletionTask converter (no such converter existed
before; every existing table source was synthetic) and the checkpoint
loading/dispatch.

Output is logged in exactly the same JSONL/CSV schema as
run_openml_baselines.py (suite, task_id, task_name, task_type, model,
n_train, n_test, n_features, n_classes, status, accuracy, balanced_accuracy,
log_loss, fit_predict_sec), so scripts/summarize_openml_baselines.py works
unchanged on the output -- our checkpoint(s) and the baselines end up ranked
together, per task, with zero changes to that script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.base import clone
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from run_openml_baselines import (  # noqa: E402
    get_suite_task_ids,
    load_openml_task,
    infer_column_types,
    make_tree_preprocessor,
    build_classification_models,
    classification_metrics,
    JSONLLogger,
    write_csv_from_jsonl,
)
from train_tabpfn_v1_baseline import build_xy  # noqa: E402
from tab_completion.model import NUMERICAL, CATEGORICAL, ModelConfig  # noqa: E402
from tab_completion.model_tabpfn_v1 import TabPFNV1Config, TabPFNV1Model  # noqa: E402
from tab_completion.model_single_stream import SingleStreamModel  # noqa: E402
from tab_completion.model_perm_ar import (  # noqa: E402
    PermARCompletionModel,
    build_rank_tensor,
    get_context_row_mask_from_task,
)
from tab_completion.model_perm_ar_sparse import (  # noqa: E402
    PermARCompletionModel as PermARCompletionModelSparse,
)
from tab_completion.sampling import CompletionTask  # noqa: E402
from tab_completion.factorization import ParallelFactorizer  # noqa: E402
from tab_completion.synthetic_data import FullSyntheticTable  # noqa: E402
from tab_completion.episode_utils import task_to_torch_batch  # noqa: E402


# ---------------------------------------------------------------------
# Checkpoint loading (auto-detects TabPFN-v1 vs single_stream/two_stream
# family from the checkpoint's own contents -- see plan doc for the exact
# distinguishing keys).
# ---------------------------------------------------------------------


class LoadedCheckpoint:
    def __init__(
        self,
        tag: str,
        family: str,
        model: torch.nn.Module,
        class_cap: int,
        feature_cardinality_cap: Optional[int],
        step: int,
    ):
        self.tag = tag
        self.family = family  # "tabpfn_v1" | "single_stream" | "two_stream_ar" | "two_stream_ar_sparse"
        self.model = model
        self.class_cap = class_cap
        self.feature_cardinality_cap = feature_cardinality_cap
        self.step = step


def load_checkpoint(path: str, tag: str, device: torch.device) -> LoadedCheckpoint:
    ckpt = torch.load(path, map_location=device)
    args = ckpt.get("args", {})

    if "model_cfg" in ckpt:
        # single_stream / two_stream_ar / two_stream_ar_sparse family.
        model_cfg = ModelConfig(**ckpt["model_cfg"])
        architecture = args.get("architecture", "single_stream")

        if architecture == "single_stream":
            model = SingleStreamModel(model_cfg)
        elif architecture == "two_stream_ar":
            model = PermARCompletionModel(model_cfg)
        elif architecture == "two_stream_ar_sparse":
            model = PermARCompletionModelSparse(model_cfg)
        else:
            raise ValueError(
                f"Unsupported architecture {architecture!r} in checkpoint {path}."
            )

        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        return LoadedCheckpoint(
            tag=tag,
            family=architecture,
            model=model,
            class_cap=model_cfg.k_max,
            feature_cardinality_cap=model_cfg.k_max,
            step=ckpt.get("step", -1),
        )

    # TabPFN-v1 reference baseline family (scripts/train_tabpfn_v1_baseline.py).
    cfg = TabPFNV1Config(
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        mlp_hidden=args["mlp_hidden"],
        n_layers=args["n_layers"],
        max_num_classes=args["max_num_classes"],
        dropout=args.get("dropout", 0.0),
    )
    model = TabPFNV1Model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return LoadedCheckpoint(
        tag=tag,
        family="tabpfn_v1",
        model=model,
        class_cap=args["max_num_classes"],
        feature_cardinality_cap=None,  # TabPFN-v1 casts categorical features to float, no cap.
        step=ckpt.get("step", -1),
    )


def parse_checkpoints_arg(spec: str) -> List[Tuple[str, str]]:
    """'path1=tag1,path2=tag2' -> [(path1, tag1), (path2, tag2)]. Tag defaults
    to the checkpoint's run-directory name if omitted."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            path, tag = item.split("=", 1)
        else:
            path, tag = item, Path(item).resolve().parent.name
        out.append((path.strip(), tag.strip()))
    return out


# ---------------------------------------------------------------------
# OpenML DataFrame -> FullSyntheticTable + hand-built CompletionTask.
#
# No sampler is used: task.row_idx alone determines which rows enter the
# episode and in what order (episode_utils.py), so a pre-determined
# context/query split is built directly instead of going through
# TargetPredictionSampler's random row selection.
# ---------------------------------------------------------------------


class ConvertedTable:
    def __init__(
        self,
        full: FullSyntheticTable,
        task: CompletionTask,
        n_context: int,
        n_query: int,
        target_col: int,
        y_query_true: np.ndarray,  # [n_query], plain LabelEncoder ids (0..n_classes-1)
        n_classes: int,
        max_feature_cardinality: int,
        X_train_df: pd.DataFrame,
        X_test_df: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        self.full = full
        self.task = task
        self.n_context = n_context
        self.n_query = n_query
        self.target_col = target_col
        self.y_query_true = y_query_true
        self.n_classes = n_classes
        self.max_feature_cardinality = max_feature_cardinality
        # Kept around so tree baselines fit on the EXACT same split.
        self.X_train_df = X_train_df
        self.X_test_df = X_test_df
        self.y_train = y_train
        self.y_test = y_test


def convert_openml_table(
    X: pd.DataFrame, y: pd.Series, rng: np.random.Generator, max_context: int, max_query: int
) -> ConvertedTable:
    n_rows = len(X)
    numeric_cols, categorical_cols = infer_column_types(X)
    n_features = X.shape[1]
    target_col = n_features  # target goes in the last column slot.
    n_cols = n_features + 1

    x_num = np.zeros((n_rows, n_cols), dtype=np.float32)
    x_cat = np.zeros((n_rows, n_cols), dtype=np.int64)
    col_types = np.full(n_cols, NUMERICAL, dtype=np.int64)
    cat_cardinalities = np.ones(n_cols, dtype=np.int64)
    cat_decode_types = np.arange(n_cols, dtype=np.int64)

    col_position = {col: i for i, col in enumerate(X.columns)}
    max_feature_cardinality = 1

    for col in numeric_cols:
        idx = col_position[col]
        x_num[:, idx] = X[col].to_numpy(dtype=np.float64)

    for col in categorical_cols:
        idx = col_position[col]
        encoded, uniques = pd.factorize(X[col])
        x_cat[:, idx] = encoded.astype(np.int64)
        col_types[idx] = CATEGORICAL
        cat_cardinalities[idx] = max(len(uniques), 1)
        max_feature_cardinality = max(max_feature_cardinality, len(uniques))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)
    x_cat[:, target_col] = y_encoded
    col_types[target_col] = CATEGORICAL
    cat_cardinalities[target_col] = n_classes

    full = FullSyntheticTable(
        x_num=x_num,
        x_cat=x_cat,
        col_types=col_types,
        cat_cardinalities=cat_cardinalities,
        cat_decode_types=cat_decode_types,
        target_col=target_col,
    )

    perm = rng.permutation(n_rows)
    n_context = min(n_rows // 2, max_context)
    n_query = min(n_rows - n_context, max_query)
    context_idx = perm[:n_context]
    query_idx = perm[n_context : n_context + n_query]

    row_idx = np.concatenate([context_idx, query_idx])
    col_idx = np.arange(n_cols, dtype=np.int64)
    n_ep = len(row_idx)

    observed_mask = np.ones((n_ep, n_cols), dtype=bool)
    query_mask = np.zeros((n_ep, n_cols), dtype=bool)
    observed_mask[n_context:, target_col] = False
    query_mask[n_context:, target_col] = True

    task = CompletionTask(
        row_idx=row_idx,
        col_idx=col_idx,
        observed_mask=observed_mask,
        query_mask=query_mask,
        task_name="openml_target",
        meta={
            "target_col": int(target_col),
            "n_context": int(n_context),
            "n_query": int(n_query),
            "conditioning_mode": "inductive_rows",
            "context_rows_local": np.arange(n_context, dtype=np.int64),
            "query_rows_local": np.arange(n_context, n_ep, dtype=np.int64),
        },
    )

    y_query_true = y_encoded[query_idx]

    return ConvertedTable(
        full=full,
        task=task,
        n_context=n_context,
        n_query=n_query,
        target_col=target_col,
        y_query_true=y_query_true,
        n_classes=n_classes,
        max_feature_cardinality=max_feature_cardinality,
        X_train_df=X.iloc[context_idx].copy(),
        X_test_df=X.iloc[query_idx].copy(),
        y_train=y_encoded[context_idx],
        y_test=y_encoded[query_idx],
    )


# ---------------------------------------------------------------------
# Prediction with a loaded checkpoint.
# ---------------------------------------------------------------------


@torch.no_grad()
def predict_tabpfn_v1(
    ckpt: LoadedCheckpoint, table: ConvertedTable, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_proba) with consistent label indexing
    (build_xy's per-episode densified label rank -- valid for accuracy/
    log_loss regardless of what the indices semantically mean, as long as
    y_true and y_proba use the same indexing, which they do here)."""
    x_feat, y_context, y_query, num_valid = build_xy(
        table.full, table.task, table.n_context, table.n_query
    )
    x_feat_t = torch.as_tensor(x_feat[None], dtype=torch.float32, device=device)
    y_ctx_t = torch.as_tensor(y_context[None], dtype=torch.float32, device=device)
    nvc_t = torch.as_tensor([num_valid], dtype=torch.long, device=device)

    logits = ckpt.model(x_feat_t, y_ctx_t, table.n_context, num_valid_classes=nvc_t)
    probs = F.softmax(logits[0, :, :num_valid], dim=-1).cpu().numpy()
    preds = probs.argmax(axis=-1)

    valid = y_query != -100
    return y_query[valid], preds[valid], probs[valid]


@torch.no_grad()
def predict_stream_family(
    ckpt: LoadedCheckpoint, table: ConvertedTable, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """single_stream / two_stream_ar / two_stream_ar_sparse. Target labels
    here are the plain LabelEncoder ids used throughout (no densification),
    so directly comparable to the tree baselines' label space."""
    batch = task_to_torch_batch(table.full, table.task, device)

    plan = ParallelFactorizer().build(table.task, np.random.default_rng(0))
    rank_np = build_rank_tensor(table.task, plan)
    rank_t = torch.as_tensor(rank_np[None], dtype=torch.long, device=device)

    context_row_mask_np = get_context_row_mask_from_task(table.task)
    context_row_mask_t = None
    if context_row_mask_np is not None:
        context_row_mask_t = torch.as_tensor(
            context_row_mask_np[None], dtype=torch.bool, device=device
        )

    if ckpt.family == "two_stream_ar_sparse":
        query_t = torch.as_tensor(table.task.query_mask[None], dtype=torch.bool, device=device)
        out = ckpt.model(batch, rank_t, context_row_mask=context_row_mask_t, prediction_mask=query_t)
    else:
        out = ckpt.model(batch, rank_t, context_row_mask=context_row_mask_t)

    query_rows_local = table.task.meta["query_rows_local"]
    logits = out.cat_logits[0, query_rows_local, table.target_col, :]  # [n_query, K_max]
    n_classes = table.n_classes
    probs = F.softmax(logits[:, :n_classes], dim=-1).cpu().numpy()
    preds = probs.argmax(axis=-1)

    return table.y_query_true, preds, probs


def predict_checkpoint(
    ckpt: LoadedCheckpoint, table: ConvertedTable, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ckpt.family == "tabpfn_v1":
        return predict_tabpfn_v1(ckpt, table, device)
    return predict_stream_family(ckpt, table, device)


# ---------------------------------------------------------------------
# Tree baselines with light tuning (RandomizedSearchCV on the context/train
# split only). Wraps build_classification_models's base estimators.
# ---------------------------------------------------------------------


PARAM_DISTRIBUTIONS = {
    "rf": {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "hgb": {
        "model__max_iter": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 1e-4, 1e-2],
    },
    "xgb": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
}


def build_tuned_pipelines(
    seed: int,
    n_jobs: int,
    model_names: List[str],
    n_classes: int,
    preprocessor,
    n_iter: int,
    cv_folds: int,
) -> Dict[str, Any]:
    base_models = build_classification_models(seed, n_jobs, model_names, n_classes)
    tuned: Dict[str, Any] = {}

    for name, base_model in base_models.items():
        pipe = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", clone(base_model))])
        param_dist = PARAM_DISTRIBUTIONS.get(name)
        if param_dist is None:
            tuned[name] = pipe
            continue
        tuned[name] = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            random_state=seed,
            n_jobs=1,  # avoid nested-parallelism thread oversubscription with the base model's own n_jobs
            error_score="raise",
        )

    return tuned


# ---------------------------------------------------------------------
# Per-task driver.
# ---------------------------------------------------------------------


def eligible_for_checkpoint(table: ConvertedTable, ckpt: LoadedCheckpoint) -> Optional[str]:
    if table.n_classes > ckpt.class_cap:
        return f"n_classes={table.n_classes} > class_cap={ckpt.class_cap} for {ckpt.tag}"
    if (
        ckpt.feature_cardinality_cap is not None
        and table.max_feature_cardinality > ckpt.feature_cardinality_cap
    ):
        return (
            f"max_feature_cardinality={table.max_feature_cardinality} > "
            f"cap={ckpt.feature_cardinality_cap} for {ckpt.tag}"
        )
    return None


def run_task(
    task_id: int,
    args,
    checkpoints: List[LoadedCheckpoint],
    device: torch.device,
    logger: JSONLLogger,
) -> None:
    rng = np.random.default_rng(args.seed + task_id)

    task, X, y_raw = load_openml_task(task_id)
    task_name = getattr(task, "name", None) or str(task_id)

    n_rows = len(X)
    n_features = X.shape[1]

    def log_dataset_skip(reason: str) -> None:
        print(f"[skip] task={task_id} name={task_name}: {reason}")
        logger.log(
            {
                "suite": args.suite,
                "task_id": task_id,
                "task_name": task_name,
                "task_type": "classification",
                "model": None,
                "n_features": n_features,
                "status": "skipped",
                "error": reason,
            }
        )

    if n_rows > args.max_total_rows or n_rows < args.min_rows:
        log_dataset_skip(f"n_rows={n_rows} outside [{args.min_rows},{args.max_total_rows}]")
        return
    if n_features > args.max_features:
        log_dataset_skip(f"n_features={n_features} > {args.max_features}")
        return
    if X.isna().any().any() or y_raw.isna().any():
        log_dataset_skip("contains missing values")
        return

    table = convert_openml_table(X, y_raw, rng, args.max_context, args.max_query)

    print(
        f"[task] id={task_id} name={task_name} n_rows={n_rows} n_features={n_features} "
        f"n_classes={table.n_classes} n_context={table.n_context} n_query={table.n_query}"
    )

    def log_row(model_name: str, extra: Dict[str, Any]) -> None:
        row = {
            "suite": args.suite,
            "task_id": task_id,
            "task_name": task_name,
            "task_type": "classification",
            "model": model_name,
            "n_train": table.n_context,
            "n_test": table.n_query,
            "n_features": n_features,
            "n_classes": table.n_classes,
            "status": "ok",
        }
        row.update(extra)
        logger.log(row)

    # --- Our checkpoint(s): no tuning, one forward pass each. ---
    for ckpt in checkpoints:
        skip_reason = eligible_for_checkpoint(table, ckpt)
        if skip_reason is not None:
            print(f"  [skip-model] {ckpt.tag}: {skip_reason}")
            log_row(ckpt.tag, {"status": "skipped", "error": skip_reason})
            continue

        start = time.perf_counter()
        try:
            y_true, y_pred, y_proba = predict_checkpoint(ckpt, table, device)
            metrics = classification_metrics(y_true, y_pred, y_proba)
            log_row(
                ckpt.tag,
                {
                    **metrics,
                    "fit_predict_sec": time.perf_counter() - start,
                    "n_query_scored": int(len(y_true)),
                },
            )
        except Exception as e:
            print(f"  [error] {ckpt.tag}: {e}")
            log_row(
                ckpt.tag,
                {
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "fit_predict_sec": time.perf_counter() - start,
                },
            )

    # --- Tree baselines, tuned, fit on the SAME split. ---
    preprocessor = make_tree_preprocessor(table.X_train_df)
    tuned_models = build_tuned_pipelines(
        args.seed,
        args.n_jobs,
        args.baseline_models.split(","),
        table.n_classes,
        preprocessor,
        args.baseline_tuning_iters,
        args.baseline_cv_folds,
    )

    for model_name, model in tuned_models.items():
        start = time.perf_counter()
        try:
            model.fit(table.X_train_df, table.y_train)
            y_pred = model.predict(table.X_test_df)
            y_proba = model.predict_proba(table.X_test_df) if hasattr(model, "predict_proba") else None
            metrics = classification_metrics(table.y_test, y_pred, y_proba)
            log_row(model_name, {**metrics, "fit_predict_sec": time.perf_counter() - start})
        except Exception as e:
            print(f"  [error] {model_name}: {e}")
            log_row(
                model_name,
                {
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "fit_predict_sec": time.perf_counter() - start,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="'path1=tag1,path2=tag2,...'")
    parser.add_argument("--suite", type=str, default="cc18", choices=["cc18"])
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)

    parser.add_argument("--max-total-rows", type=int, default=1024)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=63)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--max-query", type=int, default=512)

    parser.add_argument("--baseline-models", type=str, default="rf,hgb,xgb")
    parser.add_argument("--baseline-tuning-iters", type=int, default=25)
    parser.add_argument("--baseline-cv-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=4)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results/openml_incontext")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if device.type == "cpu" and args.device == "auto":
        device = torch.device("cpu")

    checkpoint_specs = parse_checkpoints_arg(args.checkpoints)
    checkpoints = [load_checkpoint(path, tag, device) for path, tag in checkpoint_specs]

    if args.run_name is None:
        timestamp = int(time.time())
        tags = "-".join(c.tag for c in checkpoints)
        args.run_name = f"{args.suite}_{tags}_{timestamp}"

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"
    logger = JSONLLogger(jsonl_path)

    if args.task_ids is not None and args.task_ids.strip():
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = get_suite_task_ids(args.suite)
        if args.max_tasks is not None:
            task_ids = task_ids[: args.max_tasks]

    print("=== OpenML in-context checkpoint evaluation ===")
    print(f"device={device}")
    print(f"checkpoints={[(c.tag, c.family, c.step) for c in checkpoints]}")
    print(f"suite={args.suite} num_tasks={task_ids and len(task_ids)}")
    print(f"out_dir={out_dir}")

    for idx, task_id in enumerate(task_ids):
        print(f"\n=== [{idx + 1}/{len(task_ids)}] task_id={task_id} ===")
        try:
            run_task(task_id, args, checkpoints, device, logger)
        except Exception as e:
            print(f"[error] task={task_id}: failed to load/run task: {e}")
            logger.log(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task_name": str(task_id),
                    "task_type": "classification",
                    "model": None,
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )
        write_csv_from_jsonl(jsonl_path, csv_path)

    print("\nDone.")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")
    print(
        f"Summarize with: python scripts/summarize_openml_baselines.py --input-csv {csv_path}"
    )


if __name__ == "__main__":
    main()
