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
from dataclasses import dataclass
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
from threadpoolctl import threadpool_limits

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
from eval_multi_col_baselines import (  # noqa: E402
    _build_families as build_multi_col_baseline_families,
    _multi_context_query_features as multi_context_query_features,
)
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
from tab_completion.ar_generate import generate_ar, numeric_context_stats  # noqa: E402
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


def _build_full_table_from_df(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[FullSyntheticTable, np.ndarray, int, int, int, int]:
    """Shared by convert_openml_table and convert_openml_table_multi. Returns
    (full, y_encoded, n_classes, target_col, max_feature_cardinality, n_cols)."""
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
    return full, y_encoded, n_classes, target_col, max_feature_cardinality, n_cols


def convert_openml_table(
    X: pd.DataFrame, y: pd.Series, rng: np.random.Generator, max_context: int, max_query: int
) -> ConvertedTable:
    full, y_encoded, n_classes, target_col, max_feature_cardinality, n_cols = (
        _build_full_table_from_df(X, y)
    )
    n_rows = len(X)

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
# Multi-target-column variant (regime 2): hold out the SAME k>1 columns for
# every query row, matching ColumnBlockSampler's synthetic-data task shape.
# Unlike convert_openml_table, this can build the task under either
# conditioning_mode -- "inductive_rows" (context rows guaranteed clean,
# query rows isolated from each other) or "transductive" (no context
# guarantee passed to the model, matching what checkpoints trained
# exclusively on random_cell have actually seen). context_rows_local/
# query_rows_local are set either way purely so tree baselines always have a
# well-defined fit/score split -- get_context_row_mask_from_task only reads
# conditioning_mode for the "transductive" branch (returns None regardless
# of whether those keys are present), so this doesn't leak context into the
# model under transductive.
# ---------------------------------------------------------------------


class ConvertedTableMulti:
    def __init__(
        self,
        full: FullSyntheticTable,
        task: CompletionTask,
        n_context: int,
        n_query: int,
        query_cols: np.ndarray,
        n_classes: int,
        max_feature_cardinality: int,
        conditioning_mode: str,
    ):
        self.full = full
        self.task = task
        self.n_context = n_context
        self.n_query = n_query
        self.query_cols = query_cols
        self.n_classes = n_classes
        self.max_feature_cardinality = max_feature_cardinality
        self.conditioning_mode = conditioning_mode


def convert_openml_table_multi(
    X: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
    query_cols: np.ndarray,
    max_context: int,
    max_query: int,
    conditioning_mode: str = "inductive_rows",
    query_frac: float = 1.0,
) -> ConvertedTableMulti:
    full, _y_encoded, n_classes, _target_col, max_feature_cardinality, n_cols = (
        _build_full_table_from_df(X, y)
    )
    n_rows = len(X)

    perm = rng.permutation(n_rows)
    n_context = min(n_rows // 2, max_context)
    n_query = min(n_rows - n_context, max_query)
    if query_frac < 1.0:
        # Cuts cost by scoring fewer query rows per target-set -- deliberately
        # applied after n_query is fixed (not by shrinking max_query), so it
        # only trims the query side and leaves the context/query row split
        # itself, and thus n_context, untouched.
        n_query = max(1, int(round(n_query * query_frac)))
    context_idx = perm[:n_context]
    query_idx = perm[n_context : n_context + n_query]

    row_idx = np.concatenate([context_idx, query_idx])
    col_idx = np.arange(n_cols, dtype=np.int64)
    n_ep = len(row_idx)

    observed_mask = np.ones((n_ep, n_cols), dtype=bool)
    query_mask = np.zeros((n_ep, n_cols), dtype=bool)
    query_rows = np.arange(n_context, n_ep)
    observed_mask[np.ix_(query_rows, query_cols)] = False
    query_mask[np.ix_(query_rows, query_cols)] = True

    task = CompletionTask(
        row_idx=row_idx,
        col_idx=col_idx,
        observed_mask=observed_mask,
        query_mask=query_mask,
        task_name="openml_multi_target",
        meta={
            "query_cols": query_cols.tolist(),
            "n_context": int(n_context),
            "n_query": int(n_query),
            "conditioning_mode": conditioning_mode,
            "context_rows_local": np.arange(n_context, dtype=np.int64),
            "query_rows_local": np.arange(n_context, n_ep, dtype=np.int64),
        },
    )

    return ConvertedTableMulti(
        full=full,
        task=task,
        n_context=n_context,
        n_query=n_query,
        query_cols=query_cols,
        n_classes=n_classes,
        max_feature_cardinality=max_feature_cardinality,
        conditioning_mode=conditioning_mode,
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
# Multi-target-column prediction (regime 2). Both the "parallel" one-pass
# path and the genuine AR-generation path return the same per-query-cell
# MultiCellPrediction shape (rows/cols local to the episode, aligned with
# table.task.col_idx), so multi_cell_metrics below scores every prediction
# source (checkpoint or tree baseline) identically.
# ---------------------------------------------------------------------


@dataclass
class MultiCellPrediction:
    rows: np.ndarray  # [K] local row index within the episode
    cols: np.ndarray  # [K] local col index within the episode
    y_true: np.ndarray  # [K] class id (categorical) or float value (numerical)
    y_pred: np.ndarray  # [K] same convention
    y_proba: Optional[np.ndarray]  # [K, k_max], NaN rows for numerical cells
    is_categorical: np.ndarray  # [K] bool
    step_index: np.ndarray  # [K] AR step that revealed this cell (0 for parallel/baselines)


@torch.no_grad()
def predict_stream_family_parallel_multi(
    ckpt: LoadedCheckpoint, table: ConvertedTableMulti, device: torch.device
) -> MultiCellPrediction:
    """Generalizes predict_stream_family to k>1 query columns: same one-pass
    ParallelFactorizer construction, just scoring every query_mask cell
    instead of only the target column."""
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

    return _model_output_to_multi_prediction(out, table.full, table.task)


@torch.no_grad()
def predict_stream_family_ar_multi(
    ckpt: LoadedCheckpoint, table: ConvertedTableMulti, device: torch.device
) -> MultiCellPrediction:
    """Genuine non-teacher-forced AR generation (ar_generate.generate_ar):
    predicts one held-out column at a time, feeding the model's OWN
    prediction back in as the revealed value for later columns -- not the
    ground truth. ar_unit="column" is correct here regardless of
    conditioning_mode (see plan doc): every query row shares the same
    held-out column set, so there's no per-row cell-scheduling ambiguity
    the way there would be for CellBlockSampler-style irregular masks."""
    is_sparse = ckpt.family == "two_stream_ar_sparse"
    result = generate_ar(
        ckpt.model,
        table.full,
        table.task,
        device,
        ar_unit="column",
        rng=np.random.default_rng(0),
        is_sparse=is_sparse,
    )
    return MultiCellPrediction(
        rows=result.rows,
        cols=result.cols,
        y_true=result.y_true,
        y_pred=result.y_pred,
        y_proba=result.y_proba,
        is_categorical=result.is_categorical,
        step_index=result.step_index,
    )


def _model_output_to_multi_prediction(
    out, full: FullSyntheticTable, task: CompletionTask
) -> MultiCellPrediction:
    """Reads out.cat_logits/out.num_mu at every task.query_mask cell (one
    single-pass forward already computed) into the same MultiCellPrediction
    shape generate_ar returns -- shared by the parallel prediction path."""
    coords = task.query_coords_local()
    col_types = full.col_types[task.col_idx]
    cat_card = full.cat_cardinalities[task.col_idx]
    x_num_true = full.x_num[np.ix_(task.row_idx, task.col_idx)]
    x_cat_true = full.x_cat[np.ix_(task.row_idx, task.col_idx)]

    # num_mu is in the model's roughly-standardized training scale, never
    # un-normalized internally -- see numeric_context_stats's docstring.
    ctx_mean, ctx_std = numeric_context_stats(x_num_true, task.observed_mask)

    k_max = int(cat_card[col_types == CATEGORICAL].max()) if (col_types == CATEGORICAL).any() else 1

    rows_out, cols_out, y_true, y_pred, y_proba, is_cat = [], [], [], [], [], []
    for r, c in coords:
        r, c = int(r), int(c)
        rows_out.append(r)
        cols_out.append(c)
        if col_types[c] == CATEGORICAL:
            n_valid = int(cat_card[c])
            logits = out.cat_logits[0, r, c, :n_valid].float()
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            pred_cls = int(probs.argmax())
            proba_row = np.full(k_max, np.nan, dtype=np.float32)
            proba_row[:n_valid] = probs
            y_proba.append(proba_row)
            y_true.append(int(x_cat_true[r, c]))
            y_pred.append(pred_cls)
            is_cat.append(True)
        else:
            pred_val_norm = float(out.num_mu[0, r, c].item())
            pred_val = pred_val_norm * float(ctx_std[c]) + float(ctx_mean[c])
            y_proba.append(np.full(k_max, np.nan, dtype=np.float32))
            y_true.append(float(x_num_true[r, c]))
            y_pred.append(pred_val)
            is_cat.append(False)

    return MultiCellPrediction(
        rows=np.asarray(rows_out, dtype=np.int64),
        cols=np.asarray(cols_out, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.float64),
        y_pred=np.asarray(y_pred, dtype=np.float64),
        y_proba=np.stack(y_proba) if y_proba else None,
        is_categorical=np.asarray(is_cat, dtype=bool),
        step_index=np.zeros(len(rows_out), dtype=np.int64),
    )


_RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [1, 2, 4],
}
_XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

# (classifier_param_dist, regressor_param_dist) -- rf/xgb share hyperparameter
# names between their classifier/regressor variants, but LogisticRegression's
# "C" and Ridge's "alpha" don't, so logreg128 needs two distinct dicts.
TUNED_MULTI_COL_PARAM_DISTRIBUTIONS = {
    "rf128": (_RF_PARAM_DIST, _RF_PARAM_DIST),
    "xgb128": (_XGB_PARAM_DIST, _XGB_PARAM_DIST),
    "logreg128": (
        {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
}


def _make_tuned_estimator_factory(base_factory, param_dist, seed, n_jobs, n_iter, cv_folds):
    def factory():
        est = base_factory()
        if hasattr(est, "n_jobs"):
            # eval_multi_col_baselines.py's rf128/xgb128 hardcode n_jobs=-1 --
            # a joblib-level oversubscription risk on a shared node that
            # threadpool_limits (native BLAS/OpenMP only) can't catch, so cap
            # it explicitly here instead.
            est.n_jobs = n_jobs
        return RandomizedSearchCV(
            est,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            random_state=seed,
            n_jobs=1,  # avoid nested-parallelism with the base estimator's own n_jobs
            error_score="raise",
        )

    return factory


def build_tuned_multi_col_families(
    seed: int, n_jobs: int, n_iter: int, cv_folds: int
) -> Dict[str, Any]:
    """Tuned analogue of eval_multi_col_baselines.py's _build_families(): same
    estimator families (logreg/ridge, rf, xgb), each wrapped in
    RandomizedSearchCV (fit on the raw per-column X_ctx/y_ctx arrays --
    already one-hot-encoded/numeric via multi_context_query_features, no
    Pipeline/preprocessor needed here unlike the single-target baselines)."""
    base_families = build_multi_col_baseline_families()
    tuned: Dict[str, Any] = {}

    for name, (clf_factory, reg_factory) in base_families.items():
        param_dists = TUNED_MULTI_COL_PARAM_DISTRIBUTIONS.get(name)
        if param_dists is None:
            tuned[name] = (clf_factory, reg_factory)
            continue
        clf_param_dist, reg_param_dist = param_dists
        tuned[name] = (
            _make_tuned_estimator_factory(clf_factory, clf_param_dist, seed, n_jobs, n_iter, cv_folds),
            _make_tuned_estimator_factory(reg_factory, reg_param_dist, seed, n_jobs, n_iter, cv_folds),
        )

    return tuned


def predict_baseline_family_multi(
    full: FullSyntheticTable, task: CompletionTask, clf_factory, reg_factory
) -> Optional[MultiCellPrediction]:
    """Tree/logreg baseline analogue of predict_stream_family_ar_multi: one
    independently-fit model per queried column (reusing
    eval_multi_col_baselines.py's leak-free feature extraction -- every
    queried column is excluded from every OTHER column's own features),
    collected into the same per-cell MultiCellPrediction shape so
    multi_cell_metrics scores checkpoints and baselines identically. This is
    the "product of independent marginals" comparison point for perm_ar's
    genuine joint conditioning."""
    cols = multi_context_query_features(full, task)
    if not cols:
        return None

    query_rows_local = task.meta["query_rows_local"]
    col_types_all = full.col_types[task.col_idx]
    cat_card_all = full.cat_cardinalities[task.col_idx]
    k_max = (
        int(cat_card_all[col_types_all == CATEGORICAL].max())
        if (col_types_all == CATEGORICAL).any()
        else 1
    )

    rows_out, cols_out, y_true, y_pred, y_proba, is_cat = [], [], [], [], [], []
    for col in cols:
        c, X_ctx, y_ctx, X_qry, y_qry = col["col"], col["X_ctx"], col["y_ctx"], col["X_qry"], col["y_qry"]

        if col["is_cat"]:
            if len(np.unique(y_ctx)) < 2:
                pred = np.full_like(y_qry, y_ctx[0])
                probs_all = None
            else:
                classes, y_ctx_codes = np.unique(y_ctx, return_inverse=True)
                clf = clf_factory()
                clf.fit(X_ctx, y_ctx_codes)
                pred_codes = clf.predict(X_qry)
                pred = classes[pred_codes]
                probs_all = None
                if hasattr(clf, "predict_proba"):
                    probs_codes = clf.predict_proba(X_qry)
                    probs_all = np.zeros((len(y_qry), k_max), dtype=np.float32)
                    probs_all[:, classes] = probs_codes

            for i, row_local in enumerate(query_rows_local):
                rows_out.append(int(row_local))
                cols_out.append(int(c))
                y_true.append(int(y_qry[i]))
                y_pred.append(int(pred[i]))
                is_cat.append(True)
                if probs_all is not None:
                    y_proba.append(probs_all[i])
                else:
                    proba_row = np.full(k_max, np.nan, dtype=np.float32)
                    proba_row[int(pred[i])] = 1.0
                    y_proba.append(proba_row)
        else:
            reg = reg_factory()
            reg.fit(X_ctx, y_ctx)
            pred = reg.predict(X_qry)
            for i, row_local in enumerate(query_rows_local):
                rows_out.append(int(row_local))
                cols_out.append(int(c))
                y_true.append(float(y_qry[i]))
                y_pred.append(float(pred[i]))
                is_cat.append(False)
                y_proba.append(np.full(k_max, np.nan, dtype=np.float32))

    return MultiCellPrediction(
        rows=np.asarray(rows_out, dtype=np.int64),
        cols=np.asarray(cols_out, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.float64),
        y_pred=np.asarray(y_pred, dtype=np.float64),
        y_proba=np.stack(y_proba) if y_proba else None,
        is_categorical=np.asarray(is_cat, dtype=bool),
        step_index=np.zeros(len(rows_out), dtype=np.int64),
    )


def predict_baseline_family_chained(
    full: FullSyntheticTable, task: CompletionTask, clf_factory, reg_factory
) -> Optional[MultiCellPrediction]:
    """Chained analogue of predict_baseline_family_multi -- the fair
    comparison point for perm_ar specifically, not just parallel.

    predict_baseline_family_multi excludes ALL k held-out columns from every
    one of their own feature sets, even though in context rows every column
    (including the other k-1 held-out ones) really is observed -- so that
    baseline can never learn the correlation structure between held-out
    columns at all, unlike perm_ar which conditions on them via its own
    self-predicted values.

    This fits each held-out column in a fixed order (sorted by column index),
    augmenting the never-hidden feature set with the OTHER already-chained
    held-out columns: their TRUE value when fitting on context (context rows
    have every column observed, so this is legitimate, not leakage), their
    OWN PREDICTED value at query time (query rows never have it either,
    exactly like context wouldn't at real inference time). This mirrors
    perm_ar's teacher-forced-train / self-conditioned-generate split exactly.
    """
    cols = multi_context_query_features(full, task)
    if not cols:
        return None

    query_rows_local = task.meta["query_rows_local"]
    col_types_all = full.col_types[task.col_idx]
    cat_card_all = full.cat_cardinalities[task.col_idx]
    k_max = (
        int(cat_card_all[col_types_all == CATEGORICAL].max())
        if (col_types_all == CATEGORICAL).any()
        else 1
    )

    cols_sorted = sorted(cols, key=lambda col: col["col"])

    rows_out, cols_out, y_true, y_pred, y_proba, is_cat = [], [], [], [], [], []
    X_ctx_extra: List[np.ndarray] = []  # grows by one [n_ctx,1] column per already-chained column
    X_qry_extra: List[np.ndarray] = []  # same, but the PREDICTED value for query rows

    for col in cols_sorted:
        c, X_ctx_base, y_ctx, X_qry_base, y_qry = (
            col["col"], col["X_ctx"], col["y_ctx"], col["X_qry"], col["y_qry"]
        )
        extra_ctx = np.concatenate(X_ctx_extra, axis=1) if X_ctx_extra else np.zeros((len(y_ctx), 0))
        extra_qry = np.concatenate(X_qry_extra, axis=1) if X_qry_extra else np.zeros((len(y_qry), 0))
        X_ctx = np.concatenate([X_ctx_base, extra_ctx], axis=1)
        X_qry = np.concatenate([X_qry_base, extra_qry], axis=1)

        if col["is_cat"]:
            if len(np.unique(y_ctx)) < 2:
                pred = np.full_like(y_qry, y_ctx[0])
                probs_all = None
            else:
                classes, y_ctx_codes = np.unique(y_ctx, return_inverse=True)
                clf = clf_factory()
                clf.fit(X_ctx, y_ctx_codes)
                pred_codes = clf.predict(X_qry)
                pred = classes[pred_codes]
                probs_all = None
                if hasattr(clf, "predict_proba"):
                    probs_codes = clf.predict_proba(X_qry)
                    probs_all = np.zeros((len(y_qry), k_max), dtype=np.float32)
                    probs_all[:, classes] = probs_codes

            for i, row_local in enumerate(query_rows_local):
                rows_out.append(int(row_local))
                cols_out.append(int(c))
                y_true.append(int(y_qry[i]))
                y_pred.append(int(pred[i]))
                is_cat.append(True)
                if probs_all is not None:
                    y_proba.append(probs_all[i])
                else:
                    proba_row = np.full(k_max, np.nan, dtype=np.float32)
                    proba_row[int(pred[i])] = 1.0
                    y_proba.append(proba_row)
        else:
            reg = reg_factory()
            reg.fit(X_ctx, y_ctx)
            pred = reg.predict(X_qry)
            for i, row_local in enumerate(query_rows_local):
                rows_out.append(int(row_local))
                cols_out.append(int(c))
                y_true.append(float(y_qry[i]))
                y_pred.append(float(pred[i]))
                is_cat.append(False)
                y_proba.append(np.full(k_max, np.nan, dtype=np.float32))

        # Chain augmentation for subsequent columns: true value in context,
        # this model's own prediction for query (never the query ground truth).
        X_ctx_extra.append(y_ctx.reshape(-1, 1).astype(np.float64))
        X_qry_extra.append(pred.reshape(-1, 1).astype(np.float64))

    return MultiCellPrediction(
        rows=np.asarray(rows_out, dtype=np.int64),
        cols=np.asarray(cols_out, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.float64),
        y_pred=np.asarray(y_pred, dtype=np.float64),
        y_proba=np.stack(y_proba) if y_proba else None,
        is_categorical=np.asarray(is_cat, dtype=bool),
        step_index=np.zeros(len(rows_out), dtype=np.int64),
    )


def multi_cell_metrics(pred: MultiCellPrediction) -> Dict[str, float]:
    """Marginal (per-cell) + joint (per-query-row, categorical cells only --
    exact-match and chain-rule NLL aren't well-defined for numerical cells
    without a different treatment) metrics from one MultiCellPrediction.
    Same formula for every prediction source (parallel/perm_ar/baseline):
    joint_nll = mean over rows of sum(-log p_true) across that row's held-out
    categorical cells. For perm_ar the per-cell probabilities already
    incorporate genuine conditioning on earlier-revealed (self-predicted)
    columns; for parallel and every baseline they don't -- so this is the
    number that isolates whether perm_ar's conditioning earns anything over
    the independence assumption everyone else makes."""
    out: Dict[str, float] = {}
    cat_mask = pred.is_categorical

    if cat_mask.any():
        y_true_cat = pred.y_true[cat_mask].astype(np.int64)
        y_pred_cat = pred.y_pred[cat_mask].astype(np.int64)
        rows_cat = pred.rows[cat_mask]
        proba_cat = pred.y_proba[cat_mask] if pred.y_proba is not None else None
        correct_cat = y_true_cat == y_pred_cat

        out["marginal_cat_acc"] = float(correct_cat.mean())
        out["marginal_cat_cells"] = float(cat_mask.sum())

        joint_correct = 0
        joint_nll_sum = 0.0
        n_rows = 0
        for row in np.unique(rows_cat):
            row_idx = np.flatnonzero(rows_cat == row)
            n_rows += 1
            if correct_cat[row_idx].all():
                joint_correct += 1
            if proba_cat is not None:
                p_true = np.array(
                    [proba_cat[i, y_true_cat[i]] for i in row_idx], dtype=np.float64
                )
                p_true = np.clip(p_true, 1e-12, 1.0)
                joint_nll_sum += float(-np.sum(np.log(p_true)))

        out["joint_exact_match"] = joint_correct / n_rows if n_rows else float("nan")
        out["joint_nll"] = joint_nll_sum / n_rows if n_rows else float("nan")
        out["joint_rows"] = float(n_rows)

    num_mask = ~cat_mask
    if num_mask.any():
        err = pred.y_true[num_mask] - pred.y_pred[num_mask]
        out["marginal_num_mse"] = float(np.mean(err**2))
        out["marginal_num_cells"] = float(num_mask.sum())

    return out


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
            # HistGradientBoostingClassifier has no n_jobs knob (it uses OpenMP
            # internally); on a shared/contended node an uncapped thread pool
            # thrashes badly, so cap every model's threads explicitly here.
            with threadpool_limits(limits=args.n_jobs):
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


def _balanced_draws(
    pool: np.ndarray, group_size: int, n_groups: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """n_groups non-overlapping windows of size group_size, built by slicing
    consecutive shuffled copies of pool concatenated end to end. Guarantees
    every element of pool is used floor(n_groups*group_size/len(pool)) or
    that +1 times across the n_groups windows -- independent rng.choice draws
    give each element only an *expected* count with real sample-to-sample
    variance, which is what this sidesteps."""
    if group_size == 0:
        return [np.array([], dtype=np.int64) for _ in range(n_groups)]
    needed = n_groups * group_size
    chunks = []
    total = 0
    while total < needed:
        chunk = rng.permutation(pool)
        chunks.append(chunk)
        total += len(chunk)
    seq = np.concatenate(chunks)[:needed]
    return [seq[i * group_size : (i + 1) * group_size].astype(np.int64) for i in range(n_groups)]


def sample_balanced_target_sets(
    categorical_cols: np.ndarray,
    numeric_cols: np.ndarray,
    k: int,
    m_cat: int,
    n_sets: int,
    rng: np.random.Generator,
) -> Optional[List[np.ndarray]]:
    """n_sets balanced target-sets of size k, each with exactly m_cat
    categorical columns (from categorical_cols) + (k - m_cat) numeric columns
    (from numeric_cols) -- m_cat is guaranteed by construction (drawn from
    disjoint pools), never by filtering/rejecting mixed draws. Returns None
    if this (k, m_cat) cell isn't reachable given how many categorical/
    numeric columns this dataset actually has."""
    n_num = k - m_cat
    if m_cat > len(categorical_cols) or n_num > len(numeric_cols):
        return None
    cat_draws = _balanced_draws(categorical_cols, m_cat, n_sets, rng)
    num_draws = _balanced_draws(numeric_cols, n_num, n_sets, rng)
    return [np.sort(np.concatenate([c, n])).astype(np.int64) for c, n in zip(cat_draws, num_draws)]


def run_task_multi_target(
    task_id: int,
    args,
    checkpoints: List[LoadedCheckpoint],
    device: torch.device,
    logger: JSONLLogger,
) -> None:
    """Regime 2: hold out the same k>1 columns for every query row, compare
    genuine perm_ar generation (ar_generate.generate_ar) against the
    one-pass parallel path and independent-per-column baselines. See the
    plan doc (wobbly-stargazing-micali.md) for the full design."""
    task_obj, X, y_raw = load_openml_task(task_id)
    task_name = getattr(task_obj, "name", None) or str(task_id)

    n_rows = len(X)
    n_features = X.shape[1]

    def log_dataset_skip(reason: str) -> None:
        print(f"[skip] task={task_id} name={task_name}: {reason}")
        logger.log(
            {
                "suite": args.suite,
                "task_id": task_id,
                "task_name": task_name,
                "task_type": "multi_target",
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

    full_probe, _y_enc, _n_classes, _target_col, _max_card, n_cols_probe = _build_full_table_from_df(
        X, y_raw
    )
    categorical_cols = np.where(full_probe.col_types == CATEGORICAL)[0].astype(np.int64)
    numeric_cols = np.where(full_probe.col_types != CATEGORICAL)[0].astype(np.int64)

    k_list = [int(k) for k in args.k_cols.split(",")]
    cond_modes = [m.strip() for m in args.conditioning_modes.split(",")]
    n_sets = args.num_target_sets

    for k in k_list:
        for m_cat in range(1, k + 1):
            col_rng = np.random.default_rng([task_id, k, m_cat, args.multi_target_seed])
            target_sets = sample_balanced_target_sets(
                categorical_cols, numeric_cols, k, m_cat, n_sets, col_rng
            )
            if target_sets is None:
                reason = (
                    f"m_cat={m_cat} needs {m_cat} categorical (have {len(categorical_cols)}) + "
                    f"{k - m_cat} numeric (have {len(numeric_cols)}) columns"
                )
                print(f"  [skip-k,m_cat] task={task_id} k={k} m_cat={m_cat}: {reason}")
                logger.log(
                    {
                        "suite": args.suite,
                        "task_id": task_id,
                        "task_name": task_name,
                        "task_type": "multi_target",
                        "model": None,
                        "k_cols": k,
                        "m_cat": m_cat,
                        "n_features": n_features,
                        "status": "skipped",
                        "error": reason,
                    }
                )
                continue

            for cond_idx, conditioning_mode in enumerate(cond_modes):
                for set_idx, query_cols in enumerate(target_sets):
                    # Same seed key regardless of m_cat/set_idx -> identical
                    # context/query row split for every target-set at this
                    # (task, k), so only the columns vary across draws.
                    rng = np.random.default_rng([args.seed, task_id, k])
                    table = convert_openml_table_multi(
                        X,
                        y_raw,
                        rng,
                        query_cols,
                        args.max_context,
                        args.max_query,
                        conditioning_mode,
                        query_frac=args.query_frac,
                    )

                    print(
                        f"[multi_target] task={task_id} name={task_name} k={k} m_cat={m_cat} "
                        f"set={set_idx}/{n_sets} cond={conditioning_mode} n_context={table.n_context} "
                        f"n_query={table.n_query} query_cols={query_cols.tolist()}"
                    )

                    def log_row(model_name: str, ar_mode: str, extra: Dict[str, Any]) -> None:
                        row = {
                            "suite": args.suite,
                            "task_id": task_id,
                            "task_name": task_name,
                            "task_type": "multi_target",
                            "model": model_name,
                            "k_cols": k,
                            "m_cat": m_cat,
                            "target_set_idx": set_idx,
                            "conditioning_mode": conditioning_mode,
                            "ar_mode": ar_mode,
                            "n_train": table.n_context,
                            "n_test": table.n_query,
                            "n_features": n_features,
                            "n_classes": table.n_classes,
                            "status": "ok",
                        }
                        row.update(extra)
                        logger.log(row)

                    for ckpt in checkpoints:
                        if ckpt.family == "tabpfn_v1":
                            continue  # architecturally single-target only.

                        skip_reason = eligible_for_checkpoint(table, ckpt)
                        if skip_reason is not None:
                            print(f"  [skip-model] {ckpt.tag}: {skip_reason}")
                            log_row(ckpt.tag, "parallel", {"status": "skipped", "error": skip_reason})
                            continue

                        start = time.perf_counter()
                        try:
                            pred = predict_stream_family_parallel_multi(ckpt, table, device)
                            metrics = multi_cell_metrics(pred)
                            log_row(
                                ckpt.tag,
                                "parallel",
                                {**metrics, "fit_predict_sec": time.perf_counter() - start},
                            )
                        except Exception as e:
                            print(f"  [error] {ckpt.tag} parallel: {e}")
                            log_row(
                                ckpt.tag,
                                "parallel",
                                {"status": "error", "error": repr(e), "traceback": traceback.format_exc()},
                            )

                        if ckpt.family in ("two_stream_ar", "two_stream_ar_sparse"):
                            start = time.perf_counter()
                            try:
                                pred = predict_stream_family_ar_multi(ckpt, table, device)
                                metrics = multi_cell_metrics(pred)
                                log_row(
                                    ckpt.tag,
                                    "perm_ar",
                                    {**metrics, "fit_predict_sec": time.perf_counter() - start},
                                )
                            except Exception as e:
                                print(f"  [error] {ckpt.tag} perm_ar: {e}")
                                log_row(
                                    ckpt.tag,
                                    "perm_ar",
                                    {
                                        "status": "error",
                                        "error": repr(e),
                                        "traceback": traceback.format_exc(),
                                    },
                                )

                    # Baselines don't depend on the model's conditioning_mode
                    # (they never see a context_row_mask at all) -- only
                    # fit/score them once per (task, k, m_cat, target_set), on
                    # the first conditioning_mode pass.
                    if cond_idx != 0:
                        continue

                    # Unlike the "target" regime's build_tuned_pipelines, this
                    # block previously ignored --baseline-models entirely
                    # (always ran all 3 tuned families x 2 modes via
                    # RandomizedSearchCV, regardless of the flag) -- reusing
                    # the same empty-string-disables convention here so a
                    # checkpoint-only run doesn't pay for baseline tuning it
                    # doesn't need. Note this now refits per target-set (not
                    # once per k) -- leave --baseline-models empty for
                    # checkpoint-only runs against this grid.
                    if not args.baseline_models.strip():
                        continue

                    families = build_tuned_multi_col_families(
                        args.seed, args.n_jobs, args.baseline_tuning_iters, args.baseline_cv_folds
                    )
                    baseline_modes = [m.strip() for m in args.baseline_modes.split(",") if m.strip()]
                    baseline_predict_fns = {
                        "independent": predict_baseline_family_multi,
                        "chained": predict_baseline_family_chained,
                    }
                    for name, (clf_factory, reg_factory) in families.items():
                        for mode in baseline_modes:
                            predict_fn = baseline_predict_fns[mode]
                            # independent = the fair comparison point for
                            # `parallel` (no cross-held-out-column
                            # conditioning, matching that factorization's own
                            # independence assumption); chained = the fair
                            # comparison point for `perm_ar` (conditions on
                            # other held-out columns via its own predictions,
                            # mirroring perm_ar's self-conditioned generation).
                            model_name = name if mode == "independent" else f"{name}_chained"
                            start = time.perf_counter()
                            try:
                                with threadpool_limits(limits=args.n_jobs):
                                    pred = predict_fn(table.full, table.task, clf_factory, reg_factory)
                                if pred is None:
                                    continue
                                metrics = multi_cell_metrics(pred)
                                log_row(
                                    model_name,
                                    "n/a",
                                    {**metrics, "fit_predict_sec": time.perf_counter() - start},
                                )
                            except Exception as e:
                                print(f"  [error] {model_name}: {e}")
                                log_row(
                                    model_name,
                                    "n/a",
                                    {
                                        "status": "error",
                                        "error": repr(e),
                                        "traceback": traceback.format_exc(),
                                    },
                                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="'path1=tag1,path2=tag2,...'")
    parser.add_argument("--suite", type=str, default="cc18", choices=["cc18"])
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)

    parser.add_argument(
        "--regime", type=str, default="target", choices=["target", "multi_target"]
    )
    parser.add_argument("--k-cols", type=str, default="1,2,4,8")
    parser.add_argument("--multi-target-seed", type=int, default=0)
    parser.add_argument(
        "--num-target-sets",
        type=int,
        default=10,
        help="multi_target regime only: S balanced target-sets sampled per (task, k, m_cat) "
        "cell (see sample_balanced_target_sets). m_cat=k is the old --categorical-only "
        "case (all held-out columns categorical) -- it's swept automatically now, no "
        "separate flag needed; checkpoints with an untrained num_head should be pointed "
        "at the m_cat=k rows specifically rather than run against the full grid.",
    )
    parser.add_argument(
        "--query-frac",
        type=float,
        default=1.0,
        help="multi_target regime only: fraction of the row-split's query rows actually "
        "scored (trims cost while keeping the same context/query row split, and thus "
        "n_context, fixed across every m_cat/target-set draw for a given (task, k)).",
    )
    parser.add_argument(
        "--conditioning-modes",
        type=str,
        default="inductive_rows",
        help="Comma-separated, multi_target regime only: inductive_rows,transductive. "
        "Only checkpoints whose own training used transductive (e.g. random_cell-trained "
        "two_stream_ar_sparse runs) benefit from also including transductive.",
    )

    parser.add_argument("--max-total-rows", type=int, default=1024)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=63)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--max-query", type=int, default=512)

    parser.add_argument("--baseline-models", type=str, default="rf,hgb,xgb")
    parser.add_argument("--baseline-tuning-iters", type=int, default=25)
    parser.add_argument("--baseline-cv-folds", type=int, default=3)
    parser.add_argument(
        "--baseline-modes",
        type=str,
        default="independent,chained",
        help="multi_target regime only: independent (fair vs. `parallel`) and/or "
        "chained (fair vs. `perm_ar` -- conditions on other held-out columns via "
        "its own predictions, mirroring perm_ar's self-conditioned generation). "
        "Logged as separate baseline rows, chained ones suffixed '_chained'.",
    )
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

    task_runner = run_task_multi_target if args.regime == "multi_target" else run_task

    for idx, task_id in enumerate(task_ids):
        print(f"\n=== [{idx + 1}/{len(task_ids)}] task_id={task_id} ===")
        try:
            task_runner(task_id, args, checkpoints, device, logger)
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
