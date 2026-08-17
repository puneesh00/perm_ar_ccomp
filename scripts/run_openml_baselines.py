# scripts/run_openml_baselines.py

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import openml


# ---------------------------------------------------------------------
# Optional libraries
# ---------------------------------------------------------------------


def has_package(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_XGBOOST = has_package("xgboost")
HAS_LIGHTGBM = has_package("lightgbm")
HAS_CATBOOST = has_package("catboost")


# ---------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------


SUITE_IDS = {
    "cc18": 99,     # OpenML-CC18 classification
    "ctr23": 353,  # OpenML-CTR23 regression
}


def get_suite_task_ids(suite_name: str) -> List[int]:
    if suite_name not in SUITE_IDS:
        raise ValueError(f"Unknown suite {suite_name}. Choose from {list(SUITE_IDS)}.")

    suite_id = SUITE_IDS[suite_name]
    suite = openml.study.get_suite(suite_id)
    return list(suite.tasks)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_openml_task(task_id: int, max_retries: int = 4, backoff_sec: float = 5.0):
    """
    openml.org occasionally returns transient errors (e.g. 503s under load).
    Retry with exponential backoff before giving up on this task.
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            task = openml.tasks.get_task(task_id)

            # This works for most modern openml-python versions.
            try:
                X, y = task.get_X_and_y(dataset_format="dataframe")
            except Exception:
                dataset = task.get_dataset()
                target_name = task.target_name
                X, y, _, _ = dataset.get_data(
                    target=target_name,
                    dataset_format="dataframe",
                )

            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            y = pd.Series(y)

            return task, X, y

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_sec = backoff_sec * (2 ** attempt)
                print(
                    f"[warn] task={task_id}: load failed on attempt "
                    f"{attempt + 1}/{max_retries} ({e!r}); "
                    f"retrying in {sleep_sec:.0f}s."
                )
                time.sleep(sleep_sec)

    assert last_err is not None
    raise last_err


def get_split_indices(task, repeat: int, fold: int, sample: int):
    train_idx, test_idx = task.get_train_test_split_indices(
        repeat=repeat,
        fold=fold,
        sample=sample,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def subsample_indices(
    rng: np.random.Generator,
    idx: np.ndarray,
    max_size: Optional[int],
) -> np.ndarray:
    if max_size is None or len(idx) <= max_size:
        return idx

    return rng.choice(idx, size=max_size, replace=False)


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------


def infer_column_types(X: pd.DataFrame):
    categorical_cols = []
    numeric_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def make_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    CPU-friendly tree preprocessing.

    Numerical:
        median imputation.

    Categorical:
        most-frequent imputation + ordinal encoding.

    This is not perfect for all models, but it is robust and fast enough for
    baseline scripts. Later, CatBoost can get raw categorical columns directly.
    """
    numeric_cols, categorical_cols = infer_column_types(X)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable columns found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


def build_classification_models(
    seed: int,
    n_jobs: int,
    model_names: Iterable[str],
    n_classes: int,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    requested = set(model_names)

    if "rf" in requested:
        models["rf"] = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=n_jobs,
            random_state=seed,
        )

    if "extratrees" in requested:
        models["extratrees"] = ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=n_jobs,
            random_state=seed,
        )

    if "hgb" in requested:
        models["hgb"] = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=seed,
        )

    if "xgb" in requested:
        if not HAS_XGBOOST:
            print("[warn] xgboost not installed; skipping xgb.")
        else:
            from xgboost import XGBClassifier

            if n_classes == 2:
                xgb_kwargs = dict(objective="binary:logistic", eval_metric="logloss")
            else:
                xgb_kwargs = dict(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=n_classes,
                )

            models["xgb"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                device="cpu",
                n_jobs=n_jobs,
                random_state=seed,
                **xgb_kwargs,
            )

    if "lgbm" in requested:
        if not HAS_LIGHTGBM:
            print("[warn] lightgbm not installed; skipping lgbm.")
        else:
            from lightgbm import LGBMClassifier

            models["lgbm"] = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=n_jobs,
                random_state=seed,
                verbose=-1,
            )

    if "catboost" in requested:
        if not HAS_CATBOOST:
            print("[warn] catboost not installed; skipping catboost.")
        else:
            from catboost import CatBoostClassifier

            models["catboost"] = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="MultiClass",
                task_type="CPU",
                thread_count=n_jobs,
                random_seed=seed,
                verbose=False,
            )

    return models


def build_regression_models(
    seed: int,
    n_jobs: int,
    model_names: Iterable[str],
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    requested = set(model_names)

    if "rf" in requested:
        models["rf"] = RandomForestRegressor(
            n_estimators=300,
            max_features=1.0,
            min_samples_leaf=1,
            n_jobs=n_jobs,
            random_state=seed,
        )

    if "extratrees" in requested:
        models["extratrees"] = ExtraTreesRegressor(
            n_estimators=300,
            max_features=1.0,
            min_samples_leaf=1,
            n_jobs=n_jobs,
            random_state=seed,
        )

    if "hgb" in requested:
        models["hgb"] = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=seed,
        )

    if "xgb" in requested:
        if not HAS_XGBOOST:
            print("[warn] xgboost not installed; skipping xgb.")
        else:
            from xgboost import XGBRegressor

            models["xgb"] = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                n_jobs=n_jobs,
                random_state=seed,
            )

    if "lgbm" in requested:
        if not HAS_LIGHTGBM:
            print("[warn] lightgbm not installed; skipping lgbm.")
        else:
            from lightgbm import LGBMRegressor

            models["lgbm"] = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=n_jobs,
                random_state=seed,
                verbose=-1,
            )

    if "catboost" in requested:
        if not HAS_CATBOOST:
            print("[warn] catboost not installed; skipping catboost.")
        else:
            from catboost import CatBoostRegressor

            models["catboost"] = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                task_type="CPU",
                thread_count=n_jobs,
                random_seed=seed,
                verbose=False,
            )

    return models


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            out["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            out["log_loss"] = float("nan")

        try:
            n_classes = y_proba.shape[1]
            labels = np.arange(n_classes)
            if n_classes == 2:
                out["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba[:, 1], labels=labels)
                )
            else:
                out["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovo",
                        average="macro",
                        labels=labels,
                    )
                )
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["log_loss"] = float("nan")
        out["roc_auc"] = float("nan")

    return out


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        row = {k: convert(v) for k, v in row.items()}

        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    rows = []
    with jsonl_path.open("r") as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------


def run_task_classification(
    task_id: int,
    args,
    logger: JSONLLogger,
) -> None:
    rng = np.random.default_rng(args.seed + task_id)

    task, X, y_raw = load_openml_task(task_id)
    task_name = getattr(task, "name", None) or str(task_id)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    n_classes = len(le.classes_)
    if n_classes < 2:
        print(f"[skip] task={task_id}: fewer than 2 classes.")
        return

    train_idx, test_idx = get_split_indices(
        task,
        repeat=args.repeat,
        fold=args.fold,
        sample=args.sample,
    )

    train_idx = subsample_indices(rng, train_idx, args.max_train_rows)
    test_idx = subsample_indices(rng, test_idx, args.max_test_rows)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    preprocessor = make_tree_preprocessor(X_train)
    models = build_classification_models(args.seed, args.n_jobs, args.models.split(","), n_classes)

    print(
        f"[classification] task={task_id} name={task_name} "
        f"train={len(train_idx)} test={len(test_idx)} "
        f"features={X.shape[1]} classes={n_classes}"
    )

    for model_name, model in models.items():
        start = time.perf_counter()

        row: Dict[str, Any] = {
            "suite": args.suite,
            "task_id": task_id,
            "task_name": task_name,
            "task_type": "classification",
            "model": model_name,
            "repeat": args.repeat,
            "fold": args.fold,
            "sample": args.sample,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": X.shape[1],
            "n_classes": n_classes,
            "status": "ok",
        }

        try:
            pipe = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", clone(model)),
                ]
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_test)
            else:
                y_proba = None

            row.update(classification_metrics(y_test, y_pred, y_proba))
            row["fit_predict_sec"] = time.perf_counter() - start

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)
            row["traceback"] = traceback.format_exc()
            row["fit_predict_sec"] = time.perf_counter() - start
            print(f"[error] task={task_id} model={model_name}: {e}")

        logger.log(row)


def run_task_regression(
    task_id: int,
    args,
    logger: JSONLLogger,
) -> None:
    rng = np.random.default_rng(args.seed + task_id)

    task, X, y_raw = load_openml_task(task_id)
    task_name = getattr(task, "name", None) or str(task_id)

    y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y)

    if valid.sum() < len(y):
        X = X.iloc[valid].copy()
        y = y[valid]
        # OpenML task split indices no longer line up after filtering; skip this case.
        # Most curated regression tasks should not hit this branch.
        print(f"[skip] task={task_id}: target has non-finite values after conversion.")
        return

    train_idx, test_idx = get_split_indices(
        task,
        repeat=args.repeat,
        fold=args.fold,
        sample=args.sample,
    )

    train_idx = subsample_indices(rng, train_idx, args.max_train_rows)
    test_idx = subsample_indices(rng, test_idx, args.max_test_rows)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    preprocessor = make_tree_preprocessor(X_train)
    models = build_regression_models(args.seed, args.n_jobs, args.models.split(","))

    print(
        f"[regression] task={task_id} name={task_name} "
        f"train={len(train_idx)} test={len(test_idx)} features={X.shape[1]}"
    )

    for model_name, model in models.items():
        start = time.perf_counter()

        row: Dict[str, Any] = {
            "suite": args.suite,
            "task_id": task_id,
            "task_name": task_name,
            "task_type": "regression",
            "model": model_name,
            "repeat": args.repeat,
            "fold": args.fold,
            "sample": args.sample,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": X.shape[1],
            "status": "ok",
        }

        try:
            pipe = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", clone(model)),
                ]
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            row.update(regression_metrics(y_test, y_pred))
            row["fit_predict_sec"] = time.perf_counter() - start

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)
            row["traceback"] = traceback.format_exc()
            row["fit_predict_sec"] = time.perf_counter() - start
            print(f"[error] task={task_id} model={model_name}: {e}")

        logger.log(row)


def parse_task_ids(args) -> List[int]:
    if args.task_ids is not None and args.task_ids.strip():
        return [int(x) for x in args.task_ids.split(",")]

    task_ids = get_suite_task_ids(args.suite)

    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    return task_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite",
        type=str,
        default="cc18",
        choices=["cc18", "ctr23"],
        help="OpenML suite: cc18 classification or ctr23 regression.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated OpenML task ids. Overrides --suite task list.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Only run first max_tasks tasks from the suite.",
    )

    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)

    parser.add_argument(
        "--models",
        type=str,
        default="rf,hgb,xgb",
        help=(
            "Comma-separated models. Choices include: "
            "rf,extratrees,hgb,xgb,lgbm,catboost."
        ),
    )

    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)

    parser.add_argument("--n-jobs", type=int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results/openml_baselines")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.run_name is None:
        timestamp = int(time.time())
        args.run_name = f"{args.suite}_{args.models.replace(',', '-')}_{timestamp}"

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"

    logger = JSONLLogger(jsonl_path)

    task_ids = parse_task_ids(args)

    print("=== OpenML tree baselines ===")
    print(f"suite={args.suite}")
    print(f"num_tasks={len(task_ids)}")
    print(f"models={args.models}")
    print(f"out_dir={out_dir}")
    print(f"optional packages: xgboost={HAS_XGBOOST}, lightgbm={HAS_LIGHTGBM}, catboost={HAS_CATBOOST}")

    suite_type = "classification" if args.suite == "cc18" else "regression"

    for idx, task_id in enumerate(task_ids):
        print(f"\n=== [{idx + 1}/{len(task_ids)}] task_id={task_id} ===")

        try:
            if suite_type == "classification":
                run_task_classification(task_id, args, logger)
            else:
                run_task_regression(task_id, args, logger)
        except Exception as e:
            # A single flaky/unavailable task (e.g. an openml.org outage)
            # should not take down the rest of the suite run.
            print(f"[error] task={task_id}: failed to load/run task: {e}")
            logger.log(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task_name": str(task_id),
                    "task_type": suite_type,
                    "model": None,
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )

        # Update CSV after every task for convenience.
        write_csv_from_jsonl(jsonl_path, csv_path)

    print(f"\nDone.")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()