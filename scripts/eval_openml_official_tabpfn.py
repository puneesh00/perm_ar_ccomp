# scripts/eval_openml_official_tabpfn.py
"""
Evaluates the OFFICIAL pip-installed TabPFN checkpoints (v1: `tabpfn==0.1.11`,
v2: `tabpfn>=2.0`, both from PyPI, weights auto-downloaded by the package
itself) on the exact same OpenML task/split as
scripts/eval_openml_incontext.py's `target` regime -- reuses
run_openml_baselines.load_openml_task + make_tree_preprocessor and
eval_openml_incontext.convert_openml_table so the context/query row split
for a given (seed, task_id) is byte-for-byte identical to the one our own
retrained checkpoints were scored on, letting the output rows sit directly
next to those runs' metrics.csv.

The two official packages both install under the name `tabpfn` with
mutually incompatible APIs (v1 predates the v2 rewrite entirely), so they
cannot live in one environment -- this script is meant to be invoked twice,
once from each dedicated venv (.venv_tabpfn_v1, .venv_tabpfn_v2), via
--version v1|v2. Only the import + model construction branch differs
between the two; everything else (data loading, split, preprocessing,
metrics, logging schema) is shared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_openml_baselines import (  # noqa: E402
    load_openml_task,
    make_tree_preprocessor,
    classification_metrics,
    JSONLLogger,
    write_csv_from_jsonl,
)
from eval_openml_incontext import convert_openml_table  # noqa: E402


def build_model(version: str, device: str, seed: int):
    if version == "v1":
        from tabpfn import TabPFNClassifier

        # N_ensemble_configurations=32 matches the TabPFN-v1 paper's default
        # full ensemble (the pip package's own default of 3 is a fast/light
        # mode) -- this is meant to be the strongest official-checkpoint
        # comparison point, not the cheapest one.
        return TabPFNClassifier(device=device, seed=seed, N_ensemble_configurations=32)
    elif version == "v2":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device, random_state=seed)
    raise ValueError(f"Unknown version {version!r}")


def padded_proba(y_proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    """classes_ is the subset of 0..n_classes-1 actually present in y_train
    (sklearn-style estimators only emit columns for labels they were fit
    on); pad back out to the full [n_query, n_classes] width used
    everywhere else in this repo's eval harness, zero-filling any class
    missing from the context split."""
    if len(classes_) == n_classes and np.array_equal(classes_, np.arange(n_classes)):
        return y_proba
    out = np.zeros((y_proba.shape[0], n_classes), dtype=y_proba.dtype)
    out[:, classes_.astype(np.int64)] = y_proba
    return out


def run_task(task_id: int, args, logger: JSONLLogger) -> None:
    rng = np.random.default_rng(args.seed + task_id)

    task, X, y_raw = load_openml_task(task_id)
    task_name = getattr(task, "name", None) or str(task_id)
    n_rows = len(X)
    n_features = X.shape[1]

    model_name = f"tabpfn_{args.version}_official"

    def log_row(extra: dict) -> None:
        row = {
            "suite": args.suite,
            "task_id": task_id,
            "task_name": task_name,
            "task_type": "classification",
            "model": model_name,
            "n_features": n_features,
            "status": "ok",
        }
        row.update(extra)
        logger.log(row)

    if n_rows > args.max_total_rows or n_rows < args.min_rows:
        reason = f"n_rows={n_rows} outside [{args.min_rows},{args.max_total_rows}]"
        print(f"[skip] task={task_id} name={task_name}: {reason}")
        log_row({"status": "skipped", "error": reason})
        return
    if n_features > args.max_features:
        reason = f"n_features={n_features} > {args.max_features}"
        print(f"[skip] task={task_id} name={task_name}: {reason}")
        log_row({"status": "skipped", "error": reason})
        return
    if X.isna().any().any() or y_raw.isna().any():
        reason = "contains missing values"
        print(f"[skip] task={task_id} name={task_name}: {reason}")
        log_row({"status": "skipped", "error": reason})
        return

    table = convert_openml_table(X, y_raw, rng, args.max_context, args.max_query)

    print(
        f"[task] id={task_id} name={task_name} n_rows={n_rows} n_features={n_features} "
        f"n_classes={table.n_classes} n_context={table.n_context} n_query={table.n_query}"
    )

    preprocessor = make_tree_preprocessor(table.X_train_df)
    X_train = preprocessor.fit_transform(table.X_train_df)
    X_test = preprocessor.transform(table.X_test_df)

    start = time.perf_counter()
    try:
        clf = build_model(args.version, args.device, args.seed)
        clf.fit(X_train, table.y_train)
        proba_raw = clf.predict_proba(X_test)
        proba = padded_proba(proba_raw, np.asarray(clf.classes_), table.n_classes)
        y_pred = proba.argmax(axis=1)
        metrics = classification_metrics(table.y_test, y_pred, proba)
        log_row(
            {
                **metrics,
                "n_train": table.n_context,
                "n_test": table.n_query,
                "n_classes": table.n_classes,
                "fit_predict_sec": time.perf_counter() - start,
                "n_query_scored": int(len(table.y_test)),
            }
        )
    except Exception as e:
        print(f"  [error] {model_name}: {e}")
        log_row(
            {
                "n_train": table.n_context,
                "n_test": table.n_query,
                "n_classes": table.n_classes,
                "status": "error",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "fit_predict_sec": time.perf_counter() - start,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2"])
    parser.add_argument("--suite", type=str, default="cc18")
    parser.add_argument("--task-ids", type=str, required=True, help="comma-separated OpenML task ids")

    parser.add_argument("--max-total-rows", type=int, default=1024)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=63)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--max-query", type=int, default=512)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results/openml_incontext")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.run_name is None:
        timestamp = int(time.time())
        args.run_name = f"tabpfn_{args.version}_official_{timestamp}"

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"
    logger = JSONLLogger(jsonl_path)

    task_ids = [int(x) for x in args.task_ids.split(",")]

    print(f"=== Official TabPFN-{args.version} OpenML eval ===")
    print(f"device={args.device} n_tasks={len(task_ids)} out_dir={out_dir}")

    for idx, task_id in enumerate(task_ids):
        print(f"\n=== [{idx + 1}/{len(task_ids)}] task_id={task_id} ===")
        try:
            run_task(task_id, args, logger)
        except Exception as e:
            print(f"[error] task={task_id}: failed to load/run task: {e}")
            logger.log(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task_name": str(task_id),
                    "task_type": "classification",
                    "model": f"tabpfn_{args.version}_official",
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )
        write_csv_from_jsonl(jsonl_path, csv_path)

    print("\nDone.")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
