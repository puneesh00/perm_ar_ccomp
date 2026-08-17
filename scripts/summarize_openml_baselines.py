# scripts/summarize_openml_baselines.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "roc_auc": True,
    "r2": True,
}

LOWER_IS_BETTER = {
    "log_loss": True,
    "rmse": True,
    "mae": True,
    "fit_predict_sec": True,
}


def available_metrics(df: pd.DataFrame) -> List[str]:
    candidates = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "log_loss",
        "rmse",
        "mae",
        "r2",
        "fit_predict_sec",
    ]
    return [m for m in candidates if m in df.columns]


def rank_direction(metric: str) -> bool:
    """
    Returns True if higher is better.
    """
    if metric in HIGHER_IS_BETTER:
        return True
    if metric in LOWER_IS_BETTER:
        return False
    raise ValueError(f"Unknown metric direction for {metric}")


def mean_std_str(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize(
    input_csv: Path, out_dir: Path, exclude_task_ids: List[int] | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if exclude_task_ids:
        df = df[~df["task_id"].isin(exclude_task_ids)].copy()

    if len(df) == 0:
        raise ValueError("No successful rows found.")

    metrics = available_metrics(df)

    id_cols = [
        "suite",
        "task_id",
        "task_name",
        "task_type",
        "model",
    ]

    split_cols = [
        c for c in ["repeat", "fold", "sample"] if c in df.columns
    ]

    meta_cols = [
        c
        for c in ["n_train", "n_test", "n_features", "n_classes"]
        if c in df.columns
    ]

    # 1. Average repeated splits for each task/model.
    group_cols = id_cols
    agg_dict: Dict[str, str] = {m: "mean" for m in metrics}

    for c in meta_cols:
        agg_dict[c] = "mean"

    per_task_model = (
        df.groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    # 2. Compute ranks within each task.
    ranked = per_task_model.copy()

    for metric in metrics:
        higher_better = rank_direction(metric)

        ranked[f"rank_{metric}"] = ranked.groupby("task_id")[metric].rank(
            method="average",
            ascending=not higher_better,
        )

    rank_cols = [f"rank_{m}" for m in metrics]

    # 3. Summary by model.
    summary_rows = []

    for model, g in ranked.groupby("model"):
        row = {
            "model": model,
            "n_tasks": int(g["task_id"].nunique()),
        }

        for metric in metrics:
            row[f"mean_{metric}"] = float(g[metric].mean())
            row[f"median_{metric}"] = float(g[metric].median())
            row[f"std_{metric}"] = float(g[metric].std())

            rank_col = f"rank_{metric}"
            row[f"avg_rank_{metric}"] = float(g[rank_col].mean())
            row[f"median_rank_{metric}"] = float(g[rank_col].median())

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Choose primary sorting key.
    if "accuracy" in metrics:
        primary = "avg_rank_accuracy"
    elif "rmse" in metrics:
        primary = "avg_rank_rmse"
    elif "r2" in metrics:
        primary = "avg_rank_r2"
    else:
        primary = f"avg_rank_{metrics[0]}"

    summary = summary.sort_values(primary)

    # 4. Dataset/task metadata summary.
    task_meta_cols = ["suite", "task_id", "task_name", "task_type"] + meta_cols

    task_meta = (
        per_task_model[task_meta_cols]
        .drop_duplicates(subset=["task_id"])
        .copy()
    )

    dataset_summary = []

    for suite, g in task_meta.groupby("suite"):
        row = {
            "suite": suite,
            "n_tasks": int(g["task_id"].nunique()),
        }

        for c in meta_cols:
            row[f"min_{c}"] = float(g[c].min())
            row[f"median_{c}"] = float(g[c].median())
            row[f"max_{c}"] = float(g[c].max())

        dataset_summary.append(row)

    dataset_summary = pd.DataFrame(dataset_summary)

    # 5. Win counts by metric.
    win_rows = []

    for metric in metrics:
        rank_col = f"rank_{metric}"

        winners = ranked[ranked[rank_col] == 1.0]
        counts = winners.groupby("model")["task_id"].nunique()

        for model in ranked["model"].unique():
            win_rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "wins": int(counts.get(model, 0)),
                    "n_tasks": int(ranked["task_id"].nunique()),
                    "win_rate": float(counts.get(model, 0))
                    / float(ranked["task_id"].nunique()),
                }
            )

    wins = pd.DataFrame(win_rows).sort_values(["metric", "wins"], ascending=[True, False])

    # 6. Save files.
    ranked.to_csv(out_dir / "per_task_model_ranked.csv", index=False)
    summary.to_csv(out_dir / "summary_by_model.csv", index=False)
    dataset_summary.to_csv(out_dir / "dataset_summary.csv", index=False)
    wins.to_csv(out_dir / "wins_by_metric.csv", index=False)

    # 7. Print slide-friendly output.
    print("\n=== Dataset summary ===")
    print(dataset_summary.to_string(index=False))

    print("\n=== Summary by model ===")
    display_cols = ["model", "n_tasks"]

    for metric in metrics:
        display_cols.append(f"avg_rank_{metric}")
        display_cols.append(f"mean_{metric}")
        display_cols.append(f"median_{metric}")

    print(summary[display_cols].to_string(index=False))

    print("\n=== Wins by metric ===")
    print(wins.to_string(index=False))

    print(f"\nWrote:")
    print(f"  {out_dir / 'per_task_model_ranked.csv'}")
    print(f"  {out_dir / 'summary_by_model.csv'}")
    print(f"  {out_dir / 'dataset_summary.csv'}")
    print(f"  {out_dir / 'wins_by_metric.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to metrics.csv produced by run_openml_baselines.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to input_csv parent / summary.",
    )
    parser.add_argument(
        "--exclude-task-ids",
        type=str,
        default=None,
        help="Comma-separated OpenML task_ids to drop before summarizing "
        "(e.g. to exclude a dataset that isn't representative).",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)

    if args.out_dir is None:
        out_dir = input_csv.parent / "summary"
    else:
        out_dir = Path(args.out_dir)

    exclude_task_ids = None
    if args.exclude_task_ids:
        exclude_task_ids = [int(t) for t in args.exclude_task_ids.split(",")]

    summarize(input_csv, out_dir, exclude_task_ids=exclude_task_ids)


if __name__ == "__main__":
    main()