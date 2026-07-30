from __future__ import annotations

import argparse
import os
import sys
import time
import resource
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tab_completion.sampling import (
    TableInfo,
    TargetPredictionSampler,
    RandomCellSampler,
    ColumnBlockSampler,
    RowBlockSampler,
    LabelFeatureSampler,
)
from tab_completion.factorization import ParallelFactorizer, PermARFactorizer


def get_max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e6
    return rss / 1024.0


def build_sampler(
    name: str,
    n_episode_rows: int,
    n_context: int,
    n_query: int,
    n_feature_cols: int,
    n_cols_query_max: int,
    query_frac: float,
    max_query_cells: int | None,
    target_col: int,
):
    if name == "target":
        return TargetPredictionSampler(
            n_context=n_context,
            n_query=n_query,
            target_col=target_col,
        )

    if name == "random_cell":
        return RandomCellSampler(
            n_episode_rows=n_episode_rows,
            query_frac=query_frac,
            max_query_cells=max_query_cells,
        )

    if name == "column_block":
        return ColumnBlockSampler(
            n_context=n_context,
            n_query=n_query,
            min_query_cols=1,
            max_query_cols=n_cols_query_max,
        )

    if name == "row_block":
        return RowBlockSampler(
            n_context=n_context,
            n_query=n_query,
            query_frac_cols=1.0,
        )

    if name == "label_feature":
        return LabelFeatureSampler(
            n_context=n_context,
            n_query=n_query,
            n_feature_cols=n_feature_cols,
            target_col=target_col,
        )

    raise ValueError(f"Unknown sampler: {name}")


def build_factorizer(name: str, ar_unit: str):
    if name == "parallel":
        return ParallelFactorizer()

    if name == "perm_ar":
        return PermARFactorizer(unit=ar_unit)

    raise ValueError(f"Unknown factorization: {name}")


def bench_one(
    *,
    n_rows: int,
    n_cols: int,
    n_episode_rows: int,
    sampler_name: str,
    factorization_name: str,
    ar_unit: str,
    query_frac: float,
    max_query_cells: int | None,
    n_feature_cols: int,
    n_cols_query_max: int,
    iters: int,
    warmup: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    target_col = n_cols - 1

    # For samplers that split context/query rows.
    n_context = n_episode_rows // 2
    n_query = n_episode_rows - n_context

    info = TableInfo(
        n_rows=n_rows,
        n_cols=n_cols,
        target_col=target_col,
    )

    sampler = build_sampler(
        name=sampler_name,
        n_episode_rows=n_episode_rows,
        n_context=n_context,
        n_query=n_query,
        n_feature_cols=n_feature_cols,
        n_cols_query_max=n_cols_query_max,
        query_frac=query_frac,
        max_query_cells=max_query_cells,
        target_col=target_col,
    )
    factorizer = build_factorizer(factorization_name, ar_unit)

    # Warmup.
    for _ in range(warmup):
        task = sampler.sample(info, rng)
        _ = factorizer.build(task, rng)

    query_cells = []
    observed_cells = []
    steps = []
    mask_mb = []

    start = time.perf_counter()
    for _ in range(iters):
        task = sampler.sample(info, rng)
        plan = factorizer.build(task, rng)

        query_cells.append(task.num_query_cells)
        observed_cells.append(task.num_observed_cells)
        steps.append(plan.num_steps)
        mask_mb.append(task.mask_memory_mb())

    elapsed = time.perf_counter() - start

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_episode_rows": n_episode_rows,
        "sampler": sampler_name,
        "factorization": factorization_name,
        "ar_unit": ar_unit if factorization_name == "perm_ar" else "none",
        "query_frac": query_frac,
        "max_query_cells": max_query_cells if max_query_cells is not None else -1,
        "n_feature_cols": n_feature_cols,
        "n_cols_query_max": n_cols_query_max,
        "iters": iters,
        "elapsed_sec": elapsed,
        "per_iter_ms": 1000.0 * elapsed / iters,
        "tasks_per_sec": iters / elapsed,
        "avg_query_cells": float(np.mean(query_cells)),
        "avg_observed_cells": float(np.mean(observed_cells)),
        "avg_steps": float(np.mean(steps)),
        "avg_mask_mb": float(np.mean(mask_mb)),
        "max_rss_mb": get_max_rss_mb(),
    }


def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    out_path: Path,
    title: str,
    ylabel: str,
):
    plt.figure(figsize=(8, 5))

    for key, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        plt.plot(sub[x], sub[y], marker="o", label=str(key))

    plt.xscale("log" if df[x].max() / max(df[x].min(), 1) >= 100 else "linear")
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: Path,
    title: str,
    ylabel: str,
):
    grouped = df.groupby(x)[y].mean().sort_values()
    plt.figure(figsize=(8, 5))
    grouped.plot(kind="barh")
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Does full n_rows matter?
    sub = df[
        (df["sampler"] == "label_feature")
        & (df["factorization"] == "perm_ar")
        & (df["ar_unit"] == "column")
    ]
    if len(sub) > 0:
        plot_line(
            sub,
            x="n_rows",
            y="per_iter_ms",
            hue="n_episode_rows",
            out_path=out_dir / "time_vs_full_rows.png",
            title="Sampler/factorizer time vs full table rows",
            ylabel="ms per task",
        )

    # 2. Episode rows scaling.
    sub = df[
        (df["sampler"] == "label_feature")
        & (df["factorization"] == "perm_ar")
        & (df["ar_unit"] == "column")
    ]
    if len(sub) > 0:
        plot_line(
            sub,
            x="n_episode_rows",
            y="per_iter_ms",
            hue="n_cols",
            out_path=out_dir / "time_vs_episode_rows.png",
            title="Time vs sampled episode rows",
            ylabel="ms per task",
        )

        plot_line(
            sub,
            x="n_episode_rows",
            y="avg_mask_mb",
            hue="n_cols",
            out_path=out_dir / "mask_memory_vs_episode_rows.png",
            title="Mask memory vs sampled episode rows",
            ylabel="MB per task",
        )

    # 3. Columns scaling.
    sub = df[
        (df["sampler"] == "label_feature")
        & (df["factorization"] == "perm_ar")
        & (df["ar_unit"] == "column")
    ]
    if len(sub) > 0:
        plot_line(
            sub,
            x="n_cols",
            y="per_iter_ms",
            hue="n_episode_rows",
            out_path=out_dir / "time_vs_columns.png",
            title="Time vs number of columns",
            ylabel="ms per task",
        )

    # 4. Sampler comparison.
    sub = df[
        (df["n_rows"] == df["n_rows"].max())
        & (df["n_cols"] == 100)
        & (df["n_episode_rows"] == 2048)
        & (df["factorization"] == "parallel")
    ]
    if len(sub) > 0:
        plot_bar(
            sub,
            x="sampler",
            y="per_iter_ms",
            out_path=out_dir / "sampler_parallel_time_comparison.png",
            title="Parallel factorization: sampler time comparison",
            ylabel="ms per task",
        )

    # 5. Factorization comparison.
    sub = df[
        (df["sampler"] == "label_feature")
        & (df["n_cols"] == 100)
        & (df["n_episode_rows"] == 2048)
    ].copy()
    if len(sub) > 0:
        sub["factorization_label"] = sub["factorization"] + "/" + sub["ar_unit"]
        plot_bar(
            sub,
            x="factorization_label",
            y="per_iter_ms",
            out_path=out_dir / "factorization_time_comparison.png",
            title="Factorization time comparison",
            ylabel="ms per task",
        )

        plot_bar(
            sub,
            x="factorization_label",
            y="avg_steps",
            out_path=out_dir / "factorization_steps_comparison.png",
            title="Average number of prediction steps",
            ylabel="steps per task",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default="results/sampler_sweep")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows_grid = [20_000, 50_000, 100_000, 1_000_000]
    n_cols_grid = [50, 100, 200]
    n_episode_rows_grid = [512, 1024, 2048, 4096]

    sampler_grid = [
        "target",
        "random_cell",
        "column_block",
        "row_block",
        "label_feature",
    ]

    factorization_grid = [
        ("parallel", "none"),
        ("perm_ar", "column"),
        ("perm_ar", "cell"),
    ]

    rows = []
    run_id = 0

    for n_rows, n_cols, n_episode_rows, sampler_name, (factorization_name, ar_unit) in product(
        n_rows_grid,
        n_cols_grid,
        n_episode_rows_grid,
        sampler_grid,
        factorization_grid,
    ):
        # Avoid huge cell-level AR sweeps; it is intentionally expensive.
        if factorization_name == "perm_ar" and ar_unit == "cell":
            if sampler_name in {"random_cell", "row_block"} and n_episode_rows > 1024:
                continue

        run_id += 1
        print(
            f"[{run_id}] n_rows={n_rows:,}, n_cols={n_cols}, "
            f"episode_rows={n_episode_rows}, sampler={sampler_name}, "
            f"factorization={factorization_name}/{ar_unit}"
        )

        row = bench_one(
            n_rows=n_rows,
            n_cols=n_cols,
            n_episode_rows=n_episode_rows,
            sampler_name=sampler_name,
            factorization_name=factorization_name,
            ar_unit=ar_unit,
            query_frac=0.05,
            max_query_cells=5000,
            n_feature_cols=2,
            n_cols_query_max=5,
            iters=args.iters,
            warmup=args.warmup,
            seed=args.seed + run_id,
        )
        rows.append(row)

        # Save incrementally.
        pd.DataFrame(rows).to_csv(out_dir / "sampler_factorization_sweep.csv", index=False)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sampler_factorization_sweep.csv"
    df.to_csv(csv_path, index=False)

    make_plots(df, out_dir)

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()