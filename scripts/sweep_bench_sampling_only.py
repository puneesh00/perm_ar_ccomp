from __future__ import annotations

import argparse
import os
import sys
import time
import resource
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Allows running without installing package.
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


@dataclass(frozen=True)
class BenchSpec:
    experiment: str
    sampler: str
    n_rows: int
    n_cols: int
    n_episode_rows: int
    n_feature_cols: int = 2
    query_frac: float = 0.05
    query_frac_cols: float = 1.0
    max_query_cols: int = 5


def get_max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e6
    return rss / 1024.0


def build_sampler(spec: BenchSpec, max_query_cells: int | None):
    target_col = spec.n_cols - 1
    n_context = spec.n_episode_rows // 2
    n_query = spec.n_episode_rows - n_context

    if spec.sampler == "target":
        return TargetPredictionSampler(
            n_context=n_context,
            n_query=n_query,
            target_col=target_col,
        )

    if spec.sampler == "random_cell":
        return RandomCellSampler(
            n_episode_rows=spec.n_episode_rows,
            query_frac=spec.query_frac,
            max_query_cells=max_query_cells,
        )

    if spec.sampler == "column_block":
        return ColumnBlockSampler(
            n_context=n_context,
            n_query=n_query,
            min_query_cols=1,
            max_query_cols=spec.max_query_cols,
        )

    if spec.sampler == "row_block":
        return RowBlockSampler(
            n_context=n_context,
            n_query=n_query,
            query_frac_cols=spec.query_frac_cols,
        )

    if spec.sampler == "label_feature":
        return LabelFeatureSampler(
            n_context=n_context,
            n_query=n_query,
            n_feature_cols=spec.n_feature_cols,
            target_col=target_col,
        )

    raise ValueError(f"Unknown sampler: {spec.sampler}")


def benchmark_one_spec(
    *,
    spec: BenchSpec,
    iters: int,
    warmup: int,
    seed: int,
    max_query_cells: int | None,
) -> dict:
    rng = np.random.default_rng(seed)

    info = TableInfo(
        n_rows=spec.n_rows,
        n_cols=spec.n_cols,
        target_col=spec.n_cols - 1,
    )

    sampler = build_sampler(spec, max_query_cells=max_query_cells)

    # Warmup: not timed.
    for _ in range(warmup):
        sparse_task = sampler.sample_sparse(info, rng)
        _ = sparse_task.to_dense_task()

    sparse_times = []
    dense_times = []
    total_times = []

    sparse_mb = []
    dense_mask_mb = []
    query_cells = []
    observed_cells = []

    for _ in range(iters):
        # Time sparse task sampling.
        t0 = time.perf_counter()
        sparse_task = sampler.sample_sparse(info, rng)
        t1 = time.perf_counter()

        # Time dense mask materialization separately.
        dense_task = sparse_task.to_dense_task()
        t2 = time.perf_counter()

        sparse_times.append(t1 - t0)
        dense_times.append(t2 - t1)
        total_times.append(t2 - t0)

        sparse_mb.append(sparse_task.sparse_memory_mb())
        dense_mask_mb.append(dense_task.mask_memory_mb())
        query_cells.append(dense_task.num_query_cells)
        observed_cells.append(dense_task.num_observed_cells)

    sparse_sec = float(np.sum(sparse_times))
    dense_sec = float(np.sum(dense_times))
    total_sec = float(np.sum(total_times))

    return {
        "experiment": spec.experiment,
        "sampler": spec.sampler,
        "n_rows": spec.n_rows,
        "n_cols": spec.n_cols,
        "n_episode_rows": spec.n_episode_rows,
        "n_feature_cols": spec.n_feature_cols,
        "query_frac": spec.query_frac,
        "query_frac_cols": spec.query_frac_cols,
        "max_query_cols": spec.max_query_cols,
        "iters": iters,
        "warmup": warmup,
        "sparse_sample_ms": 1000.0 * sparse_sec / iters,
        "dense_materialize_ms": 1000.0 * dense_sec / iters,
        "dense_total_ms": 1000.0 * total_sec / iters,
        "tasks_per_sec": iters / total_sec,
        "avg_sparse_mb": float(np.mean(sparse_mb)),
        "avg_dense_mask_mb": float(np.mean(dense_mask_mb)),
        "avg_query_cells": float(np.mean(query_cells)),
        "avg_observed_cells": float(np.mean(observed_cells)),
        "max_rss_mb": get_max_rss_mb(),
    }


def local_iters_for_spec(spec: BenchSpec, base_iters: int, base_warmup: int) -> tuple[int, int]:
    """
    Use fewer iterations for expensive samplers.
    """
    iters = base_iters
    warmup = base_warmup

    # Row-block can create many queried cells.
    if spec.sampler == "row_block" and spec.n_episode_rows >= 2048:
        iters = max(50, base_iters // 5)
        warmup = max(10, base_warmup // 3)

    if spec.sampler == "row_block" and spec.n_episode_rows >= 4096:
        iters = max(30, base_iters // 10)
        warmup = max(5, base_warmup // 5)

    return iters, warmup


def run_specs(
    specs: list[BenchSpec],
    *,
    iters: int,
    warmup: int,
    repeats: int,
    seed: int,
    max_query_cells: int | None,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    total_runs = len(specs) * repeats
    run_id = 0

    for spec_id, spec in enumerate(specs):
        local_iters, local_warmup = local_iters_for_spec(spec, iters, warmup)

        for repeat in range(repeats):
            run_id += 1

            print(
                f"[{run_id}/{total_runs}] "
                f"experiment={spec.experiment}, sampler={spec.sampler}, "
                f"n_rows={spec.n_rows:,}, n_cols={spec.n_cols}, "
                f"episode_rows={spec.n_episode_rows}, "
                f"n_feature_cols={spec.n_feature_cols}, "
                f"query_frac={spec.query_frac}, "
                f"repeat={repeat}, iters={local_iters}",
                flush=True,
            )

            t_start = time.perf_counter()

            row = benchmark_one_spec(
                spec=spec,
                iters=local_iters,
                warmup=local_warmup,
                seed=seed + 100_000 * spec_id + repeat,
                max_query_cells=max_query_cells,
            )
            row["repeat"] = repeat
            row["requested_iters"] = iters
            row["actual_iters"] = local_iters
            rows.append(row)

            elapsed = time.perf_counter() - t_start

            print(
                f"  done in {elapsed:.2f}s | "
                f"sparse_ms={row['sparse_sample_ms']:.4f} | "
                f"dense_ms={row['dense_materialize_ms']:.4f} | "
                f"total_ms={row['dense_total_ms']:.4f} | "
                f"query={row['avg_query_cells']:.1f} | "
                f"dense_mask_mb={row['avg_dense_mask_mb']:.4f}",
                flush=True,
            )

            pd.DataFrame(rows).to_csv(out_dir / "sampling_only_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sampling_only_raw.csv", index=False)
    return df


def aggregate(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
    )
    out["std"] = out["std"].fillna(0.0)
    return out


def plot_line_mean_std(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
):
    plt.figure(figsize=(8, 5))

    if hue is None:
        g = aggregate(df, [x], y).sort_values(x)
        plt.errorbar(g[x], g["mean"], yerr=g["std"], marker="o", capsize=3)
    else:
        for key, sub in df.groupby(hue):
            g = aggregate(sub, [x], y).sort_values(x)
            plt.errorbar(
                g[x],
                g["mean"],
                yerr=g["std"],
                marker="o",
                capsize=3,
                label=str(key),
            )
        plt.legend(title=hue)

    if logx:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar_mean_std(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    out_path: Path,
    title: str,
    xlabel: str,
):
    g = aggregate(df, [x], y).sort_values("mean")

    plt.figure(figsize=(8, 5))
    plt.barh(g[x].astype(str), g["mean"], xerr=g["std"], capsize=3)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_specs(profile: str) -> list[BenchSpec]:
    if profile == "quick":
        full_rows_grid = [20_000, 100_000, 1_000_000]
        n_cols_grid = [50, 100, 200]
        episode_rows_grid = [512, 1024, 2048]
        n_feature_cols_grid = [1, 2, 4, 8]
        query_frac_grid = [0.01, 0.05, 0.10]
    elif profile == "full":
        full_rows_grid = [20_000, 100_000, 1_000_000, 10_000_000]
        n_cols_grid = [50, 100, 200, 500]
        episode_rows_grid = [512, 1024, 2048, 4096]
        n_feature_cols_grid = [1, 2, 4, 8, 16]
        query_frac_grid = [0.01, 0.05, 0.10, 0.20]
    else:
        raise ValueError(f"Unknown profile: {profile}")

    fixed_n_rows = 1_000_000
    fixed_n_cols = 100
    fixed_episode_rows = 2048
    fixed_sampler = "label_feature"
    fixed_n_feature_cols = 2

    specs: list[BenchSpec] = []

    # 1. Full table row scaling.
    # Fixed: sampler, columns, episode size.
    for n_rows in full_rows_grid:
        specs.append(
            BenchSpec(
                experiment="full_rows_scaling",
                sampler=fixed_sampler,
                n_rows=n_rows,
                n_cols=fixed_n_cols,
                n_episode_rows=fixed_episode_rows,
                n_feature_cols=fixed_n_feature_cols,
            )
        )

    # 2. Episode row scaling.
    # Fixed: full rows. Vary episode rows and columns.
    for n_cols in n_cols_grid:
        for n_episode_rows in episode_rows_grid:
            specs.append(
                BenchSpec(
                    experiment="episode_rows_scaling",
                    sampler=fixed_sampler,
                    n_rows=fixed_n_rows,
                    n_cols=n_cols,
                    n_episode_rows=n_episode_rows,
                    n_feature_cols=fixed_n_feature_cols,
                )
            )

    # 3. Column scaling.
    # Fixed: full rows. Vary columns and episode rows.
    for n_episode_rows in episode_rows_grid:
        for n_cols in n_cols_grid:
            specs.append(
                BenchSpec(
                    experiment="columns_scaling",
                    sampler=fixed_sampler,
                    n_rows=fixed_n_rows,
                    n_cols=n_cols,
                    n_episode_rows=n_episode_rows,
                    n_feature_cols=fixed_n_feature_cols,
                )
            )

    # 4. Sampler comparison.
    # Fixed: full rows, columns, episode rows.
    for sampler in ["target", "random_cell", "column_block", "row_block", "label_feature"]:
        specs.append(
            BenchSpec(
                experiment="sampler_comparison",
                sampler=sampler,
                n_rows=fixed_n_rows,
                n_cols=fixed_n_cols,
                n_episode_rows=fixed_episode_rows,
                n_feature_cols=fixed_n_feature_cols,
            )
        )

    # 5. Label-feature query-size scaling.
    # Fixed: full rows, columns, episode rows. Vary sampled feature columns.
    for n_feature_cols in n_feature_cols_grid:
        specs.append(
            BenchSpec(
                experiment="label_feature_cols_scaling",
                sampler="label_feature",
                n_rows=fixed_n_rows,
                n_cols=fixed_n_cols,
                n_episode_rows=fixed_episode_rows,
                n_feature_cols=n_feature_cols,
            )
        )

    # 6. Random-cell mask fraction scaling.
    # Fixed: full rows, columns, episode rows. Vary query_frac.
    for query_frac in query_frac_grid:
        specs.append(
            BenchSpec(
                experiment="random_cell_query_frac_scaling",
                sampler="random_cell",
                n_rows=fixed_n_rows,
                n_cols=fixed_n_cols,
                n_episode_rows=fixed_episode_rows,
                query_frac=query_frac,
            )
        )

    return specs


def make_plots(df_raw: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full rows scaling.
    sub = df_raw[df_raw["experiment"] == "full_rows_scaling"]
    if len(sub) > 0:
        plot_line_mean_std(
            sub,
            x="n_rows",
            y="sparse_sample_ms",
            hue=None,
            out_path=out_dir / "sparse_sampling_time_vs_full_rows.png",
            title="Sparse sampling time vs full table rows\nfixed: sampler=label_feature, columns=100, episode_rows=2048",
            xlabel="full table rows",
            ylabel="ms per sampled task",
            logx=True,
        )
        plot_line_mean_std(
            sub,
            x="n_rows",
            y="dense_materialize_ms",
            hue=None,
            out_path=out_dir / "dense_materialization_time_vs_full_rows.png",
            title="Dense mask materialization time vs full table rows\nfixed: sampler=label_feature, columns=100, episode_rows=2048",
            xlabel="full table rows",
            ylabel="ms per task",
            logx=True,
        )

    # Episode rows scaling.
    sub = df_raw[df_raw["experiment"] == "episode_rows_scaling"]
    if len(sub) > 0:
        plot_line_mean_std(
            sub,
            x="n_episode_rows",
            y="sparse_sample_ms",
            hue="n_cols",
            out_path=out_dir / "sparse_sampling_time_vs_episode_rows.png",
            title="Sparse sampling time vs sampled episode rows\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="sampled episode rows",
            ylabel="ms per sampled task",
        )
        plot_line_mean_std(
            sub,
            x="n_episode_rows",
            y="dense_materialize_ms",
            hue="n_cols",
            out_path=out_dir / "dense_materialization_time_vs_episode_rows.png",
            title="Dense mask materialization time vs sampled episode rows\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="sampled episode rows",
            ylabel="ms per task",
        )
        plot_line_mean_std(
            sub,
            x="n_episode_rows",
            y="avg_dense_mask_mb",
            hue="n_cols",
            out_path=out_dir / "dense_mask_memory_vs_episode_rows.png",
            title="Dense mask memory vs sampled episode rows\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="sampled episode rows",
            ylabel="MB per task",
        )
        plot_line_mean_std(
            sub,
            x="n_episode_rows",
            y="avg_sparse_mb",
            hue="n_cols",
            out_path=out_dir / "sparse_memory_vs_episode_rows.png",
            title="Sparse task memory vs sampled episode rows\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="sampled episode rows",
            ylabel="MB per task",
        )

    # Column scaling.
    sub = df_raw[df_raw["experiment"] == "columns_scaling"]
    if len(sub) > 0:
        plot_line_mean_std(
            sub,
            x="n_cols",
            y="sparse_sample_ms",
            hue="n_episode_rows",
            out_path=out_dir / "sparse_sampling_time_vs_columns.png",
            title="Sparse sampling time vs number of columns\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="columns",
            ylabel="ms per sampled task",
        )
        plot_line_mean_std(
            sub,
            x="n_cols",
            y="dense_materialize_ms",
            hue="n_episode_rows",
            out_path=out_dir / "dense_materialization_time_vs_columns.png",
            title="Dense mask materialization time vs number of columns\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="columns",
            ylabel="ms per task",
        )
        plot_line_mean_std(
            sub,
            x="n_cols",
            y="avg_dense_mask_mb",
            hue="n_episode_rows",
            out_path=out_dir / "dense_mask_memory_vs_columns.png",
            title="Dense mask memory vs number of columns\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="columns",
            ylabel="MB per task",
        )
        plot_line_mean_std(
            sub,
            x="n_cols",
            y="avg_sparse_mb",
            hue="n_episode_rows",
            out_path=out_dir / "sparse_memory_vs_columns.png",
            title="Sparse task memory vs number of columns\nfixed: sampler=label_feature, full_rows=1,000,000",
            xlabel="columns",
            ylabel="MB per task",
        )

    # Sampler comparison.
    sub = df_raw[df_raw["experiment"] == "sampler_comparison"]
    if len(sub) > 0:
        plot_bar_mean_std(
            sub,
            x="sampler",
            y="sparse_sample_ms",
            out_path=out_dir / "sparse_sampling_time_by_sampler.png",
            title="Sparse sampling time by sampler\nfixed: full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="ms per sampled task",
        )
        plot_bar_mean_std(
            sub,
            x="sampler",
            y="dense_materialize_ms",
            out_path=out_dir / "dense_materialization_time_by_sampler.png",
            title="Dense mask materialization time by sampler\nfixed: full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="ms per task",
        )
        plot_bar_mean_std(
            sub,
            x="sampler",
            y="avg_query_cells",
            out_path=out_dir / "query_cells_by_sampler.png",
            title="Average queried cells by sampler\nfixed: full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="queried cells per task",
        )
        plot_bar_mean_std(
            sub,
            x="sampler",
            y="avg_dense_mask_mb",
            out_path=out_dir / "dense_mask_memory_by_sampler.png",
            title="Dense mask memory by sampler\nfixed: full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="MB per task",
        )
        plot_bar_mean_std(
            sub,
            x="sampler",
            y="avg_sparse_mb",
            out_path=out_dir / "sparse_memory_by_sampler.png",
            title="Sparse task memory by sampler\nfixed: full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="MB per task",
        )

    # Label-feature number of sampled feature columns.
    sub = df_raw[df_raw["experiment"] == "label_feature_cols_scaling"]
    if len(sub) > 0:
        plot_line_mean_std(
            sub,
            x="n_feature_cols",
            y="sparse_sample_ms",
            hue=None,
            out_path=out_dir / "sparse_sampling_time_vs_n_feature_cols.png",
            title="Sparse sampling time vs sampled feature columns\nfixed: sampler=label_feature, full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="sampled feature columns",
            ylabel="ms per sampled task",
        )
        plot_line_mean_std(
            sub,
            x="n_feature_cols",
            y="avg_query_cells",
            hue=None,
            out_path=out_dir / "query_cells_vs_n_feature_cols.png",
            title="Queried cells vs sampled feature columns\nfixed: sampler=label_feature, full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="sampled feature columns",
            ylabel="queried cells per task",
        )

    # Random-cell query fraction.
    sub = df_raw[df_raw["experiment"] == "random_cell_query_frac_scaling"]
    if len(sub) > 0:
        plot_line_mean_std(
            sub,
            x="query_frac",
            y="sparse_sample_ms",
            hue=None,
            out_path=out_dir / "sparse_sampling_time_vs_random_cell_query_frac.png",
            title="Sparse sampling time vs random-cell query fraction\nfixed: sampler=random_cell, full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="query fraction",
            ylabel="ms per sampled task",
        )
        plot_line_mean_std(
            sub,
            x="query_frac",
            y="avg_query_cells",
            hue=None,
            out_path=out_dir / "query_cells_vs_random_cell_query_frac.png",
            title="Queried cells vs random-cell query fraction\nfixed: sampler=random_cell, full_rows=1,000,000, columns=100, episode_rows=2048",
            xlabel="query fraction",
            ylabel="queried cells per task",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default="results/sampling_only_sparse")
    parser.add_argument("--profile", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-query-cells", type=int, default=5000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = make_specs(args.profile)

    print(f"Running {len(specs)} benchmark specs with repeats={args.repeats}")
    print(f"Output directory: {out_dir}")

    df_raw = run_specs(
        specs,
        iters=args.iters,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        max_query_cells=args.max_query_cells,
        out_dir=out_dir,
    )

    df_raw.to_csv(out_dir / "sampling_only_raw.csv", index=False)

    summary = (
        df_raw.groupby(
            ["experiment", "sampler", "n_rows", "n_cols", "n_episode_rows"],
            as_index=False,
        )
        .agg(
            sparse_sample_ms_mean=("sparse_sample_ms", "mean"),
            sparse_sample_ms_std=("sparse_sample_ms", "std"),
            dense_materialize_ms_mean=("dense_materialize_ms", "mean"),
            dense_materialize_ms_std=("dense_materialize_ms", "std"),
            dense_total_ms_mean=("dense_total_ms", "mean"),
            dense_total_ms_std=("dense_total_ms", "std"),
            tasks_per_sec_mean=("tasks_per_sec", "mean"),
            avg_sparse_mb_mean=("avg_sparse_mb", "mean"),
            avg_dense_mask_mb_mean=("avg_dense_mask_mb", "mean"),
            avg_query_cells_mean=("avg_query_cells", "mean"),
            avg_observed_cells_mean=("avg_observed_cells", "mean"),
            max_rss_mb_mean=("max_rss_mb", "mean"),
        )
    )
    summary.to_csv(out_dir / "sampling_only_summary.csv", index=False)

    make_plots(df_raw, out_dir)

    print("\nDone.")
    print(f"Raw CSV: {out_dir / 'sampling_only_raw.csv'}")
    print(f"Summary CSV: {out_dir / 'sampling_only_summary.csv'}")
    print(f"Plots: {out_dir}")


if __name__ == "__main__":
    main()