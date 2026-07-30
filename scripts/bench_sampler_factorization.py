from __future__ import annotations

import argparse
import time
import os
import sys
import resource
from dataclasses import dataclass

import numpy as np

# Allows running this script without installing package.
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
    MixtureSampler,
)
from tab_completion.factorization import ParallelFactorizer, PermARFactorizer


def get_max_rss_mb() -> float:
    """
    Linux: ru_maxrss is KB.
    macOS: ru_maxrss is bytes.
    This is mainly for rough server diagnostics.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e6
    return rss / 1024.0


def build_sampler(args):
    if args.sampler == "target":
        return TargetPredictionSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            target_col=args.target_col,
            replace_rows=args.replace_rows,
        )

    if args.sampler == "random_cell":
        return RandomCellSampler(
            n_episode_rows=args.n_episode_rows,
            query_frac=args.query_frac,
            max_query_cells=args.max_query_cells,
            replace_rows=args.replace_rows,
        )

    if args.sampler == "column_block":
        return ColumnBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            min_query_cols=args.min_query_cols,
            max_query_cols=args.max_query_cols,
            exclude_target=args.exclude_target,
            replace_rows=args.replace_rows,
        )

    if args.sampler == "row_block":
        return RowBlockSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            query_frac_cols=args.query_frac_cols,
            replace_rows=args.replace_rows,
        )

    if args.sampler == "label_feature":
        return LabelFeatureSampler(
            n_context=args.n_context,
            n_query=args.n_query,
            n_feature_cols=args.n_feature_cols,
            target_col=args.target_col,
            replace_rows=args.replace_rows,
        )

    if args.sampler == "mixture":
        return MixtureSampler(
            samplers=[
                TargetPredictionSampler(args.n_context, args.n_query, args.target_col),
                RandomCellSampler(args.n_episode_rows, args.query_frac, max_query_cells=args.max_query_cells),
                ColumnBlockSampler(args.n_context, args.n_query, args.min_query_cols, args.max_query_cols),
                LabelFeatureSampler(args.n_context, args.n_query, args.n_feature_cols, args.target_col),
            ],
            weights=[0.25, 0.25, 0.25, 0.25],
        )

    raise ValueError(f"Unknown sampler: {args.sampler}")


def build_factorizer(args):
    if args.factorization == "parallel":
        return ParallelFactorizer()

    if args.factorization == "perm_ar":
        return PermARFactorizer(
            unit=args.ar_unit,
            group_size=args.group_size,
        )

    raise ValueError(f"Unknown factorization: {args.factorization}")


@dataclass
class RunningStats:
    n: int = 0
    total_query_cells: int = 0
    total_observed_cells: int = 0
    total_steps: int = 0
    total_mask_mb: float = 0.0

    def update(self, task, plan) -> None:
        self.n += 1
        self.total_query_cells += task.num_query_cells
        self.total_observed_cells += task.num_observed_cells
        self.total_steps += plan.num_steps
        self.total_mask_mb += task.mask_memory_mb()

    def report(self) -> str:
        return (
            f"avg_query_cells={self.total_query_cells / self.n:.1f}, "
            f"avg_observed_cells={self.total_observed_cells / self.n:.1f}, "
            f"avg_steps={self.total_steps / self.n:.1f}, "
            f"avg_mask_mb={self.total_mask_mb / self.n:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-rows", type=int, default=1_000_000)
    parser.add_argument("--n-cols", type=int, default=100)
    parser.add_argument("--target-col", type=int, default=None)

    parser.add_argument("--sampler", type=str, default="label_feature",
                        choices=["target", "random_cell", "column_block", "row_block", "label_feature", "mixture"])
    parser.add_argument("--factorization", type=str, default="perm_ar",
                        choices=["parallel", "perm_ar"])
    parser.add_argument("--ar-unit", type=str, default="column",
                        choices=["cell", "column", "row"])

    parser.add_argument("--n-context", type=int, default=1024)
    parser.add_argument("--n-query", type=int, default=1024)
    parser.add_argument("--n-episode-rows", type=int, default=2048)

    parser.add_argument("--query-frac", type=float, default=0.15)
    parser.add_argument("--query-frac-cols", type=float, default=1.0)
    parser.add_argument("--min-query-cols", type=int, default=1)
    parser.add_argument("--max-query-cols", type=int, default=3)
    parser.add_argument("--n-feature-cols", type=int, default=2)
    parser.add_argument("--max-query-cells", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=1)

    parser.add_argument("--exclude-target", action="store_true")
    parser.add_argument("--replace-rows", action="store_true")
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)

    args = parser.parse_args()

    if args.target_col is None:
        args.target_col = args.n_cols - 1

    rng = np.random.default_rng(args.seed)

    info = TableInfo(
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        target_col=args.target_col,
    )

    sampler = build_sampler(args)
    factorizer = build_factorizer(args)

    # Warmup
    for _ in range(args.warmup):
        task = sampler.sample(info, rng)
        _ = factorizer.build(task, rng)

    stats = RunningStats()

    start = time.perf_counter()
    for _ in range(args.iters):
        task = sampler.sample(info, rng)
        plan = factorizer.build(task, rng)
        stats.update(task, plan)
    end = time.perf_counter()

    elapsed = end - start
    per_iter_ms = elapsed / args.iters * 1000.0
    throughput = args.iters / elapsed

    print("=== Sampler + factorization benchmark ===")
    print(f"n_rows={args.n_rows:,}, n_cols={args.n_cols}")
    print(f"sampler={args.sampler}")
    print(f"factorization={args.factorization}, ar_unit={args.ar_unit}")
    print(f"iters={args.iters:,}, elapsed={elapsed:.3f}s")
    print(f"per_iter_ms={per_iter_ms:.4f}")
    print(f"throughput_tasks_per_sec={throughput:.1f}")
    print(stats.report())
    print(f"max_rss_mb={get_max_rss_mb():.1f}")
    print("sample_task:", task.summary())
    print("sample_meta:", task.meta)


if __name__ == "__main__":
    main()