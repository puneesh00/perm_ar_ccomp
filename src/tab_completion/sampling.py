from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Protocol, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class TableInfo:
    """
    Lightweight metadata about the full table.

    We intentionally do NOT create full-table masks of shape [n_rows, n_cols].
    Samplers first select a small episode of rows, then create local task objects.
    """
    n_rows: int
    n_cols: int
    target_col: Optional[int] = None
    col_types: Optional[np.ndarray] = None
    name: str = "table"


@dataclass
class CompletionTask:
    """
    Dense sampled conditional-completion task.

    row_idx:
        Global row indices included in this episode. Shape [n_episode_rows].

    col_idx:
        Global column indices included in this episode. Shape [n_episode_cols].

    observed_mask:
        Local mask over [n_episode_rows, n_episode_cols].
        True means observed evidence O.

    query_mask:
        Local mask over [n_episode_rows, n_episode_cols].
        True means queried target M.

    Cells outside row_idx x col_idx are neither observed nor queried in this episode.
    """
    row_idx: np.ndarray
    col_idx: np.ndarray
    observed_mask: np.ndarray
    query_mask: np.ndarray
    task_name: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.row_idx = np.asarray(self.row_idx, dtype=np.int64)
        self.col_idx = np.asarray(self.col_idx, dtype=np.int64)
        self.observed_mask = np.asarray(self.observed_mask, dtype=bool)
        self.query_mask = np.asarray(self.query_mask, dtype=bool)

        expected_shape = (len(self.row_idx), len(self.col_idx))

        if self.observed_mask.shape != expected_shape:
            raise ValueError(
                f"observed_mask shape {self.observed_mask.shape} "
                f"does not match episode shape {expected_shape}."
            )

        if self.query_mask.shape != expected_shape:
            raise ValueError(
                f"query_mask shape {self.query_mask.shape} "
                f"does not match episode shape {expected_shape}."
            )

        if np.any(self.observed_mask & self.query_mask):
            raise ValueError("A cell cannot be both observed and queried.")

        if not np.any(self.query_mask):
            raise ValueError("CompletionTask has no queried cells.")

    @property
    def n_episode_rows(self) -> int:
        return len(self.row_idx)

    @property
    def n_episode_cols(self) -> int:
        return len(self.col_idx)

    @property
    def num_query_cells(self) -> int:
        return int(self.query_mask.sum())

    @property
    def num_observed_cells(self) -> int:
        return int(self.observed_mask.sum())

    def query_coords_local(self) -> np.ndarray:
        """
        Returns local queried coordinates.
        Shape: [num_query_cells, 2], columns are [local_row, local_col].
        """
        return np.argwhere(self.query_mask)

    def observed_coords_local(self) -> np.ndarray:
        """
        Returns local observed coordinates.
        Shape: [num_observed_cells, 2], columns are [local_row, local_col].
        """
        return np.argwhere(self.observed_mask)

    def local_to_global_coords(self, coords_local: np.ndarray) -> np.ndarray:
        """
        Convert local episode coordinates to global table coordinates.

        coords_local:
            [k, 2] array, columns are [local_row, local_col].

        returns:
            [k, 2] array, columns are [global_row, global_col].
        """
        coords_local = np.asarray(coords_local, dtype=np.int64)
        rows = self.row_idx[coords_local[:, 0]]
        cols = self.col_idx[coords_local[:, 1]]
        return np.stack([rows, cols], axis=1)

    def mask_memory_mb(self) -> float:
        """
        Memory used by dense observed/query masks only.
        Does not include table values, tokens, model activations, etc.
        """
        return (self.observed_mask.nbytes + self.query_mask.nbytes) / 1e6

    def summary(self) -> str:
        return (
            f"CompletionTask(name={self.task_name}, "
            f"rows={self.n_episode_rows}, cols={self.n_episode_cols}, "
            f"observed={self.num_observed_cells}, "
            f"query={self.num_query_cells}, "
            f"mask_mb={self.mask_memory_mb():.3f})"
        )


@dataclass
class SparseCompletionTask:
    """
    Sparse sampled conditional-completion task.

    This stores only:
      - sampled episode rows,
      - sampled episode columns,
      - queried local coordinates M.

    It does NOT immediately build dense observed/query masks.

    observed_mode:
        "all_except_query":
            all cells in the episode are observed except query_coords.

        "explicit":
            observed_coords explicitly specifies observed cells.
    """
    row_idx: np.ndarray
    col_idx: np.ndarray
    query_coords: np.ndarray  # [num_query_cells, 2], local row/col coords
    observed_mode: str = "all_except_query"
    observed_coords: Optional[np.ndarray] = None
    task_name: str = "task"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.row_idx = np.asarray(self.row_idx, dtype=np.int64)
        self.col_idx = np.asarray(self.col_idx, dtype=np.int64)
        self.query_coords = np.asarray(self.query_coords, dtype=np.int64)

        if self.query_coords.ndim != 2 or self.query_coords.shape[1] != 2:
            raise ValueError("query_coords must have shape [num_query_cells, 2].")

        n, d = self.shape

        if len(self.query_coords) == 0:
            raise ValueError("SparseCompletionTask has no queried cells.")

        if np.any(self.query_coords[:, 0] < 0) or np.any(self.query_coords[:, 0] >= n):
            raise ValueError("query row index out of local episode bounds.")

        if np.any(self.query_coords[:, 1] < 0) or np.any(self.query_coords[:, 1] >= d):
            raise ValueError("query col index out of local episode bounds.")

        if self.observed_mode not in {"all_except_query", "explicit"}:
            raise ValueError(f"Unknown observed_mode: {self.observed_mode}")

        if self.observed_mode == "explicit":
            if self.observed_coords is None:
                raise ValueError("observed_coords required for observed_mode='explicit'.")
            self.observed_coords = np.asarray(self.observed_coords, dtype=np.int64)
            if self.observed_coords.ndim != 2 or self.observed_coords.shape[1] != 2:
                raise ValueError("observed_coords must have shape [num_observed_cells, 2].")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.row_idx), len(self.col_idx))

    @property
    def n_episode_rows(self) -> int:
        return len(self.row_idx)

    @property
    def n_episode_cols(self) -> int:
        return len(self.col_idx)

    @property
    def num_query_cells(self) -> int:
        return int(len(self.query_coords))

    def query_coords_local(self) -> np.ndarray:
        return self.query_coords

    def sparse_memory_mb(self) -> float:
        """
        Memory used by sparse coordinates only.
        This is usually much smaller than dense mask memory.
        """
        total = self.row_idx.nbytes + self.col_idx.nbytes + self.query_coords.nbytes
        if self.observed_coords is not None:
            total += self.observed_coords.nbytes
        return total / 1e6

    def dense_mask_memory_mb_estimate(self) -> float:
        """
        Estimate memory that would be used by dense observed/query masks.
        Two bool arrays: observed_mask + query_mask.
        """
        n, d = self.shape
        return (2 * n * d) / 1e6

    def to_dense_task(self) -> CompletionTask:
        """
        Materialize dense observed_mask/query_mask.

        This is the part that scales as n_episode_rows * n_cols.
        """
        n, d = self.shape

        query = np.zeros((n, d), dtype=bool)
        query[self.query_coords[:, 0], self.query_coords[:, 1]] = True

        if self.observed_mode == "all_except_query":
            observed = ~query
        elif self.observed_mode == "explicit":
            observed = np.zeros((n, d), dtype=bool)
            assert self.observed_coords is not None
            observed[self.observed_coords[:, 0], self.observed_coords[:, 1]] = True
        else:
            raise ValueError(f"Unknown observed_mode: {self.observed_mode}")

        return CompletionTask(
            row_idx=self.row_idx,
            col_idx=self.col_idx,
            observed_mask=observed,
            query_mask=query,
            task_name=self.task_name,
            meta=dict(self.meta),
        )

    def summary(self) -> str:
        return (
            f"SparseCompletionTask(name={self.task_name}, "
            f"rows={self.n_episode_rows}, cols={self.n_episode_cols}, "
            f"query={self.num_query_cells}, "
            f"sparse_mb={self.sparse_memory_mb():.4f}, "
            f"dense_mask_mb_est={self.dense_mask_memory_mb_estimate():.4f})"
        )


@runtime_checkable
class TaskSampler(Protocol):
    """
    Dense sampler interface.

    sample(...) returns CompletionTask with dense observed/query masks.
    """
    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        ...


@runtime_checkable
class SparseTaskSampler(Protocol):
    """
    Optional sparse sampler interface.

    sample_sparse(...) returns SparseCompletionTask with query coordinates only.
    """
    def sample_sparse(
        self,
        info: TableInfo,
        rng: np.random.Generator,
    ) -> SparseCompletionTask:
        ...


def sample_unique_rows_fast(
    rng: np.random.Generator,
    n_rows: int,
    n_episode_rows: int,
    replace: bool = False,
) -> np.ndarray:
    """
    Fast row sampler.

    For huge n_rows and modest n_episode_rows, this avoids constructing
    a full permutation of n_rows.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    if n_episode_rows <= 0:
        raise ValueError("n_episode_rows must be positive.")

    if replace:
        return rng.integers(0, n_rows, size=n_episode_rows, dtype=np.int64)

    if n_episode_rows > n_rows:
        raise ValueError(
            f"Cannot sample {n_episode_rows} unique rows from {n_rows} rows."
        )

    # For relatively dense sampling, numpy choice is fine.
    if n_rows <= 10 * n_episode_rows:
        return rng.choice(n_rows, size=n_episode_rows, replace=False).astype(np.int64)

    # For sparse sampling from a huge table, use oversampling + unique.
    vals = rng.integers(0, n_rows, size=n_episode_rows, dtype=np.int64)
    vals = np.unique(vals)

    while len(vals) < n_episode_rows:
        needed = n_episode_rows - len(vals)
        extra = rng.integers(0, n_rows, size=max(2 * needed, 16), dtype=np.int64)
        vals = np.unique(np.concatenate([vals, extra]))

    rng.shuffle(vals)
    return vals[:n_episode_rows].astype(np.int64)


def all_columns(info: TableInfo) -> np.ndarray:
    if info.n_cols <= 0:
        raise ValueError("n_cols must be positive.")
    return np.arange(info.n_cols, dtype=np.int64)


def global_col_to_local(col_idx: np.ndarray, global_col: int) -> int:
    hits = np.where(col_idx == global_col)[0]
    if len(hits) != 1:
        raise ValueError(f"Global column {global_col} not found in col_idx.")
    return int(hits[0])


def make_grid_coords(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """
    Cartesian product of local rows and local cols.

    returns:
        [len(rows) * len(cols), 2] array of [row, col] coords.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)


def flat_indices_to_coords(flat: np.ndarray, n_cols: int) -> np.ndarray:
    """
    Convert flat indices into [row, col] local coordinates.
    """
    flat = np.asarray(flat, dtype=np.int64)
    rows = flat // n_cols
    cols = flat % n_cols
    return np.stack([rows, cols], axis=1).astype(np.int64)


@dataclass
class TargetPredictionSampler:
    """
    Context rows: all cells observed, including target.
    Query rows: features observed, target queried.
    """
    n_context: int
    n_query: int
    target_col: Optional[int] = None
    replace_rows: bool = False
    task_name: str = "target_prediction"

    def sample_sparse(self, info: TableInfo, rng: np.random.Generator) -> SparseCompletionTask:
        target_col = self.target_col if self.target_col is not None else info.target_col
        if target_col is None:
            raise ValueError("TargetPredictionSampler requires target_col.")

        n_ep = self.n_context + self.n_query
        row_idx = sample_unique_rows_fast(rng, info.n_rows, n_ep, self.replace_rows)
        col_idx = all_columns(info)

        target_local = global_col_to_local(col_idx, target_col)
        query_rows = np.arange(self.n_context, n_ep, dtype=np.int64)
        query_cols = np.array([target_local], dtype=np.int64)

        query_coords = make_grid_coords(query_rows, query_cols)

        return SparseCompletionTask(
            row_idx=row_idx,
            col_idx=col_idx,
            query_coords=query_coords,
            observed_mode="all_except_query",
            task_name=self.task_name,
            meta={
                "target_col": target_col,
                "n_context": self.n_context,
                "n_query": self.n_query,
                "conditioning_mode": "inductive_rows",
                "context_rows_local": np.arange(self.n_context, dtype=np.int64),
                "query_rows_local": np.arange(self.n_context, n_ep, dtype=np.int64),
                "sparse": True,
            },
        )

    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        return self.sample_sparse(info, rng).to_dense_task()


@dataclass
class RandomCellSampler:
    """
    Randomly query a fraction of episode cells.
    All non-query cells in the episode are observed.
    """
    n_episode_rows: int
    query_frac: float = 0.15
    min_query_cells: int = 1
    max_query_cells: Optional[int] = None
    replace_rows: bool = False
    task_name: str = "random_cell"

    def sample_sparse(self, info: TableInfo, rng: np.random.Generator) -> SparseCompletionTask:
        if not (0.0 < self.query_frac < 1.0):
            raise ValueError("query_frac must be in (0, 1).")

        row_idx = sample_unique_rows_fast(
            rng=rng,
            n_rows=info.n_rows,
            n_episode_rows=self.n_episode_rows,
            replace=self.replace_rows,
        )
        col_idx = all_columns(info)

        n_ep, d_ep = self.n_episode_rows, info.n_cols
        total_cells = n_ep * d_ep

        k = max(int(round(self.query_frac * total_cells)), self.min_query_cells)
        if self.max_query_cells is not None:
            k = min(k, self.max_query_cells)

        # Leave at least one observed cell.
        k = min(k, total_cells - 1)

        flat = rng.choice(total_cells, size=k, replace=False)
        query_coords = flat_indices_to_coords(flat, d_ep)

        return SparseCompletionTask(
            row_idx=row_idx,
            col_idx=col_idx,
            query_coords=query_coords,
            observed_mode="all_except_query",
            task_name=self.task_name,
            meta={
                "query_frac": self.query_frac,
                "num_query_cells_requested": k,
                "conditioning_mode": "transductive",
                "sparse": True,
            },
        )

    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        return self.sample_sparse(info, rng).to_dense_task()


@dataclass
class ColumnBlockSampler:
    """
    Context rows are fully observed.
    Query rows have sampled columns queried.
    """
    n_context: int
    n_query: int
    min_query_cols: int = 1
    max_query_cols: int = 3
    exclude_target: bool = False
    replace_rows: bool = False
    conditioning_mode: str = "inductive_rows"
    task_name: str = "column_block"

    def sample_sparse(self, info: TableInfo, rng: np.random.Generator) -> SparseCompletionTask:
        n_ep = self.n_context + self.n_query
        row_idx = sample_unique_rows_fast(rng, info.n_rows, n_ep, self.replace_rows)
        col_idx = all_columns(info)

        eligible_cols = list(range(info.n_cols))
        if self.exclude_target and info.target_col is not None:
            eligible_cols = [c for c in eligible_cols if c != info.target_col]

        if not eligible_cols:
            raise ValueError("No eligible columns to query.")

        if self.min_query_cols <= 0:
            raise ValueError("min_query_cols must be positive.")

        if self.max_query_cols < self.min_query_cols:
            raise ValueError("max_query_cols must be >= min_query_cols.")

        k = int(rng.integers(self.min_query_cols, self.max_query_cols + 1))
        k = min(k, len(eligible_cols))

        query_cols_global = rng.choice(eligible_cols, size=k, replace=False)
        query_cols_local = np.array(
            [global_col_to_local(col_idx, c) for c in query_cols_global],
            dtype=np.int64,
        )

        query_rows = np.arange(self.n_context, n_ep, dtype=np.int64)
        query_coords = make_grid_coords(query_rows, query_cols_local)

        return SparseCompletionTask(
            row_idx=row_idx,
            col_idx=col_idx,
            query_coords=query_coords,
            observed_mode="all_except_query",
            task_name=self.task_name,
            meta={
                "query_cols_global": query_cols_global,
                "n_context": self.n_context,
                "n_query": self.n_query,
                "conditioning_mode": self.conditioning_mode,
                "context_rows_local": np.arange(self.n_context, dtype=np.int64),
                "query_rows_local": np.arange(self.n_context, n_ep, dtype=np.int64),
                "sparse": True,
            },
        )

    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        return self.sample_sparse(info, rng).to_dense_task()


@dataclass
class RowBlockSampler:
    """
    Context rows are fully observed.
    Query rows have a fraction of columns queried.
    """
    n_context: int
    n_query: int
    query_frac_cols: float = 1.0
    replace_rows: bool = False
    conditioning_mode: str = "inductive_rows"
    task_name: str = "row_block"

    def sample_sparse(self, info: TableInfo, rng: np.random.Generator) -> SparseCompletionTask:
        if not (0.0 < self.query_frac_cols <= 1.0):
            raise ValueError("query_frac_cols must be in (0, 1].")

        n_ep = self.n_context + self.n_query
        row_idx = sample_unique_rows_fast(rng, info.n_rows, n_ep, self.replace_rows)
        col_idx = all_columns(info)

        query_rows = np.arange(self.n_context, n_ep, dtype=np.int64)

        k_cols = max(1, int(round(self.query_frac_cols * info.n_cols)))
        k_cols = min(k_cols, info.n_cols)

        query_cols_local = rng.choice(info.n_cols, size=k_cols, replace=False)
        query_coords = make_grid_coords(query_rows, query_cols_local)

        return SparseCompletionTask(
            row_idx=row_idx,
            col_idx=col_idx,
            query_coords=query_coords,
            observed_mode="all_except_query",
            task_name=self.task_name,
            meta={
                "query_frac_cols": self.query_frac_cols,
                "query_cols_local": query_cols_local,
                "conditioning_mode": self.conditioning_mode,
                "context_rows_local": np.arange(self.n_context, dtype=np.int64),
                "query_rows_local": np.arange(self.n_context, n_ep, dtype=np.int64),
                "sparse": True,
            },
        )

    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        return self.sample_sparse(info, rng).to_dense_task()


@dataclass
class LabelFeatureSampler:
    """
    Query target column plus sampled feature columns for query rows.
    """
    n_context: int
    n_query: int
    n_feature_cols: int = 2
    target_col: Optional[int] = None
    replace_rows: bool = False
    conditioning_mode: str = "inductive_rows"
    task_name: str = "label_plus_feature"

    def sample_sparse(self, info: TableInfo, rng: np.random.Generator) -> SparseCompletionTask:
        target_col = self.target_col if self.target_col is not None else info.target_col
        if target_col is None:
            raise ValueError("LabelFeatureSampler requires target_col.")

        n_ep = self.n_context + self.n_query
        row_idx = sample_unique_rows_fast(
            rng=rng,
            n_rows=info.n_rows,
            n_episode_rows=n_ep,
            replace=self.replace_rows,
        )
        col_idx = all_columns(info)

        feature_cols = [c for c in range(info.n_cols) if c != target_col]
        if not feature_cols:
            raise ValueError("No feature columns available.")

        k = min(self.n_feature_cols, len(feature_cols))
        sampled_features = rng.choice(feature_cols, size=k, replace=False)

        query_cols_global = np.concatenate(
            [
                np.array([target_col], dtype=np.int64),
                sampled_features.astype(np.int64),
            ]
        )

        query_cols_local = np.array(
            [global_col_to_local(col_idx, c) for c in query_cols_global],
            dtype=np.int64,
        )

        query_rows = np.arange(self.n_context, n_ep, dtype=np.int64)
        query_coords = make_grid_coords(query_rows, query_cols_local)

        return SparseCompletionTask(
            row_idx=row_idx,
            col_idx=col_idx,
            query_coords=query_coords,
            observed_mode="all_except_query",
            task_name=self.task_name,
            meta={
                "target_col": target_col,
                "sampled_feature_cols": sampled_features,
                "query_cols_global": query_cols_global,
                "conditioning_mode": self.conditioning_mode,
                "context_rows_local": np.arange(self.n_context, dtype=np.int64),
                "query_rows_local": np.arange(self.n_context, n_ep, dtype=np.int64),
                "sparse": True,
            },
        )

    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        return self.sample_sparse(info, rng).to_dense_task()


@dataclass
class MixtureSampler:
    samplers: Sequence[TaskSampler]
    weights: Sequence[float]
    task_name: str = "mixture"

    def __post_init__(self) -> None:
        if len(self.samplers) != len(self.weights):
            raise ValueError("samplers and weights must have same length.")

        weights = np.asarray(self.weights, dtype=np.float64)

        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative.")

        if weights.sum() <= 0:
            raise ValueError("weights must sum to > 0.")

        self._probs = weights / weights.sum()

    def sample_sparse(self, info: TableInfo, rng: np.random.Generator) -> SparseCompletionTask:
        idx = int(rng.choice(len(self.samplers), p=self._probs))
        sampler = self.samplers[idx]

        if not hasattr(sampler, "sample_sparse"):
            # Fallback: dense then convert to sparse query coords.
            dense = sampler.sample(info, rng)
            sparse = SparseCompletionTask(
                row_idx=dense.row_idx,
                col_idx=dense.col_idx,
                query_coords=dense.query_coords_local(),
                observed_mode="explicit",
                observed_coords=dense.observed_coords_local(),
                task_name=dense.task_name,
                meta=dict(dense.meta),
            )
        else:
            sparse = sampler.sample_sparse(info, rng)  # type: ignore[attr-defined]

        sparse.meta = dict(sparse.meta)
        sparse.meta["mixture_component"] = sparse.task_name
        sparse.task_name = f"{self.task_name}:{sparse.task_name}"
        return sparse

    def sample(self, info: TableInfo, rng: np.random.Generator) -> CompletionTask:
        return self.sample_sparse(info, rng).to_dense_task()