# src/tab_completion/synthetic_data.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tab_completion.sampling import TableInfo
from tab_completion.model import NUMERICAL, CATEGORICAL


@dataclass
class FullSyntheticTable:
    """
    Full synthetic table stored in memory.

    x_num:
        [n_rows, n_cols], normalized numerical values.
        Dummy 0 for categorical columns.

    x_cat:
        [n_rows, n_cols], local categorical ids.
        Dummy 0 for numerical columns.

    col_types:
        [n_cols], 0 numerical / 1 categorical.

    cat_cardinalities:
        [n_cols], number of valid categories for categorical columns.
        For numerical columns, use 1.

    cat_decode_types:
        [n_cols], categorical decoder type per column.
        v0: per-column decode type, cat_decode_types[j] = j.
    """
    x_num: np.ndarray
    x_cat: np.ndarray
    col_types: np.ndarray
    cat_cardinalities: np.ndarray
    cat_decode_types: np.ndarray
    target_col: int

    @property
    def n_rows(self) -> int:
        return self.x_num.shape[0]

    @property
    def n_cols(self) -> int:
        return self.x_num.shape[1]

    def table_info(self) -> TableInfo:
        return TableInfo(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            target_col=self.target_col,
            col_types=self.col_types,
            name="synthetic_scm",
        )


def standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Convert continuous vector to categorical ids 0..n_bins-1 by quantiles.
    """
    if n_bins <= 1:
        return np.zeros_like(x, dtype=np.int64)

    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs)
    y = np.digitize(x, edges, right=False)
    return np.clip(y, 0, n_bins - 1).astype(np.int64)


def make_synthetic_table(
    n_rows: int,
    n_cols: int,
    p_categorical: float,
    k_max: int,
    n_classes: int,
    seed: int,
    target_col: Optional[int] = None,
    latent_dim: int = 8,
    noise: float = 0.1,
) -> FullSyntheticTable:
    """
    Synthetic SCM-ish table.

    v0 assumptions:
      - target column is categorical;
      - numerical values are standardized;
      - categorical values are local ids;
      - category decoder type is per-column.
    """
    if n_cols < 4:
        raise ValueError("n_cols should be at least 4.")

    rng = np.random.default_rng(seed)

    if target_col is None:
        target_col = n_cols - 1

    z = rng.normal(size=(n_rows, latent_dim)).astype(np.float32)

    W = rng.normal(size=(latent_dim, n_cols)).astype(np.float32)
    raw = z @ W + noise * rng.normal(size=(n_rows, n_cols)).astype(np.float32)

    col_types = np.full(n_cols, NUMERICAL, dtype=np.int64)

    feature_cols = [j for j in range(n_cols) if j != target_col]
    n_cat_features = int(round(p_categorical * len(feature_cols)))
    n_cat_features = min(max(n_cat_features, 0), len(feature_cols))

    if n_cat_features > 0:
        cat_feature_cols = set(
            rng.choice(feature_cols, size=n_cat_features, replace=False).tolist()
        )
    else:
        cat_feature_cols = set()

    # Target is categorical in v0.
    col_types[target_col] = CATEGORICAL
    for j in cat_feature_cols:
        col_types[j] = CATEGORICAL

    x_num = np.zeros((n_rows, n_cols), dtype=np.float32)
    x_cat = np.zeros((n_rows, n_cols), dtype=np.int64)
    cat_cardinalities = np.ones(n_cols, dtype=np.int64)

    # Numerical columns.
    num_cols = [j for j in range(n_cols) if col_types[j] == NUMERICAL]
    if num_cols:
        x_num[:, num_cols] = standardize(raw[:, num_cols]).astype(np.float32)

    # Categorical feature columns.
    for j in cat_feature_cols:
        k_j = int(rng.integers(2, k_max + 1))
        x_cat[:, j] = quantile_bins(raw[:, j], k_j)
        cat_cardinalities[j] = k_j

    # Target depends on several features and latent factors.
    target_signal = np.zeros(n_rows, dtype=np.float32)

    useful_cols = feature_cols[: min(6, len(feature_cols))]
    for idx, j in enumerate(useful_cols):
        coef = 1.0 / (idx + 1)
        if col_types[j] == NUMERICAL:
            target_signal += coef * x_num[:, j]
        else:
            denom = max(cat_cardinalities[j] - 1, 1)
            target_signal += coef * (x_cat[:, j].astype(np.float32) / denom)

    target_signal += 0.5 * z[:, 0]
    target_signal += noise * rng.normal(size=n_rows).astype(np.float32)

    if n_classes == 2:
        probs = 1.0 / (1.0 + np.exp(-target_signal))
        y = rng.binomial(1, probs).astype(np.int64)
    else:
        y = quantile_bins(target_signal, n_classes)

    x_cat[:, target_col] = y
    cat_cardinalities[target_col] = n_classes
    x_num[:, target_col] = 0.0

    # v0: per-column categorical decoder type.
    cat_decode_types = np.arange(n_cols, dtype=np.int64)

    return FullSyntheticTable(
        x_num=x_num,
        x_cat=x_cat,
        col_types=col_types,
        cat_cardinalities=cat_cardinalities,
        cat_decode_types=cat_decode_types,
        target_col=target_col,
    )