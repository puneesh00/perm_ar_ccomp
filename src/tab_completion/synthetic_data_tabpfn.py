# src/tab_completion/synthetic_data_tabpfn.py
"""
TabPFN-style synthetic prior for conditional-completion experiments.

This is still an adaptation to this codebase's FullSyntheticTable interface
and table scale; it is not a byte-for-byte copy of the original TabPFN data
loader (Hollmann et al., ICLR 2023, arXiv:2207.01848, Appendix C.1/C.2,
Section 4.5). It mixes two branches, matching the paper's Section 4.4
(SCM and BNN priors, mixed with equal probability):

  SCM branch: sample a DAG shaped like an MLP (layer count, per-layer width,
  Gaussian edge weights, random edge dropout to sparsify into a DAG, one
  shared activation for the whole graph, per-node noise), propagate forward
  layer by layer (z_i = activation(sum_j W_ij z_j + eps_i)), then pick
  feature/target nodes from anywhere in the graph.

  BNN branch: sample i.i.d. Gaussian inputs, feed them through a fresh
  random MLP (same architecture-sampling machinery as the SCM branch) to
  produce the target.

Several more table-level behaviors are themselves random per-table switches
rather than hard-coded on/off, matching Table 5's "Choices" section:
blockwise-vs-uniform node selection, whether to keep the sampled block
order for feature columns, and whether to apply a post-hoc affine rescale
to features. Scalar outputs (the target, and a p_categorical fraction of
feature columns) become discrete classes by sampling bin boundaries FROM
THE REALIZED VALUES themselves (not fixed quantiles) and then shuffling
the resulting class ids, per Section 4.5 / C.2.5 -- this is also why some
generated tables land on imbalanced classes; that's intended, not a bug.

Known deviations from the paper, and why:
  - `p_edge_dropout` and `p_root_importance` are NOT literal Table 5
    switches. The paper's edge dropout rate is already continuous
    (0.9 * Beta(a, b), which can itself land near zero), and C.2.3
    describes root-feature-importance amplification as a standing part of
    the generation recipe, not something toggled off some fraction of the
    time. These two gates are deliberate extensions for extra prior
    diversity, kept as a conscious choice -- not something we're claiming
    is paper-faithful.
  - "Sample y from last MLP layer" (Table 5) is not implemented: target
    selection here has no relationship to graph depth, so the target is as
    likely to land on a shallow/root node as a deep one. Left as a known
    gap rather than guessed at.

Expect a real, non-trivial accuracy ceiling from this prior, and don't
assume a stuck-looking training curve means another bug: unlike the
simplified prior used earlier (which was deliberately fixed to be
recoverable), an SCM- or BNN-generated target can depend on deep, partially
latent structure by construction -- that's the whole point of moving to
this richer prior, not a regression to chase.

Output is a FullSyntheticTable, identical to what synthetic_data.py
produces, so every sampler, factorizer, and model in this codebase can
consume it with zero changes -- this is a drop-in alternative generator,
not a new data format. Currently classification-only: feature columns can
be numerical or categorical, but the target column is always categorical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import truncnorm

from tab_completion.model import NUMERICAL, CATEGORICAL
from tab_completion.synthetic_data import FullSyntheticTable


# ---------------------------------------------------------------------
# TNLU: truncated-normal log-uniform sampling (Table 5): sample mean mu and
# std sigma from a shared LogUniform(mu_min, mu_max) range, then draw v from
# TruncNormal(mu, sigma^2, a=0, b=inf); round if requested; return v + min_val.
# ---------------------------------------------------------------------


def sample_tnlu(
    rng: np.random.Generator,
    mu_min: float,
    mu_max: float,
    min_val: float,
    round_to_int: bool,
) -> float:
    floor = 1e-8
    log_mu = rng.uniform(np.log(max(mu_min, floor)), np.log(mu_max))
    mu = np.exp(log_mu)
    log_sigma = rng.uniform(np.log(max(mu_min, floor)), np.log(mu_max))
    sigma = np.exp(log_sigma)

    a = (0.0 - mu) / sigma
    v = float(truncnorm.rvs(a, np.inf, loc=mu, scale=sigma, random_state=rng))

    if round_to_int:
        v = round(v)
    return v + min_val


# Paper-exact (Table 5: "MLP Activation Functions"): Tanh, LeakyReLU, ELU,
# Identity. (A "threshold" step-function option was tried and dropped: with
# one activation shared across every layer of a graph -- correct per the
# paper -- a threshold-activated graph collapses into a chain of binary
# indicators, and empirically produced ~13% near-constant numeric columns
# across a 3k-table stress test. Not worth the "richness".)
ACTIVATIONS = ("tanh", "leaky_relu", "elu", "identity")


def apply_activation(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        return np.tanh(x)
    if kind == "leaky_relu":
        return np.where(x > 0, x, 0.01 * x)
    if kind == "elu":
        return np.where(x > 0, x, np.exp(np.clip(x, -30.0, 30.0)) - 1.0)
    if kind == "identity":
        return x
    raise ValueError(f"Unknown activation {kind!r}")


# ---------------------------------------------------------------------
# Config. Hyperparameters are ranges/probabilities sampled per table.
# ---------------------------------------------------------------------


@dataclass
class TabPFNSCMConfig:
    n_rows: int = 256
    n_cols: int = 16
    target_col: Optional[int] = None
    p_categorical: float = 0.2
    k_max: int = 16
    n_classes: int = 2
    base_seed: int = 0

    # Which prior branch to use.
    #   "scm"   : SCM branch only
    #   "bnn"   : BNN branch only
    #   "mixed" : sample BNN with probability p_bnn, else SCM (paper: 50/50)
    prior_type: str = "mixed"
    p_bnn: float = 0.5

    # Table-level switches (Table 5's "Choices"), each a per-table probability.
    p_blockwise_feature_sampling: float = 0.5
    p_keep_feature_order: float = 0.5
    p_feature_scaling: float = 0.5

    # NOT literal Table 5 switches -- see module docstring "Known deviations".
    # Kept as deliberate, documented extensions for extra prior diversity.
    p_root_importance: float = 0.75
    p_edge_dropout: float = 0.75

    # SCM structure (Table 5)
    layers_mu_min: float = 1.0
    layers_mu_max: float = 6.0
    layers_min: int = 2
    # Hard ceiling on sampled depth. layers_mu_max alone only lowers the TNLU
    # *mean* -- the underlying truncated normal is still unbounded above, so
    # tail draws can exceed it (observed: layers_mu_max=3.0 still produced a
    # depth-12 graph). Set this when you actually need a firm cap.
    layers_max: Optional[int] = None
    hidden_mu_min: float = 5.0
    hidden_mu_max: float = 130.0
    hidden_min: int = 4
    layer0_mu_min: float = 1.0
    layer0_mu_max: float = 12.0
    layer0_min: int = 1
    noise_mu_min: float = 1e-4
    noise_mu_max: float = 0.3
    weight_mu_min: float = 0.01
    weight_mu_max: float = 10.0

    # edge dropout rate, when enabled: 0.9 * Beta(a, b), a, b ~ Uniform(0.1, 5.0)
    dropout_ab_min: float = 0.1
    dropout_ab_max: float = 5.0

    def __post_init__(self) -> None:
        if self.prior_type not in {"scm", "bnn", "mixed"}:
            raise ValueError("prior_type must be one of {'scm', 'bnn', 'mixed'}")
        if self.n_cols < 4:
            raise ValueError("n_cols should be at least 4.")
        if not (0.0 <= self.p_bnn <= 1.0):
            raise ValueError("p_bnn must be in [0, 1].")


@dataclass
class SCMGraph:
    layer_sizes: List[int]
    weight_matrices: List[np.ndarray]  # weight_matrices[i]: layer i -> layer i+1
    activation: str
    node_noise_means: List[np.ndarray]
    node_noise_stds: List[np.ndarray]
    used_edge_dropout: bool
    used_root_importance: bool

    @property
    def total_nodes(self) -> int:
        return sum(self.layer_sizes)


# ---------------------------------------------------------------------
# SCM branch.
# ---------------------------------------------------------------------


def sample_scm_graph(
    rng: np.random.Generator,
    cfg: TabPFNSCMConfig,
    min_total_nodes: int,
) -> SCMGraph:
    n_layers = int(sample_tnlu(rng, cfg.layers_mu_min, cfg.layers_mu_max, cfg.layers_min, True))
    if cfg.layers_max is not None:
        n_layers = min(n_layers, cfg.layers_max)
    hidden = int(sample_tnlu(rng, cfg.hidden_mu_min, cfg.hidden_mu_max, cfg.hidden_min, True))
    layer0 = int(sample_tnlu(rng, cfg.layer0_mu_min, cfg.layer0_mu_max, cfg.layer0_min, True))

    layer_sizes = [layer0] + [hidden] * max(n_layers - 1, 1)

    # Guarantee enough nodes to select n_cols features+target from. Widening
    # the last layer preserves the sampled depth/shape rather than resampling.
    shortfall = min_total_nodes - sum(layer_sizes)
    if shortfall > 0:
        layer_sizes[-1] += shortfall

    noise_std = sample_tnlu(rng, cfg.noise_mu_min, cfg.noise_mu_max, 0.0, False)
    weight_std = sample_tnlu(rng, cfg.weight_mu_min, cfg.weight_mu_max, 0.0, False)
    activation = str(rng.choice(ACTIVATIONS))

    used_edge_dropout = rng.random() < cfg.p_edge_dropout
    if used_edge_dropout:
        dropout_a = rng.uniform(cfg.dropout_ab_min, cfg.dropout_ab_max)
        dropout_b = rng.uniform(cfg.dropout_ab_min, cfg.dropout_ab_max)
        edge_dropout_rate = 0.9 * rng.beta(dropout_a, dropout_b)
        keep_prob = 1.0 - edge_dropout_rate
    else:
        keep_prob = 1.0

    # Fan-in normalized weight scale: without this, deep/wide graphs (widths
    # up to 130, depths up to ~20 are both in-range) can blow up in float32
    # well before reaching the output -- confirmed empirically (a raw
    # RuntimeWarning: overflow encountered in reduce during generation).
    # The BNN branch below already does this; apply the same fix here so
    # both branches are on equal numerical footing.
    weight_matrices: List[np.ndarray] = []
    for i in range(len(layer_sizes) - 1):
        fan_in = max(layer_sizes[i], 1)
        W = rng.normal(0.0, weight_std / np.sqrt(fan_in), size=(layer_sizes[i], layer_sizes[i + 1])).astype(np.float32)
        keep_mask = rng.random(W.shape) < keep_prob
        weight_matrices.append(W * keep_mask)

    # C.2.3: amplify layer-0 (root) feature importances so effects don't
    # regress to the mean as hidden width grows -- some inputs end up
    # mattering far more than others, like real features do.
    used_root_importance = bool(weight_matrices) and (rng.random() < cfg.p_root_importance)
    if used_root_importance:
        importance = np.exp(rng.normal(0.0, 1.0, size=layer_sizes[0])).astype(np.float32)
        weight_matrices[0] = weight_matrices[0] * importance[:, None]

    share_noise_mean = rng.random() < 0.5
    node_noise_means: List[np.ndarray] = []
    node_noise_stds: List[np.ndarray] = []
    for size in layer_sizes:
        if share_noise_mean:
            means = np.full(size, rng.normal(0.0, 1.0), dtype=np.float32)
        else:
            means = rng.normal(0.0, 1.0, size=size).astype(np.float32)
        stds = (noise_std * np.exp(rng.normal(0.0, 0.3, size=size))).astype(np.float32)
        node_noise_means.append(means)
        node_noise_stds.append(np.abs(stds))

    return SCMGraph(
        layer_sizes=layer_sizes,
        weight_matrices=weight_matrices,
        activation=activation,
        node_noise_means=node_noise_means,
        node_noise_stds=node_noise_stds,
        used_edge_dropout=used_edge_dropout,
        used_root_importance=used_root_importance,
    )


def propagate_scm(rng: np.random.Generator, graph: SCMGraph, n_rows: int) -> List[np.ndarray]:
    """Returns one [n_rows, layer_size] array per layer."""
    layer_values: List[np.ndarray] = []

    eps0 = rng.normal(graph.node_noise_means[0], graph.node_noise_stds[0], size=(n_rows, graph.layer_sizes[0]))
    layer_values.append(apply_activation(eps0, graph.activation).astype(np.float32))

    for i in range(1, len(graph.layer_sizes)):
        eps = rng.normal(graph.node_noise_means[i], graph.node_noise_stds[i], size=(n_rows, graph.layer_sizes[i]))
        pre = layer_values[-1] @ graph.weight_matrices[i - 1] + eps
        layer_values.append(apply_activation(pre, graph.activation).astype(np.float32))

    return layer_values


# ---------------------------------------------------------------------
# BNN branch.
# ---------------------------------------------------------------------


def sample_bnn_target_raw(
    rng: np.random.Generator,
    cfg: TabPFNSCMConfig,
    X: np.ndarray,
) -> np.ndarray:
    """
    BNN-like target generator (Section 4.4): observed features are sampled
    first (i.i.d., in make_bnn_table), then a random neural network maps
    them to a scalar target before binning into classes.
    """
    n_features = X.shape[1]
    n_layers = int(sample_tnlu(rng, cfg.layers_mu_min, cfg.layers_mu_max, cfg.layers_min, True))
    if cfg.layers_max is not None:
        n_layers = min(n_layers, cfg.layers_max)
    hidden = int(sample_tnlu(rng, cfg.hidden_mu_min, cfg.hidden_mu_max, cfg.hidden_min, True))
    weight_std = sample_tnlu(rng, cfg.weight_mu_min, cfg.weight_mu_max, 0.0, False)
    noise_std = sample_tnlu(rng, cfg.noise_mu_min, cfg.noise_mu_max, 0.0, False)
    activation = str(rng.choice(ACTIVATIONS))

    h = X.astype(np.float32)
    in_dim = n_features
    n_hidden_layers = max(n_layers - 1, 1)
    for _ in range(n_hidden_layers):
        W = rng.normal(0.0, weight_std / np.sqrt(max(in_dim, 1)), size=(in_dim, hidden)).astype(np.float32)
        b = rng.normal(0.0, noise_std, size=hidden).astype(np.float32)
        h = apply_activation(h @ W + b, activation).astype(np.float32)
        in_dim = hidden

    W_out = rng.normal(0.0, weight_std / np.sqrt(max(in_dim, 1)), size=(in_dim, 1)).astype(np.float32)
    b_out = rng.normal(0.0, noise_std, size=1).astype(np.float32)
    y_raw = (h @ W_out).squeeze(-1) + b_out[0]
    y_raw = y_raw + noise_std * rng.normal(size=len(y_raw)).astype(np.float32)
    return y_raw.astype(np.float32)


# ---------------------------------------------------------------------
# Node selection (C.2.2, "blockwise feature sampling") and value binning
# (Section 4.5 / C.2.5: bounds sampled from realized values, then labels
# shuffled so there's no artificial ordinality).
# ---------------------------------------------------------------------


def select_blockwise_nodes(rng: np.random.Generator, layer_sizes: List[int], k: int) -> List[int]:
    """
    Pick k global node indices (flattened across all layers) in contiguous
    per-layer runs rather than uniformly at random, so picks from the same
    run tend to be adjacent nodes in the same layer -- adjacent nodes share
    more of their causal ancestry, which is what makes them correlate.
    """
    offsets = np.cumsum([0] + layer_sizes)
    total = int(offsets[-1])
    chosen: List[int] = []
    chosen_set = set()
    attempts = 0
    while len(chosen) < k and attempts < 200:
        attempts += 1
        layer_idx = int(rng.integers(0, len(layer_sizes)))
        size = layer_sizes[layer_idx]
        if size == 0:
            continue
        block_len = min(size, int(rng.integers(1, max(2, size // 2 + 1))), k - len(chosen))
        start = int(rng.integers(0, size - block_len + 1))
        for j in range(block_len):
            g = int(offsets[layer_idx]) + start + j
            if g not in chosen_set and len(chosen) < k:
                chosen.append(g)
                chosen_set.add(g)

    if len(chosen) < k:
        remaining = [g for g in range(total) if g not in chosen_set]
        rng.shuffle(remaining)
        chosen += remaining[: k - len(chosen)]

    return chosen[:k]


def select_uniform_nodes(rng: np.random.Generator, layer_sizes: List[int], k: int) -> List[int]:
    total = int(sum(layer_sizes))
    if k > total:
        raise ValueError(f"Cannot select {k} nodes from graph with {total} nodes.")
    return rng.choice(total, size=k, replace=False).astype(np.int64).tolist()


def select_feature_target_nodes(
    rng: np.random.Generator,
    cfg: TabPFNSCMConfig,
    layer_sizes: List[int],
    k: int,
) -> Tuple[List[int], int, dict]:
    """
    Return feature node indices, target node index, and metadata.

    Known gap (Table 5's "Sample y from last MLP layer" is not implemented):
    the target is just whichever node lands last in the assembled index
    list, with no relationship to graph depth -- it's as likely to be a
    shallow/root node as a deep one.
    """
    use_blockwise = rng.random() < cfg.p_blockwise_feature_sampling
    keep_order = rng.random() < cfg.p_keep_feature_order

    if use_blockwise:
        node_indices = select_blockwise_nodes(rng, layer_sizes, k)
    else:
        node_indices = select_uniform_nodes(rng, layer_sizes, k)

    if not keep_order:
        rng.shuffle(node_indices)

    target_node = node_indices[-1]
    feature_nodes = node_indices[:-1]

    meta = {
        "use_blockwise_feature_sampling": use_blockwise,
        "keep_feature_order": keep_order,
    }
    return feature_nodes, target_node, meta


def global_index_to_layer_pos(idx: int, layer_sizes: List[int]) -> Tuple[int, int]:
    offset = 0
    for li, size in enumerate(layer_sizes):
        if offset <= idx < offset + size:
            return li, idx - offset
        offset += size
    raise IndexError(idx)


def bin_by_realized_values(rng: np.random.Generator, values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Sample n_bins-1 boundaries FROM THE DATA ITSELF (paper: "sample Nc-1
    class bounds randomly from the set of continuous targets"), map each
    value to its interval, then shuffle the resulting bin ids -- the raw
    interval index is monotonic in value, but classes shouldn't carry an
    implied order, so the shuffle removes that.
    """
    n = len(values)
    if n_bins <= 1:
        return np.zeros(n, dtype=np.int64)

    n_bounds = min(n_bins - 1, n - 1)
    bound_vals = np.sort(rng.choice(values, size=n_bounds, replace=False))
    ids = np.searchsorted(bound_vals, values, side="right")

    perm = rng.permutation(n_bins)
    return perm[ids].astype(np.int64)


def maybe_scale_features(rng: np.random.Generator, cfg: TabPFNSCMConfig, X: np.ndarray) -> np.ndarray:
    if rng.random() >= cfg.p_feature_scaling:
        return X
    scales = np.exp(rng.normal(0.0, 1.0, size=X.shape[1])).astype(np.float32)
    shifts = rng.normal(0.0, 1.0, size=X.shape[1]).astype(np.float32)
    return X * scales[None, :] + shifts[None, :]


# ---------------------------------------------------------------------
# Table assembly: same FullSyntheticTable shape as synthetic_data.py.
# Shared by both branches.
# ---------------------------------------------------------------------


def assemble_full_table_from_raw(
    rng: np.random.Generator,
    cfg: TabPFNSCMConfig,
    feature_raw: np.ndarray,
    target_raw: np.ndarray,
) -> FullSyntheticTable:
    n_rows, n_features = feature_raw.shape
    n_cols = cfg.n_cols
    target_col = cfg.target_col if cfg.target_col is not None else n_cols - 1
    feature_cols = [c for c in range(n_cols) if c != target_col]
    if len(feature_cols) != n_features:
        raise ValueError("feature_raw must have n_cols - 1 columns.")

    x_num = np.zeros((n_rows, n_cols), dtype=np.float32)
    x_cat = np.zeros((n_rows, n_cols), dtype=np.int64)
    col_types = np.full(n_cols, NUMERICAL, dtype=np.int64)
    cat_cardinalities = np.ones(n_cols, dtype=np.int64)

    n_cat_features = int(round(cfg.p_categorical * len(feature_cols)))
    n_cat_features = min(max(n_cat_features, 0), len(feature_cols))
    cat_feature_cols = set(
        rng.choice(feature_cols, size=n_cat_features, replace=False).tolist()
    ) if n_cat_features > 0 else set()

    for source_idx, col in enumerate(feature_cols):
        raw = feature_raw[:, source_idx].astype(np.float32)
        if col in cat_feature_cols:
            k_j = int(rng.integers(2, cfg.k_max + 1))
            x_cat[:, col] = bin_by_realized_values(rng, raw, k_j)
            cat_cardinalities[col] = k_j
            col_types[col] = CATEGORICAL
        else:
            mean, std = float(raw.mean()), float(raw.std())
            x_num[:, col] = (raw - mean) / (std + 1e-6)

    y = bin_by_realized_values(rng, target_raw.astype(np.float32), cfg.n_classes)
    x_cat[:, target_col] = y
    cat_cardinalities[target_col] = cfg.n_classes
    col_types[target_col] = CATEGORICAL
    x_num[:, target_col] = 0.0

    cat_decode_types = np.arange(n_cols, dtype=np.int64)

    return FullSyntheticTable(
        x_num=x_num,
        x_cat=x_cat,
        col_types=col_types,
        cat_cardinalities=cat_cardinalities,
        cat_decode_types=cat_decode_types,
        target_col=target_col,
    )


def make_scm_table(cfg: TabPFNSCMConfig, seed: int) -> FullSyntheticTable:
    rng = np.random.default_rng(seed)
    graph = sample_scm_graph(rng, cfg, min_total_nodes=cfg.n_cols)
    layer_values = propagate_scm(rng, graph, cfg.n_rows)

    feature_nodes, target_node, _ = select_feature_target_nodes(
        rng, cfg, graph.layer_sizes, cfg.n_cols
    )

    def node_values(idx: int) -> np.ndarray:
        li, pos = global_index_to_layer_pos(idx, graph.layer_sizes)
        return layer_values[li][:, pos]

    feature_raw = np.stack([node_values(idx) for idx in feature_nodes], axis=1).astype(np.float32)
    feature_raw = maybe_scale_features(rng, cfg, feature_raw).astype(np.float32)
    target_raw = node_values(target_node).astype(np.float32)

    return assemble_full_table_from_raw(rng, cfg, feature_raw, target_raw)


def make_bnn_table(cfg: TabPFNSCMConfig, seed: int) -> FullSyntheticTable:
    rng = np.random.default_rng(seed)
    n_features = cfg.n_cols - 1

    X = rng.normal(0.0, 1.0, size=(cfg.n_rows, n_features)).astype(np.float32)
    X = maybe_scale_features(rng, cfg, X).astype(np.float32)
    target_raw = sample_bnn_target_raw(rng, cfg, X)

    return assemble_full_table_from_raw(rng, cfg, X, target_raw)


def make_tabpfn_style_table(cfg: TabPFNSCMConfig, seed: int) -> FullSyntheticTable:
    rng = np.random.default_rng(seed)
    if cfg.prior_type == "scm":
        use_bnn = False
    elif cfg.prior_type == "bnn":
        use_bnn = True
    else:
        use_bnn = bool(rng.random() < cfg.p_bnn)

    # Branch-specific seed so the branch choice doesn't consume the same
    # random stream that defines the table contents.
    branch_seed = int(rng.integers(0, 2**31 - 1))
    if use_bnn:
        return make_bnn_table(cfg, branch_seed)
    return make_scm_table(cfg, branch_seed)


class TabPFNSCMTableGenerator:
    """Drop-in replacement for SyntheticTableGenerator (synthetic_data.py)."""

    def __init__(self, cfg: TabPFNSCMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.base_seed)

    def sample_table(self) -> FullSyntheticTable:
        seed = int(self.rng.integers(0, 2**31 - 1))
        return make_tabpfn_style_table(self.cfg, seed)
