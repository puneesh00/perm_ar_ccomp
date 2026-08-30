# scripts/official_v1_prior_gen.py
"""
Generates tables from the REAL, released TabPFN-v1 prior-generation code
(tabpfn.priors.mlp + .flexible_categorical, shipped inside the pip
tabpfn==0.1.11 package -- not a reimplementation), WITH categorical feature
metadata captured (which columns got converted, and their cardinality) --
official's own FlexibleCategorical.forward computes this per-column but
discards it (see flexible_categorical.py:164-171: `num_unique_features` is
local to the loop body, never returned). This module instruments forward()
with a one-line capture, matching an earlier review's own suggested "tiny
instrumentation change ... does not change the generated distribution; it
only exposes metadata."

IMPORTANT PERFORMANCE NOTE: call torch.set_num_threads(1) (or a small
number) BEFORE importing/using this module, in whatever process does
generation. mlp.py's new_mlp_per_example=True path does one nn.Module
construction PER EXAMPLE in a serial Python loop -- with torch's default
thread count (all available cores), each of those tiny per-example tensor
ops pays huge thread-pool synchronization overhead on a many-core machine,
making generation ~300-500x slower than it needs to be (measured: ~3-5s/
table at default threading vs ~0.01s/table with num_threads(1), on a
255-core machine, same table size). This is NOT a property of official's
code being slow -- it's a well-known PyTorch anti-pattern for many small
sequential ops on high-core-count hardware. With num_threads(1), live
per-step generation is fast enough to use directly in a training loop,
same as our own synthetic generator -- no offline pre-caching needed.

Provides OfficialV1LiveTableGenerator: a drop-in replacement for
TabPFNSCMTableGenerator (synthetic_data_tabpfn.py) -- same
`.cfg.n_cols` (mutable, read fresh each call) / `.sample_table()
-> FullSyntheticTable` interface, so it works with train_synthetic.py's /
train_tabpfn_v1_baseline.py's existing resample_variable_table_shape and
TargetPredictionSampler machinery completely unchanged.

Representation note: official's own "categorical" feature columns are NOT
integer category ids routed through a separate embedding path -- by the
time FlexibleCategorical.forward returns, EVERY column (categorical or not)
has been through remove_outliers/normalize_data (z-scored), so a
"categorical" column is just a z-normalized float with a small number of
discrete underlying values, structurally no different from a continuous
column at consumption time. That's also literally how official's actual
model encoder treats it (confirmed earlier this session: only one shared
linear projection for all features, no separate categorical path wired in).
So every FEATURE column here is stored as NUMERICAL (x_num) -- this exactly
matches what official's own architecture (and train_tabpfn_v1_baseline.py's
build_xy, which just concatenates x_num/x_cat by column into one float
tensor regardless of col_types) actually does with it. Only the TARGET
column is CATEGORICAL, since that's the actual classification label
train_tabpfn_v1_baseline.py's build_xy re-densifies per its own context
split.

Requires `tabpfn` (the v1 pip package, tabpfn==0.1.11) importable on this
interpreter's path.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tab_completion.model import NUMERICAL, CATEGORICAL  # noqa: E402
from tab_completion.synthetic_data import FullSyntheticTable  # noqa: E402


def bypass_import_priors():
    """tabpfn/priors/__init__.py eagerly imports fast_gp -> gpytorch (often
    not installed). Register a stub package module so mlp.py/
    flexible_categorical.py's relative imports resolve without executing the
    real __init__.py. Locates the installed tabpfn package dynamically
    (importlib.util.find_spec) rather than hardcoding a venv path."""
    tabpfn_spec = importlib.util.find_spec("tabpfn")
    if tabpfn_spec is None or not tabpfn_spec.submodule_search_locations:
        raise ImportError(
            "Could not locate an installed 'tabpfn' package. Install "
            "tabpfn==0.1.11 in whatever environment runs this script."
        )
    tabpfn_root = Path(list(tabpfn_spec.submodule_search_locations)[0])
    priors_dir = tabpfn_root / "priors"

    pkg_name = "tabpfn.priors"
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(priors_dir / "__init__.py"),
        submodule_search_locations=[str(priors_dir)],
    )
    stub = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = stub
    mlp = importlib.import_module("tabpfn.priors.mlp")
    flexible_categorical = importlib.import_module("tabpfn.priors.flexible_categorical")
    return mlp, flexible_categorical


_PATCHED = False
_CAPTURED = []  # list of dicts, one per FlexibleCategorical instance's forward() call, in construction order


def _install_instrumented_forward(flexible_categorical):
    """Monkeypatch FlexibleCategorical.forward with an exact copy of the
    original (flexible_categorical.py:143-243) plus one addition: records
    {col_idx: final_cardinality} for every feature column actually converted
    to categorical, into the module-level _CAPTURED list. Idempotent --
    calling this more than once is a no-op after the first call."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    FC = flexible_categorical.FlexibleCategorical
    random = flexible_categorical.random
    torch_mod = torch
    MulticlassRank = flexible_categorical.MulticlassRank
    normalize_data = flexible_categorical.normalize_data
    nan_handling_missing_for_unknown_reason_value = flexible_categorical.nan_handling_missing_for_unknown_reason_value
    nan_handling_missing_for_no_reason_value = flexible_categorical.nan_handling_missing_for_no_reason_value
    nan_handling_missing_for_a_reason_value = flexible_categorical.nan_handling_missing_for_a_reason_value
    to_ranking_low_mem = flexible_categorical.to_ranking_low_mem
    remove_outliers = flexible_categorical.remove_outliers
    normalize_by_used_features_f = flexible_categorical.normalize_by_used_features_f

    def instrumented_forward(self, batch_size):
        x, y, y_ = self.get_batch(hyperparameters=self.h, **self.args_passed)

        if self.h['nan_prob_no_reason'] + self.h['nan_prob_a_reason'] + self.h['nan_prob_unknown_reason'] > 0 and random.random() > 0.5:
            if random.random() < self.h['nan_prob_no_reason']:
                x = self.drop_for_no_reason(x, nan_handling_missing_for_no_reason_value(self.h['set_value_to_nan']))
            if self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5:
                x = self.drop_for_reason(x, nan_handling_missing_for_a_reason_value(self.h['set_value_to_nan']))
            if self.h['nan_prob_unknown_reason'] > 0:
                if random.random() < self.h['nan_prob_unknown_reason_reason_prior']:
                    x = self.drop_for_no_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))
                else:
                    x = self.drop_for_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))

        # Categorical features -- instrumented: capture (col -> final cardinality)
        # for every column actually converted, before it's discarded.
        cat_info: Dict[int, int] = {}
        if 'categorical_feature_p' in self.h and random.random() < self.h['categorical_feature_p']:
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1, 10)), 2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    x[:, :, col] = m(x[:, :, col])
                    cat_info[col] = int(m.num_classes)  # m.num_classes is the REAL post-class_sampler_f cardinality
        _CAPTURED.append(cat_info)

        if self.h['normalize_to_ranking']:
            x = to_ranking_low_mem(x)
        else:
            x = remove_outliers(x)
        x, y = normalize_data(x), normalize_data(y)

        y = self.class_assigner(y).float()

        if self.h['normalize_by_used_features']:
            x = normalize_by_used_features_f(
                x, self.h['num_features_used'], self.args['num_features'],
                normalize_with_sqrt=self.h.get('normalize_with_sqrt', False),
            )

        x = torch_mod.cat(
            [x, torch_mod.zeros((x.shape[0], x.shape[1], self.args['num_features'] - self.h['num_features_used']),
                                 device=self.args['device'])], -1)

        if self.h['check_is_compatible']:
            for b in range(y.shape[1]):
                is_compatible, N = False, 0
                while not is_compatible and N < 10:
                    targets_in_train = torch_mod.unique(y[:self.args['single_eval_pos'], b], sorted=True)
                    targets_in_eval = torch_mod.unique(y[self.args['single_eval_pos']:, b], sorted=True)
                    is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                        targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1
                    if not is_compatible:
                        randperm = torch_mod.randperm(x.shape[0])
                        x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                    N = N + 1
                if not is_compatible:
                    y[:, b] = -100

        if self.h['normalize_labels']:
            for b in range(y.shape[1]):
                valid_labels = y[:, b] != -100
                if self.h.get('normalize_ignore_label_too', False):
                    valid_labels[:] = True
                y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()
                if y[valid_labels, b].numel() != 0 and self.h.get('rotate_normalized_labels', True):
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch_mod.randint(0, num_classes, (1,), device=self.args['device'])
                    y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes

        return x, y, y

    FC.forward = instrumented_forward


def generate_official_v1_table(
    mlp, flexible_categorical, rng: np.random.Generator,
    num_features: int, n_context: int, n_query: int,
    categorical_feature_p_choices=(0.0, 0.1, 0.2),
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    """Returns (x_np [seq_len, num_features] float32, y_np [seq_len] float,
    num_classes int, feature_cat_info {col: cardinality} for columns
    actually converted to categorical this table -- empty dict if none)."""
    _install_instrumented_forward(flexible_categorical)

    num_classes = int(rng.integers(2, 11))
    num_layers = int(rng.integers(2, 9))
    prior_mlp_hidden_dim = int(rng.integers(64, 200))
    num_causes = int(rng.integers(1, min(10, num_features) + 1))
    noise_std = float(rng.uniform(0.05, 0.3))
    dropout_prob = float(rng.uniform(0.0, 0.3))
    categorical_feature_p = float(rng.choice(categorical_feature_p_choices))
    seq_len = n_context + n_query

    hp = dict(
        num_classes=num_classes, balanced=False, multiclass_type="rank", output_multiclass_ordered_p=0.5,
        nan_prob_no_reason=0.0, nan_prob_a_reason=0.0, nan_prob_unknown_reason=0.0,
        nan_prob_unknown_reason_reason_prior=0.5, set_value_to_nan=0.5,
        categorical_feature_p=categorical_feature_p,
        normalize_to_ranking=False, normalize_by_used_features=True, num_features_used=num_features,
        check_is_compatible=True, normalize_labels=True, normalize_ignore_label_too=False,
        rotate_normalized_labels=True, seq_len_used=seq_len,
        num_layers=num_layers, is_causal=True, num_causes=num_causes, prior_mlp_hidden_dim=prior_mlp_hidden_dim,
        pre_sample_causes=True, noise_std=noise_std, pre_sample_weights=True,
        prior_mlp_activations=lambda: (lambda: torch.nn.Tanh()),
        block_wise_dropout=False, prior_mlp_dropout_prob=dropout_prob, prior_mlp_scale_weights_sqrt=True,
        init_std=1.0, new_mlp_per_example=True, mix_activations=False, sampling="normal",
        y_is_effect=True, in_clique=False, sort_features=False, random_feature_rotation=True, verbose=False,
    )

    _CAPTURED.clear()
    x, y, _ = flexible_categorical.get_batch(
        batch_size=1, seq_len=seq_len, num_features=num_features,
        get_batch=mlp.get_batch, device="cpu", hyperparameters=hp, single_eval_pos=n_context,
    )
    cat_info = _CAPTURED[-1] if _CAPTURED else {}

    x_np = x[:, 0, :].detach().numpy().astype(np.float32)
    y_np = y[:, 0].detach().numpy()
    return x_np, y_np, num_classes, cat_info


class _OfficialV1Cfg:
    """Minimal mutable config object -- resample_variable_table_shape only
    ever does `table_generator.cfg.n_cols = ...`, matching
    TabPFNSCMConfig's n_cols field."""
    def __init__(self, n_rows: int, n_cols: int, base_seed: int, n_query_for_check: int = 64):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.base_seed = base_seed
        self.n_query_for_check = n_query_for_check


class OfficialV1LiveTableGenerator:
    """Drop-in replacement for TabPFNSCMTableGenerator
    (synthetic_data_tabpfn.py), backed by the real released TabPFN-v1 prior
    code instead of our own reimplementation. Same interface: `.cfg.n_cols`
    mutable (read fresh by sample_table() every call, so
    resample_variable_table_shape's `table_generator.cfg.n_cols = ...`
    works unchanged), `.sample_table() -> FullSyntheticTable`.

    Requires torch.set_num_threads(1) to already be set in this process --
    see this module's docstring for why (without it, generation is ~300-500x
    slower on a high-core-count machine)."""

    def __init__(self, n_rows: int, n_cols: int, base_seed: int, n_query_for_check: int = 64):
        self.cfg = _OfficialV1Cfg(n_rows, n_cols, base_seed, n_query_for_check)
        self.rng = np.random.default_rng(base_seed)
        self.mlp, self.flexible_categorical = bypass_import_priors()

    def sample_table(self) -> FullSyntheticTable:
        num_features = self.cfg.n_cols - 1
        n_rows = self.cfg.n_rows
        n_context = max(n_rows - self.cfg.n_query_for_check, 2)
        n_query = n_rows - n_context

        x_np, y_np, num_classes, _cat_info = generate_official_v1_table(
            self.mlp, self.flexible_categorical, self.rng,
            num_features=num_features, n_context=n_context, n_query=n_query,
        )

        n_cols = num_features + 1
        target_col = num_features
        x_num = np.zeros((n_rows, n_cols), dtype=np.float32)
        x_num[:, :num_features] = x_np
        x_cat = np.zeros((n_rows, n_cols), dtype=np.int64)
        x_cat[:, target_col] = y_np.astype(np.int64)
        col_types = np.full(n_cols, NUMERICAL, dtype=np.int64)
        col_types[target_col] = CATEGORICAL
        cat_cardinalities = np.ones(n_cols, dtype=np.int64)
        cat_cardinalities[target_col] = num_classes
        cat_decode_types = np.arange(n_cols, dtype=np.int64)
        return FullSyntheticTable(
            x_num=x_num, x_cat=x_cat, col_types=col_types,
            cat_cardinalities=cat_cardinalities, cat_decode_types=cat_decode_types,
            target_col=target_col,
        )
