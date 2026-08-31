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
    flexible_categorical.py's/differentiable_prior.py's relative imports
    resolve without executing the real __init__.py. Locates the installed
    tabpfn package dynamically (importlib.util.find_spec) rather than
    hardcoding a venv path."""
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
    differentiable_prior = importlib.import_module("tabpfn.priors.differentiable_prior")
    return mlp, flexible_categorical, differentiable_prior


# Extracted directly from the ACTUAL released checkpoint's saved config
# (tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt --
# torch.load(path)[2]['differentiable_hyperparameters']), not from the
# current tabpfn==0.1.11 pip package's model_configs.py. The two disagree
# for several entries (num_layers/prior_mlp_hidden_dim/num_causes use Gamma
# in the currently-shipped model_configs.py's get_diff_causal(), but TNLU
# -- matching the TabPFN-v1 PAPER's Table 5 exactly -- in the checkpoint
# that actually trained the weights we're comparing against; similarly
# prior_mlp_dropout_prob's Beta scale is 0.6 in the shipped code vs 0.9 in
# the checkpoint, and prior_mlp_activations has 3 choices in the shipped
# code vs 4 in the checkpoint). The checkpoint is the more authoritative
# source for OUR purpose (explaining/matching that specific checkpoint's
# performance) -- verified with torch.load's own config_sample dict, not
# inferred. See this repo's own record of that investigation.
#
# `is_causal` IS present in the real checkpoint's differentiable_hyperparameters
# (choice_values=[True, False], meaning it really did mix in noncausal/
# "BNN"-like tables) -- deliberately OMITTED here and pinned to True in
# generate_official_v1_table's static hyperparameters instead, per explicit
# decision to exclude that branch. This is therefore a known, deliberate
# simplification relative to the real checkpoint, not an accidental match.
#
# GP: prior_bag_exp_weights_1 in the real checkpoint's config is
# Uniform(1000000, 1000001) -- softmax([1.0, ~1e6]) selects the MLP branch
# (index 1 in model_builder.py's `(get_batch_gp, get_batch_mlp)` tuple) with
# ~100% probability every time; GP (index 0) was never meaningfully invoked
# in the real checkpoint's actual training despite prior_type='prior_bag'
# nominally including it. So omitting GP here isn't just a pragmatic
# shortcut -- it's confirmed to match what the real checkpoint's training
# actually did.
#
# `prior_mlp_activations`'s 4th real choice is a lambda pickled as an
# unrecoverable string repr (function objects don't survive torch.load as
# live callables) -- from context (Tanh/Identity/ELU are the other three,
# and the paper's Table 5 lists {Tanh, LeakyReLU, ELU, Identity}) it's
# almost certainly a LeakyReLU wrapper. Using torch.nn.LeakyReLU directly
# (default negative_slope=0.01) as the closest available stand-in -- the
# one item here that's still an approximation, not a checkpoint-verified
# exact value.
DIFF_CAUSAL_CONFIG = {
    "num_layers": {"distribution": "meta_trunc_norm_log_scaled", "max_mean": 6, "min_mean": 1, "round": True, "lower_bound": 2},
    "prior_mlp_hidden_dim": {"distribution": "meta_trunc_norm_log_scaled", "max_mean": 130, "min_mean": 5, "round": True, "lower_bound": 4},
    "prior_mlp_dropout_prob": {"distribution": "meta_beta", "scale": 0.9, "min": 0.1, "max": 5.0},
    "noise_std": {"distribution": "meta_trunc_norm_log_scaled", "max_mean": 0.3, "min_mean": 0.0001, "round": False, "lower_bound": 0.0},
    "init_std": {"distribution": "meta_trunc_norm_log_scaled", "max_mean": 10.0, "min_mean": 0.01, "round": False, "lower_bound": 0.0},
    "num_causes": {"distribution": "meta_trunc_norm_log_scaled", "max_mean": 12, "min_mean": 1, "round": True, "lower_bound": 1},
    "pre_sample_weights": {"distribution": "meta_choice", "choice_values": [True, False]},
    "pre_sample_causes": {"distribution": "meta_choice", "choice_values": [True, False]},
    "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sampling": {"distribution": "meta_choice", "choice_values": ["normal", "mixed"]},
    "prior_mlp_activations": {"distribution": "meta_choice_mixed", "choice_values": [
        torch.nn.Tanh, torch.nn.Identity, torch.nn.ELU, torch.nn.LeakyReLU,
    ]},
    "block_wise_dropout": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sort_features": {"distribution": "meta_choice", "choice_values": [True, False]},
    "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
    # --- from get_diff_flex() ---
    # output_multiclass_ordered_p is NOT actually in the checkpoint's real
    # differentiable_hyperparameters dict (verified by mechanically diffing
    # every key against the checkpoint -- this repo's earlier assumption
    # that it was differentiable, because the general model_configs.py
    # lists it that way, was wrong for THIS checkpoint). It's a fixed
    # static 0.0 there instead -- set as a static hyperparameter below, not
    # sampled here.
    "multiclass_type": {"distribution": "meta_choice", "choice_values": ["value", "rank"]},
}


def build_differentiable_hparams(differentiable_prior):
    """Constructs the DifferentiableHyperparameterList ONCE (reused across
    every table generated in this process) -- official's own real sampling
    code (gamma/truncated-normal/beta draws, softmax-weighted meta_choice),
    not a reimplementation. See DIFF_CAUSAL_CONFIG's docstring for exactly
    what's in it and what's deliberately excluded. embedding_dim is a dead
    parameter in this code path (only used by commented-out embedding-layer
    code) -- value doesn't matter, kept at 1."""
    return differentiable_prior.DifferentiableHyperparameterList(
        DIFF_CAUSAL_CONFIG, embedding_dim=1, device="cpu",
    )


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
    mlp, flexible_categorical, diff_hparams, rng: np.random.Generator,
    num_features: int, n_context: int, n_query: int,
    categorical_feature_p: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    """Returns (x_np [seq_len, num_features] float32, y_np [seq_len] float,
    num_classes int, feature_cat_info {col: cardinality} for columns
    actually converted to categorical this table -- empty dict if none).

    diff_hparams: a DifferentiableHyperparameterList (see
    build_differentiable_hparams) -- sample_parameter_object() is called
    once per table, exactly mirroring official's own DifferentiablePrior.
    forward() (`hyperparameters = {**self.h, **sampled_hyperparameters_passed}`).
    Values coming out for the meta_gamma/meta_trunc_norm_log_scaled/
    meta_beta entries are one-call thunks by construction -- do NOT resolve
    them here; FlexibleCategorical.__init__'s own existing
    `hyperparameters[k]() if callable(...)` line does that, exactly as it
    does for official's real training.

    Static values below (num_classes, categorical_feature_p=0.2, is_causal,
    the nan_prob_*/set_value_to_nan/normalize_ignore_label_too/mix_activations
    block) are cross-checked against the ACTUAL released checkpoint's saved
    config (torch.load(...)[2]) wherever recoverable -- categorical_feature_p
    IS a fixed 0.2 there (not a {0,0.1,0.2} table-level draw, which was this
    repo's earlier, wrong assumption), normalize_ignore_label_too=True,
    set_value_to_nan=0.1, mix_activations=True (activation choice varies per
    LAYER within a table, not fixed per table -- DIFF_CAUSAL_CONFIG's
    prior_mlp_activations sampling already produces the right nested-callable
    shape for this; mix_activations=True here just stops mlp.py's own code
    from collapsing it to one fixed activation). num_classes remains our own
    approximation (rng.integers(2,11)) -- the checkpoint's real sampler is a
    lambda pickled as an unrecoverable string repr, confirmed genuinely
    dynamic (not hardcoded) but not recoverable exactly. is_causal stays
    pinned True -- the checkpoint's config confirms it really was
    differentiable/mixed with noncausal tables, so this is a known,
    deliberate exclusion (the "no BNN" decision), not an accidental match."""
    _install_instrumented_forward(flexible_categorical)

    num_classes = int(rng.integers(2, 11))
    seq_len = n_context + n_query

    sampled_hp, _ = diff_hparams.sample_parameter_object()

    hp = dict(
        num_classes=num_classes, balanced=False,
        nan_prob_no_reason=0.0, nan_prob_a_reason=0.0, nan_prob_unknown_reason=0.0,
        nan_prob_unknown_reason_reason_prior=1.0, set_value_to_nan=0.1,
        categorical_feature_p=categorical_feature_p,
        normalize_to_ranking=False, normalize_by_used_features=True, num_features_used=num_features,
        normalize_ignore_label_too=True,
        rotate_normalized_labels=True, seq_len_used=seq_len,
        is_causal=True,
        prior_mlp_scale_weights_sqrt=True, normalize_with_sqrt=False,
        new_mlp_per_example=True, mix_activations=True,
        random_feature_rotation=True, verbose=False,
        output_multiclass_ordered_p=0.0,
        # check_is_compatible / normalize_labels: absent from the checkpoint's
        # own saved config entirely (confirmed by the same mechanical diff --
        # this pip package's flexible_categorical.py requires both via plain
        # `self.h[...]` access, so there's no way to run this code at all
        # without supplying them; they're newer than this 2022 checkpoint,
        # so there's no historical value to match against). True is the only
        # sensible choice -- False would either skip a real safety check
        # (context/query share the same label set) or skip the label
        # densification/rotation step that's clearly active per
        # rotate_normalized_labels/normalize_ignore_label_too being real,
        # checkpoint-confirmed settings.
        check_is_compatible=True, normalize_labels=True,
    )
    hp.update(sampled_hp)

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
        self.mlp, self.flexible_categorical, differentiable_prior = bypass_import_priors()
        self.diff_hparams = build_differentiable_hparams(differentiable_prior)

    def sample_table(self) -> FullSyntheticTable:
        num_features = self.cfg.n_cols - 1
        n_rows = self.cfg.n_rows
        n_context = max(n_rows - self.cfg.n_query_for_check, 2)
        n_query = n_rows - n_context

        x_np, y_np, num_classes, _cat_info = generate_official_v1_table(
            self.mlp, self.flexible_categorical, self.diff_hparams, self.rng,
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
