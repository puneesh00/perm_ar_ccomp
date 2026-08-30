# scripts/eval_synthetic_official_tabpfn.py
"""
Evaluates the OFFICIAL pip-installed TabPFN checkpoints (v1: `tabpfn==0.1.11`,
v2: `tabpfn==2.2.1`, both from PyPI) against the exact same k=1 (label
prediction) synthetic-prior episodes as eval_synthetic_incontext.py --
reuses that module's build_table_generator/build_task so the per-episode
(row, column) draw is byte-for-byte identical to what our own retrained
checkpoints were scored on, given the same --eval-seed and generative-prior
flags (see that script's docstring: numpy's PCG64 seeding is deterministic
and platform-independent).

Meant to be invoked twice, once from each dedicated venv
(.venv_tabpfn_v1, .venv_tabpfn_v2) via --version v1|v2 -- see
eval_openml_official_tabpfn.py's docstring for why the two official
packages can't share one environment.

Only k=1 is supported (true label prediction, target_col forced) -- not
the general k-cols multi-target sweep eval_synthetic_incontext.py offers.
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
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

from run_openml_baselines import JSONLLogger, write_csv_from_jsonl  # noqa: E402
from eval_synthetic_incontext import build_table_generator, build_task  # noqa: E402
from tab_completion.model import NUMERICAL  # noqa: E402


def build_model(version: str, device: str, seed: int, n_estimators: int = 1, no_tricks: bool = False):
    """no_tricks=True: a single forward pass with every inference-time
    augmentation this package offers turned off (no feature/class shift,
    no preprocessing-transform ensemble, no fingerprint feature, no outlier
    clipping, no probability balancing, no temperature sharpening) -- as
    close to "just the pretrained transformer, once" as either package's
    public API allows. Verified against the actually-installed versions
    (tabpfn==0.1.11 for v1, tabpfn==2.2.1 for v2) -- API surfaces drift
    across tabpfn releases, so don't copy this into a different pinned
    version without re-checking.

    Still-inherent input processing that survives no_tricks (this is the
    checkpoint's own expected input contract, not an ensemble trick, so it
    is deliberately left on for both versions): context-fitted
    z-normalization, constant-feature removal, outlier-based rescaling by
    number of used features (v1); basic categorical/label encoding and the
    model's own internal per-feature normalization/grouping (v2).

    n_estimators is ignored when no_tricks=True (forced to a single pass).
    """
    if version == "v1":
        from tabpfn import TabPFNClassifier

        if no_tricks:
            return TabPFNClassifier(
                device=device,
                seed=seed,
                N_ensemble_configurations=1,
                no_preprocess_mode=True,       # no power/quantile/robust transform ensemble
                feature_shift_decoder=False,   # no feature rotation
                multiclass_decoder="none",     # no class-id rotation
            )
        return TabPFNClassifier(device=device, seed=seed, N_ensemble_configurations=n_estimators)

    elif version == "v2":
        from tabpfn import TabPFNClassifier
        from tabpfn.preprocessing import PreprocessorConfig

        if no_tricks:
            return TabPFNClassifier(
                device=device,
                random_state=seed,
                n_estimators=1,
                softmax_temperature=1.0,       # no default 0.9 sharpening
                balance_probabilities=False,
                inference_config={
                    "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none")],
                    "FEATURE_SHIFT_METHOD": None,
                    "CLASS_SHIFT_METHOD": None,
                    "FINGERPRINT_FEATURE": False,
                    "OUTLIER_REMOVAL_STD": None,
                    "POLYNOMIAL_FEATURES": "no",
                    "SUBSAMPLE_SAMPLES": None,
                },
            )
        return TabPFNClassifier(device=device, random_state=seed, n_estimators=n_estimators)

    raise ValueError(f"Unknown version {version!r}")


def flatten_xy(full, task):
    """Mirrors train_tabpfn_v1_baseline.build_xy's column-flattening, adapted
    to build_task's query_cols-style CompletionTask (query_cols[0] is the
    forced target column when k=1). No densification/remap here (unlike
    build_xy) -- official TabPFNClassifier handles arbitrary raw class ids
    itself via its own LabelEncoder."""
    target_col = int(task.meta["query_cols"][0])
    feature_cols = [c for c in task.col_idx.tolist() if c != target_col]
    rows = task.row_idx
    n_context = task.meta["n_context"]

    parts = []
    for j in feature_cols:
        if full.col_types[j] == NUMERICAL:
            parts.append(full.x_num[rows, j].astype(np.float64))
        else:
            parts.append(full.x_cat[rows, j].astype(np.float64))
    X = np.stack(parts, axis=1)
    y_all = full.x_cat[rows, target_col].astype(np.int64)

    return X[:n_context], y_all[:n_context], X[n_context:], y_all[n_context:]


def padded_proba(y_proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    if len(classes_) == n_classes and np.array_equal(classes_, np.arange(n_classes)):
        return y_proba
    out = np.zeros((y_proba.shape[0], n_classes), dtype=y_proba.dtype)
    out[:, classes_.astype(np.int64)] = y_proba
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2"])
    parser.add_argument("--eval-tasks", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--n-context", type=int, default=512)
    parser.add_argument("--n-query", type=int, default=64)

    parser.add_argument("--fresh-n-rows", type=int, default=576)
    parser.add_argument("--n-cols", type=int, default=64)
    parser.add_argument("--p-categorical", type=float, default=0.3)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--tabpfn-prior-type", type=str, default="scm")
    parser.add_argument("--tabpfn-layers-mu-max", type=float, default=6.0)
    parser.add_argument("--tabpfn-layers-max", type=int, default=None)
    parser.add_argument("--tabpfn-hidden-mu-max", type=float, default=130.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=1,
                         help="ensemble size / N_ensemble_configurations for the official checkpoint. "
                              "Default 1 (single forward pass, matching our own checkpoints -- no "
                              "ensembling). v1's paper default is 32, v2's package default is 8. "
                              "Ignored when --no-tricks is set.")
    parser.add_argument("--no-tricks", action="store_true",
                         help="Single forward pass with every inference-time augmentation disabled "
                              "(no feature/class shift, no preprocessing-transform ensemble, no "
                              "fingerprint feature, no outlier clipping, no probability balancing, "
                              "no temperature sharpening) -- see build_model's docstring for exactly "
                              "what stays on (the checkpoint's own required input contract) vs off.")
    parser.add_argument("--out-dir", type=str, default="results/synthetic_incontext")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"synthetic_tabpfn_{args.version}_official_{int(time.time())}"
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"
    logger = JSONLLogger(jsonl_path)

    model_name = f"tabpfn_{args.version}_official"
    print(f"=== Official TabPFN-{args.version} synthetic-prior eval (k=1, label prediction) ===")
    print(f"eval_seed={args.eval_seed} eval_tasks={args.eval_tasks} "
          f"n_context={args.n_context} n_query={args.n_query} device={args.device}")

    table_gen = build_table_generator(args)
    row_rng = np.random.default_rng(args.eval_seed)
    col_rng = np.random.default_rng([1, args.eval_seed])  # k=1, matches run_k's per-k reset

    correct = 0
    total = 0
    nll_sum = 0.0
    n_episodes = 0

    for ep in range(args.eval_tasks):
        full = table_gen.sample_table()
        task = build_task(
            full, 1, args.n_context, args.n_query, row_rng, col_rng,
            categorical_only=False, force_target=True,
        )
        X_train, y_train, X_test, y_test = flatten_xy(full, task)

        try:
            clf = build_model(args.version, args.device, args.seed, args.n_estimators, args.no_tricks)
            clf.fit(X_train, y_train)
            proba_raw = clf.predict_proba(X_test)
            classes_ = np.asarray(clf.classes_)
            n_classes = int(full.cat_cardinalities[int(task.meta["query_cols"][0])])
            proba = padded_proba(proba_raw, classes_, n_classes)
            y_pred = proba.argmax(axis=1)

            ep_correct = int((y_pred == y_test).sum())
            p_true = np.clip(proba[np.arange(len(y_test)), y_test], 1e-12, 1.0)
            ep_nll = float(-np.log(p_true).sum())

            correct += ep_correct
            total += len(y_test)
            nll_sum += ep_nll
            n_episodes += 1

            logger.log({
                "k_cols": 1, "model": model_name, "ar_mode": "n/a", "episode": ep,
                "n_context": args.n_context, "n_query": args.n_query, "status": "ok",
                "marginal_cat_acc": ep_correct / len(y_test),
                "marginal_cat_cells": len(y_test),
                "avg_nll": ep_nll / len(y_test),
            })
        except Exception as e:
            print(f"  [error] ep={ep}: {e}")
            logger.log({
                "k_cols": 1, "model": model_name, "episode": ep, "status": "error",
                "error": repr(e), "traceback": traceback.format_exc(),
            })

        if (ep + 1) % 20 == 0:
            print(f"  [{ep + 1}/{args.eval_tasks}] running marginal_cat_acc={correct / max(total, 1):.4f}")

    overall_acc = correct / max(total, 1)
    overall_nll = nll_sum / max(total, 1)
    print(
        f"\n=== FINAL: {model_name} k=1 n_episodes={n_episodes} "
        f"marginal_cat_acc={overall_acc:.4f} avg_nll={overall_nll:.4f} ==="
    )

    logger.log({
        "k_cols": 1, "model": model_name, "ar_mode": "n/a", "episode": "AGGREGATE",
        "n_episodes": n_episodes, "n_context": args.n_context, "n_query": args.n_query,
        "status": "ok", "marginal_cat_acc": overall_acc, "marginal_cat_cells": total,
        "avg_nll": overall_nll,
    })
    write_csv_from_jsonl(jsonl_path, csv_path)
    print(f"\nDone.\nJSONL: {jsonl_path}\nCSV:   {csv_path}")


if __name__ == "__main__":
    main()
