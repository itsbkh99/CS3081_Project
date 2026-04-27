"""
agent.py
--------
Feature Selection Agent — CS3081: Artificial Intelligence
Effat University · Spring 2026

Team: Faisal Yahya , Faisal Shamsi, Omar Almutairi, Abdulaziz Bukhari 
Instructor: Dr. Naila Marir

The agent combines:
  - Informed search (A* and Forward Selection)
  - Probabilistic reasoning (Naive Bayes with 5-fold CV)
  - Adversarial robustness (noise + irrelevant feature injection)

Usage:
  python agent.py                             # Quick demo on breast cancer
  python agent.py --dataset all              # Full experiment suite
  python agent.py --dataset breast_cancer --search astar --noise 0.3 --extra 0.5
  python agent.py --dataset wine --search forward
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from environment import load_dataset, create_adversarial_dataset
from evaluator import NaiveBayesEvaluator
from heuristics import compute_mi_scores
from search import forward_selection, astar_search, SearchResult


# ─────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║      FEATURE SELECTION AI AGENT  ·  CS3081 Artificial Intelligence
║      Effat University  ·  Spring 2026  ·  Dr. Naila Marir
║      Team: Faisal Yahya · Faisal Shamsi · Omar Almutairi
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────
# BASELINE: All features, no selection
# ─────────────────────────────────────────────────────────────────────

def baseline_all_features(X_train, X_test, y_train, y_test):
    """Train Naive Bayes on all features as the baseline comparator."""
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds, average='macro')


# ─────────────────────────────────────────────────────────────────────
# RUN SINGLE EXPERIMENT
# ─────────────────────────────────────────────────────────────────────

def run_experiment(dataset_name: str,
                   search_algo: str = "astar",
                   noise_sigma: float = 0.0,
                   extra_ratio: float = 0.0,
                   label_flip: float = 0.0,
                   cv_folds: int = 5,
                   seed: int = 42,
                   verbose: bool = True) -> dict:
    """
    Run one complete feature selection experiment.

    Args:
        dataset_name: one of breast_cancer, wine, ionosphere, spambase, madelon
        search_algo:  'forward' or 'astar'
        noise_sigma:  Gaussian noise standard deviation (0 = no noise)
        extra_ratio:  fraction of extra random features to inject (0 = none)
        label_flip:   fraction of labels to randomly corrupt (0 = none)
        cv_folds:     number of cross-validation folds for Naive Bayes oracle
        seed:         random seed for reproducibility
        verbose:      whether to print detailed output

    Returns:
        dict of result metrics
    """
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Dataset    : {dataset_name}")
        print(f"  Algorithm  : {search_algo.upper()}")
        print(f"  Noise σ    : {noise_sigma}")
        print(f"  Extra feats: {int(extra_ratio*100)}%")
        print(f"  Label flip : {int(label_flip*100)}%")
        print(f"{'─'*60}")

    # ── Load & perturb dataset ──
    if noise_sigma > 0 or extra_ratio > 0 or label_flip > 0:
        X, y, n_original, feat_names = create_adversarial_dataset(
            dataset_name, noise_sigma, extra_ratio, label_flip, seed
        )
    else:
        X_raw, y, feat_names = load_dataset(dataset_name)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        n_original = X.shape[1]

    n_total_features = X.shape[1]
    n_samples = X.shape[0]

    if verbose:
        print(f"  Samples    : {n_samples}")
        print(f"  Features   : {n_original} original + "
              f"{n_total_features - n_original} injected = {n_total_features} total")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # ── Baseline: all features ──
    t0 = time.time()
    base_acc, base_f1 = baseline_all_features(X_train, X_test, y_train, y_test)
    if verbose:
        print(f"\n  [Baseline] All features → Accuracy: {base_acc:.4f}  F1: {base_f1:.4f}")

    # ── Compute MI scores for heuristic ──
    mi_scores = compute_mi_scores(X_train, y_train, seed=seed)

    # ── Build oracle evaluator ──
    evaluator = NaiveBayesEvaluator(X_train, y_train, cv_folds=cv_folds, seed=seed)

    # ── Run selected search algorithm ──
    if verbose:
        print(f"\n  Running {search_algo.upper()} search...")

    t_search_start = time.time()

    if search_algo == "forward":
        result = forward_selection(evaluator, n_total_features)
    elif search_algo == "astar":
        result = astar_search(evaluator, n_total_features, mi_scores)
    else:
        raise ValueError(f"Unknown search algorithm: '{search_algo}'. "
                         f"Choose from: forward, astar")

    search_time = time.time() - t_search_start

    # ── Evaluate final selected subset on held-out test set ──
    if result.selected_features:
        mask = np.zeros(n_total_features, dtype=bool)
        mask[result.selected_features] = True

        X_train_sel = X_train[:, mask]
        X_test_sel  = X_test[:, mask]

        clf_final = GaussianNB()
        clf_final.fit(X_train_sel, y_train)
        preds = clf_final.predict(X_test_sel)
        final_acc = accuracy_score(y_test, preds)
        final_f1  = f1_score(y_test, preds, average='macro')

        # Count how many selected features were original vs injected noise
        orig_selected  = [f for f in result.selected_features if f < n_original]
        noise_selected = [f for f in result.selected_features if f >= n_original]
    else:
        final_acc = base_acc
        final_f1  = base_f1
        orig_selected  = []
        noise_selected = []

    total_time = time.time() - t0

    if verbose:
        print(f"\n  ✅ Search complete!")
        print(f"  Selected {len(result.selected_features)} / {n_total_features} features")
        print(f"     ├─ From original features : {len(orig_selected)}")
        print(f"     └─ From injected noise    : {len(noise_selected)} (ideally 0)")
        print(f"  CV Accuracy  (search oracle) : {result.accuracy:.4f}")
        print(f"  Test Accuracy (held-out set) : {final_acc:.4f}  "
              f"(Δ vs baseline: {final_acc - base_acc:+.4f})")
        print(f"  F1 Score                     : {final_f1:.4f}")
        print(f"  Subsets evaluated            : {result.n_subsets_evaluated}")
        print(f"  Dimensionality reduction     : "
              f"{100*(1 - len(result.selected_features)/n_total_features):.1f}%")
        print(f"  Search time                  : {search_time:.2f}s")
        print(f"  Total time                   : {total_time:.2f}s")

        if result.selected_features and n_total_features <= 60:
            sel_names = [feat_names[i] for i in sorted(result.selected_features)
                         if i < len(feat_names)]
            print(f"\n  Selected features:")
            for name in sel_names[:12]:
                print(f"     • {name}")
            if len(sel_names) > 12:
                print(f"     ... and {len(sel_names)-12} more")

    return {
        "dataset":                    dataset_name,
        "algorithm":                  search_algo,
        "noise_sigma":                noise_sigma,
        "extra_ratio":                extra_ratio,
        "label_flip":                 label_flip,
        "n_samples":                  n_samples,
        "n_features_original":        n_original,
        "n_features_total":           n_total_features,
        "n_features_selected":        len(result.selected_features),
        "n_orig_selected":            len(orig_selected),
        "n_noise_selected":           len(noise_selected),
        "baseline_accuracy":          round(base_acc, 4),
        "agent_accuracy":             round(final_acc, 4),
        "accuracy_delta":             round(final_acc - base_acc, 4),
        "agent_f1":                   round(final_f1, 4),
        "subsets_evaluated":          result.n_subsets_evaluated,
        "dimensionality_reduction_pct": round(
            100 * (1 - len(result.selected_features) / n_total_features), 1
        ),
        "search_time_s":              round(search_time, 2),
        "total_time_s":               round(total_time, 2),
    }


# ─────────────────────────────────────────────────────────────────────
# FULL EXPERIMENT SUITE
# ─────────────────────────────────────────────────────────────────────

DATASETS     = ["breast_cancer", "wine", "ionosphere"]
ALGORITHMS   = ["forward", "astar"]
NOISE_LEVELS = [0.0, 0.3]
EXTRA_RATIOS = [0.0, 0.5]


def run_full_suite(output_dir: str = "results"):
    """
    Run all experiment combinations and save a summary CSV.
    Covers: 3 datasets × 2 algorithms × 2 noise levels × 2 extra-feature ratios.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print(f"\n  Running full experiment suite ({len(DATASETS)} datasets × "
          f"{len(ALGORITHMS)} algorithms × {len(NOISE_LEVELS)} noise levels × "
          f"{len(EXTRA_RATIOS)} extra-feature ratios)...\n")

    for dataset in DATASETS:
        for algo in ALGORITHMS:
            for noise in NOISE_LEVELS:
                for extra in EXTRA_RATIOS:
                    try:
                        res = run_experiment(
                            dataset_name=dataset,
                            search_algo=algo,
                            noise_sigma=noise,
                            extra_ratio=extra,
                            label_flip=0.05 if (noise > 0 or extra > 0) else 0.0,
                            verbose=False,
                        )
                        all_results.append(res)
                        print(f"  ✓  {dataset:<15} {algo:<10} "
                              f"noise={noise}  extra={extra}  "
                              f"acc={res['agent_accuracy']:.3f}  "
                              f"Δ={res['accuracy_delta']:+.3f}  "
                              f"feats={res['n_features_selected']}/{res['n_features_total']}")
                    except Exception as e:
                        print(f"  ✗  {dataset} {algo} noise={noise} extra={extra}: {e}")

    df = pd.DataFrame(all_results)
    out_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # ── Print summary ──
    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)
    avg_delta = df["accuracy_delta"].mean()
    avg_red   = df["dimensionality_reduction_pct"].mean()
    avg_evals = df["subsets_evaluated"].mean()
    wins = (df["accuracy_delta"] > 0).sum()
    print(f"  Avg. accuracy improvement      : {avg_delta:+.4f}")
    print(f"  Avg. dimensionality reduction  : {avg_red:.1f}%")
    print(f"  Avg. subsets evaluated         : {avg_evals:.0f}")
    print(f"  Conditions with improvement    : {wins}/{len(df)}")
    print("="*60)

    return df


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="CS3081 Feature Selection AI Agent — Effat University Spring 2026"
    )
    parser.add_argument(
        "--dataset", default="breast_cancer",
        choices=["breast_cancer", "wine", "ionosphere", "spambase", "madelon", "all"],
        help="Dataset to run on (default: breast_cancer)"
    )
    parser.add_argument(
        "--search", default="astar",
        choices=["forward", "astar"],
        help="Search algorithm: 'forward' (greedy) or 'astar' (optimal) (default: astar)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Gaussian noise sigma to inject (default: 0.0 = clean)"
    )
    parser.add_argument(
        "--extra", type=float, default=0.0,
        help="Ratio of irrelevant features to inject, e.g. 0.5 = 50%% extra (default: 0.0)"
    )
    parser.add_argument(
        "--flip", type=float, default=0.0,
        help="Fraction of labels to randomly flip (default: 0.0)"
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds for Naive Bayes oracle (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--experiment", action="store_true",
        help="Run the full experiment suite across all datasets and conditions"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for experiment CSV results (default: results/)"
    )

    args = parser.parse_args()

    if args.experiment or args.dataset == "all":
        run_full_suite(output_dir=args.output)
    else:
        run_experiment(
            dataset_name=args.dataset,
            search_algo=args.search,
            noise_sigma=args.noise,
            extra_ratio=args.extra,
            label_flip=args.flip,
            cv_folds=args.cv,
            seed=args.seed,
            verbose=True,
        )


if __name__ == "__main__":
    main()
