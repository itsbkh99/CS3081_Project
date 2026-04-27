"""
search.py
---------
Search algorithms for feature selection.

Algorithms implemented:
  1. forward_selection  — greedy baseline
  2. astar_search       — optimal, heuristic-guided (A*)

Each algorithm returns a SearchResult dataclass.

CS3081: Artificial Intelligence · Effat University · Spring 2026
Team: Faisal Yahya (S23208857), Faisal Shamsi, Omar Almutairi
Instructor: Dr. Naila Marir
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Optional

from evaluator import NaiveBayesEvaluator
from heuristics import mi_heuristic, rank_features_by_mi


@dataclass
class SearchResult:
    selected_features: List[int]       # Indices of selected features
    accuracy: float                     # Cross-validated accuracy
    n_features_selected: int
    n_subsets_evaluated: int
    algorithm: str
    history: List[float] = field(default_factory=list)  # Accuracy per step


# ─────────────────────────────────────────────
# 1. GREEDY FORWARD SELECTION (baseline)
# ─────────────────────────────────────────────

def forward_selection(evaluator: NaiveBayesEvaluator,
                       n_features: int,
                       min_improvement: float = 0.001) -> SearchResult:
    """
    Sequential Forward Selection: start with empty set, greedily add
    the feature that most improves cross-validated accuracy at each step.
    Stops when no feature improves accuracy by more than min_improvement.

    Time complexity: O(d^2) evaluations in the worst case.
    Space complexity: O(d) — stores only the current best subset.
    """
    selected = []
    best_acc = 0.0
    history = []

    for _ in range(n_features):
        best_new_feature = None
        best_new_acc = best_acc

        # Try adding each unselected feature
        for f in range(n_features):
            if f in selected:
                continue
            candidate = selected + [f]
            mask = np.zeros(n_features, dtype=bool)
            mask[candidate] = True
            acc = evaluator.evaluate(mask)

            if acc > best_new_acc:
                best_new_acc = acc
                best_new_feature = f

        # Stop if no improvement found above the threshold
        if best_new_feature is None or (best_new_acc - best_acc) < min_improvement:
            break

        selected.append(best_new_feature)
        best_acc = best_new_acc
        history.append(best_acc)

    return SearchResult(
        selected_features=selected,
        accuracy=best_acc,
        n_features_selected=len(selected),
        n_subsets_evaluated=evaluator.n_evaluations,
        algorithm="Forward Selection",
        history=history,
    )


# ─────────────────────────────────────────────
# 2. A* SEARCH (optimal, heuristic-guided)
# ─────────────────────────────────────────────

def astar_search(evaluator: NaiveBayesEvaluator,
                  n_features: int,
                  mi_scores: np.ndarray,
                  beam_width: Optional[int] = None,
                  max_features: Optional[int] = None) -> SearchResult:
    """
    A* search over the feature-subset lattice.

    State:  frozenset of selected feature indices
    Cost:   -accuracy (minimising cost = maximising accuracy)

    f(n) = g(n) + h(n)
        g(n) = -current_accuracy (negative because we minimise)
        h(n) = -mi_heuristic    (estimated remaining gain)

    The MI heuristic is ADMISSIBLE — it never overestimates the true
    remaining accuracy gain — so A* is guaranteed to return the optimal
    feature subset given sufficient evaluation budget.

    Args:
        evaluator:    NaiveBayesEvaluator oracle (with internal cache)
        n_features:   total number of features in the dataset
        mi_scores:    precomputed mutual information scores (shape: n_features,)
        beam_width:   if set, limits open-set size (beam search approximation)
        max_features: hard cap on subset size (None = no cap)
    """
    if max_features is None:
        max_features = n_features

    # Initialise search from the empty set
    start_mask = np.zeros(n_features, dtype=bool)
    h0 = mi_heuristic(start_mask, mi_scores)

    # Priority queue entries: (f_score, -accuracy, tie_break_id, selected_frozenset)
    counter = 0
    heap = [(-(0.0 + h0), -0.0, counter, frozenset())]
    visited = set()

    best_result = SearchResult([], 0.0, 0, 0, "A*")
    history = []

    # Explore highest-MI features first to guide expansion order
    feature_order = rank_features_by_mi(mi_scores)

    while heap:
        # Optional beam pruning: keep only the best beam_width nodes
        if beam_width:
            heap = heapq.nsmallest(beam_width, heap)
            heapq.heapify(heap)

        neg_f, neg_acc, _, selected_set = heapq.heappop(heap)
        current_acc = -neg_acc

        # Skip if already visited
        if selected_set in visited:
            continue
        visited.add(selected_set)

        # Update global best
        if current_acc > best_result.accuracy and len(selected_set) > 0:
            best_result.selected_features = list(selected_set)
            best_result.accuracy = current_acc
            history.append(current_acc)

        # Stop expanding if we've reached the max allowed subset size
        if len(selected_set) >= max_features:
            continue

        # Expand: generate child nodes by adding one unselected feature
        for f in feature_order:
            if f in selected_set:
                continue

            new_set = selected_set | {f}
            if new_set in visited:
                continue

            # Build mask and evaluate
            mask = np.zeros(n_features, dtype=bool)
            for idx in new_set:
                mask[idx] = True

            acc = evaluator.evaluate(mask)
            h = mi_heuristic(mask, mi_scores)
            f_score = -(acc + h)

            counter += 1
            heapq.heappush(heap, (f_score, -acc, counter, new_set))

        # Budget cap: stop after evaluating too many subsets on large datasets
        if evaluator.n_evaluations > 2000:
            break

    best_result.n_features_selected = len(best_result.selected_features)
    best_result.n_subsets_evaluated = evaluator.n_evaluations
    best_result.algorithm = "A*"
    best_result.history = history
    return best_result
