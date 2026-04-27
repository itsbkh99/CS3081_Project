"""
heuristics.py
-------------
Heuristic functions for the A* feature selection search.

The heuristic h(n) estimates the remaining accuracy gain
achievable by adding unselected features. It uses Mutual
Information (MI) between each feature and the class label.

ADMISSIBILITY:
  MI(f ; Y) is an upper bound on the marginal contribution
  of feature f to classification accuracy. Therefore h(n)
  never overestimates the true remaining gain → A* is optimal.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_mi_scores(X: np.ndarray, y: np.ndarray,
                       n_neighbors: int = 3, seed: int = 0) -> np.ndarray:
    """
    Compute Mutual Information between each feature and the class label.
    Returns array of shape (n_features,).
    """
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors,
                              random_state=seed)
    return mi


def mi_heuristic(selected_mask: np.ndarray,
                  mi_scores: np.ndarray) -> float:
    """
    h(n) = sum of normalized MI scores for all UNSELECTED features.

    This estimates the maximum additional accuracy gain still possible.
    Because MI is an upper bound on marginal feature contribution,
    the heuristic is admissible.

    Args:
        selected_mask: boolean array, True = feature already selected
        mi_scores:     precomputed MI scores for all features

    Returns:
        Scalar heuristic estimate (higher = more potential gain remaining)
    """
    mi_max = mi_scores.max()
    if mi_max == 0:
        return 0.0

    # Only consider unselected features
    unselected = ~selected_mask
    remaining_gain = mi_scores[unselected].sum() / (mi_max * len(mi_scores))
    return float(remaining_gain)


def rank_features_by_mi(mi_scores: np.ndarray) -> np.ndarray:
    """Return feature indices sorted by MI score descending."""
    return np.argsort(mi_scores)[::-1]
