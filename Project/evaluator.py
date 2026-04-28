"""
evaluator.py
------------
Naive Bayes classifier used as the oracle to score feature subsets.

The evaluator:
  1. Trains a GaussianNB on the selected feature columns
  2. Uses k-fold cross-validation for a reliable accuracy estimate
  3. Caches results in a knowledge base to avoid re-evaluation
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from functools import lru_cache
from typing import Tuple


class NaiveBayesEvaluator:
    """
    Oracle that scores feature subsets using Naive Bayes + cross-validation.
    Includes an internal cache (knowledge base) to avoid redundant evaluations.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 cv_folds: int = 5, scoring: str = "accuracy", seed: int = 0):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.seed = seed
        self._cache: dict = {}          # Knowledge base: mask_key → score
        self.n_evaluations: int = 0     # How many unique subsets evaluated

    def _mask_to_key(self, mask: np.ndarray) -> tuple:
        """Convert boolean mask to hashable key."""
        return tuple(mask.tolist())

    def evaluate(self, mask: np.ndarray) -> float:
        """
        Evaluate a feature subset defined by a boolean mask.
        Returns mean cross-validated accuracy. Cached after first call.
        """
        # Must select at least 1 feature
        if mask.sum() == 0:
            return 0.0

        key = self._mask_to_key(mask)

        # Return cached result if available (knowledge base hit)
        if key in self._cache:
            return self._cache[key]

        # Extract selected feature columns
        X_sub = self.X[:, mask]

        # 5-fold cross-validated Naive Bayes
        clf = GaussianNB()
        scores = cross_val_score(clf, X_sub, self.y,
                                  cv=self.cv_folds,
                                  scoring=self.scoring)
        acc = float(scores.mean())

        # Store in knowledge base
        self._cache[key] = acc
        self.n_evaluations += 1
        return acc

    def cache_size(self) -> int:
        return len(self._cache)

    def best_cached(self) -> Tuple[tuple, float]:
        """Return the best (mask_key, score) seen so far."""
        if not self._cache:
            return None, 0.0
        best_key = max(self._cache, key=self._cache.get)
        return best_key, self._cache[best_key]
