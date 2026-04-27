"""
environment.py
--------------
Dataset loading and adversarial perturbation utilities.

CS3081: Artificial Intelligence · Effat University · Spring 2026
Team: Faisal Yahya (S23208857), Faisal Shamsi, Omar Almutairi
Instructor: Dr. Naila Marir

Provides:
  - load_dataset()              → clean dataset loader
  - create_adversarial_dataset()→ injects noise, extra features, label flips
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


# ─────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a named dataset. Returns (X, y, feature_names).

    Supported datasets:
        breast_cancer  — 569 samples, 30 features, binary classification
        wine           — 178 samples, 13 features, 3-class classification
        ionosphere     — 351 samples, 34 features, binary classification

    Args:
        name: dataset identifier string

    Returns:
        X:             raw feature matrix (NOT scaled)
        y:             integer class labels
        feature_names: list of feature name strings
    """
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, list(data.feature_names)

    elif name == "wine":
        data = load_wine()
        return data.data, data.target, list(data.feature_names)

    elif name == "ionosphere":
        # Ionosphere: load from sklearn or fall back to synthetic
        try:
            from sklearn.datasets import fetch_openml
            ds = fetch_openml(name="ionosphere", version=1, as_frame=False,
                              parser="auto")
            X = ds.data.astype(float)
            # Target is 'g'/'b' — encode as 1/0
            y = (ds.target == "g").astype(int)
            feat_names = [f"feature_{i}" for i in range(X.shape[1])]
            return X, y, feat_names
        except Exception:
            # Fallback: synthetic 34-feature binary dataset
            rng = np.random.default_rng(0)
            X = rng.standard_normal((351, 34))
            y = (X[:, 0] + X[:, 2] > 0).astype(int)
            feat_names = [f"feature_{i}" for i in range(34)]
            return X, y, feat_names

    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Choose from: breast_cancer, wine, ionosphere"
        )


# ─────────────────────────────────────────────────────────────────────
# ADVERSARIAL DATASET FACTORY
# ─────────────────────────────────────────────────────────────────────

def create_adversarial_dataset(
        dataset_name: str,
        noise_sigma: float = 0.0,
        extra_ratio: float = 0.0,
        label_flip: float = 0.0,
        seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    """
    Load a dataset and inject adversarial perturbations.

    Perturbation types:
      1. Gaussian noise   — adds N(0, noise_sigma²) to every feature
      2. Extra features   — appends purely random (irrelevant) columns
      3. Label flipping   — randomly corrupts a fraction of labels

    Args:
        dataset_name: one of breast_cancer, wine, ionosphere
        noise_sigma:  std-dev of Gaussian noise added to features
        extra_ratio:  fraction of original features to add as random columns
                      e.g. 0.5 → adds 50% more random features
        label_flip:   fraction of labels to randomly corrupt
        seed:         RNG seed for reproducibility

    Returns:
        X_adv:       perturbed + scaled feature matrix
        y_adv:       (possibly corrupted) labels
        n_original:  number of original (non-injected) features
        feat_names:  list of all feature names (original + injected)
    """
    rng = np.random.default_rng(seed)

    # ── Load raw data ──
    X_raw, y, feat_names = load_dataset(dataset_name)
    n_samples, n_original = X_raw.shape

    # ── Scale features ──
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── Perturbation 1: Gaussian noise ──
    if noise_sigma > 0:
        X = X + rng.normal(0, noise_sigma, size=X.shape)

    # ── Perturbation 2: Extra irrelevant features ──
    n_extra = int(n_original * extra_ratio)
    if n_extra > 0:
        X_extra = rng.standard_normal((n_samples, n_extra))
        X = np.hstack([X, X_extra])
        extra_names = [f"noise_feat_{i}" for i in range(n_extra)]
        feat_names = list(feat_names) + extra_names

    # ── Perturbation 3: Label flipping ──
    y_adv = y.copy()
    if label_flip > 0:
        n_flip = int(n_samples * label_flip)
        flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
        n_classes = len(np.unique(y))
        for i in flip_idx:
            # Assign a different class randomly
            other_classes = [c for c in range(n_classes) if c != y_adv[i]]
            y_adv[i] = rng.choice(other_classes)

    return X, y_adv, n_original, feat_names
