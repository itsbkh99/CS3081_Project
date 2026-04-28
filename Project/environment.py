"""
environment.py
--------------
Dataset loading and adversarial perturbation utilities.

CS3081: Artificial Intelligence · Effat University · Spring 2026
Team: Faisal Yahya (S23208857), Faisal Shamsi, Omar Almutairi
Instructor: Dr. Naila Marir
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List


# ─────────────────────────────────────────────────────────────────────
# SMART TARGET DETECTOR
# ─────────────────────────────────────────────────────────────────────

TARGET_KEYWORDS = [
    "target", "label", "class", "outcome", "output", "result",
    "survived", "diagnosis", "disease", "diabetes", "cancer",
    "churn", "fraud", "default", "spam", "response", "status",
    "risk", "grade", "category", "type", "prediction", "y",
    "heart_disease", "stroke", "infection", "death", "mortality",
    "score", "rating", "happiness", "satisfaction"
]

def detect_target_column(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    for col in cols:
        if col.lower() in TARGET_KEYWORDS:
            return col
    for col in cols:
        for kw in TARGET_KEYWORDS:
            if kw in col.lower():
                return col
    for col in cols:
        if df[col].nunique() == 2:
            return col
    candidates = {col: df[col].nunique() for col in cols if df[col].nunique() <= 20}
    if candidates:
        return min(candidates, key=candidates.get)
    return cols[-1]


# ─────────────────────────────────────────────────────────────────────
# AUTO CSV LOADER
# ─────────────────────────────────────────────────────────────────────

def load_csv_dataset(filepath: str, target_col: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(filepath)
    print(f"\n  Loaded : {os.path.basename(filepath)}")
    print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} columns")

    # Auto-detect target
    if target_col is None:
        target_col = detect_target_column(df)
        print(f"  Target : '{target_col}' (auto-detected)")
    else:
        if target_col not in df.columns:
            raise ValueError(
                f"Column '{target_col}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
        print(f"  Target : '{target_col}'")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # Separate target
    y_raw = df[target_col].reset_index(drop=True)
    df = df.drop(columns=[target_col]).reset_index(drop=True)

    # Drop non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropped non-numeric cols: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # Drop fully empty columns
    df = df.dropna(axis=1, how='all')

    # Fill remaining missing values with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Drop any columns that still have nulls
    df = df.dropna(axis=1)

    X = df.values.astype(float)
    feat_names = list(df.columns)

    # Encode target
    n_unique = y_raw.nunique()
    if y_raw.dtype == object or n_unique <= 20:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        print(f"  Classes: {list(le.classes_)}")
    else:
        # Continuous → split into 3 tiers using quantiles
        low  = y_raw.quantile(0.33)
        high = y_raw.quantile(0.67)
        binned = pd.cut(y_raw, bins=[-np.inf, low, high, np.inf], labels=[0, 1, 2])
        # Fill any NaN from cut edges with nearest bin
        binned = binned.fillna(1)
        y = binned.astype(int).values
        print(f"  Continuous target split into 3 tiers (Low=0 / Mid=1 / High=2)")

    print(f"  Features ({len(feat_names)}): {feat_names}")
    print(f"  Final shape: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y, feat_names


# ─────────────────────────────────────────────────────────────────────
# MAIN DATASET LOADER
# ─────────────────────────────────────────────────────────────────────

_TARGET_COL = None

def set_target(col: str):
    global _TARGET_COL
    _TARGET_COL = col


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, list(data.feature_names)

    elif name == "wine":
        data = load_wine()
        return data.data, data.target, list(data.feature_names)

    elif name == "ionosphere":
        try:
            from sklearn.datasets import fetch_openml
            ds = fetch_openml(name="ionosphere", version=1, as_frame=False, parser="auto")
            X = ds.data.astype(float)
            y = (ds.target == "g").astype(int)
            feat_names = [f"feature_{i}" for i in range(X.shape[1])]
            return X, y, feat_names
        except Exception:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((351, 34))
            y = (X[:, 0] + X[:, 2] > 0).astype(int)
            feat_names = [f"feature_{i}" for i in range(34)]
            return X, y, feat_names

    # Auto CSV loader
    for candidate in [name, name + ".csv"]:
        if os.path.isfile(candidate):
            return load_csv_dataset(candidate, target_col=_TARGET_COL)

    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    raise ValueError(
        f"Dataset '{name}' not found.\n"
        f"  Built-in options : breast_cancer, wine, ionosphere\n"
        f"  CSV files in your folder: {csv_files}\n"
        f"  Usage: python3 agent.py --dataset your_file.csv"
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
    rng = np.random.default_rng(seed)

    X_raw, y, feat_names = load_dataset(dataset_name)
    n_samples, n_original = X_raw.shape

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    if noise_sigma > 0:
        X = X + rng.normal(0, noise_sigma, size=X.shape)

    n_extra = int(n_original * extra_ratio)
    if n_extra > 0:
        X_extra = rng.standard_normal((n_samples, n_extra))
        X = np.hstack([X, X_extra])
        feat_names = list(feat_names) + [f"noise_feat_{i}" for i in range(n_extra)]

    y_adv = y.copy()
    if label_flip > 0:
        n_flip = int(n_samples * label_flip)
        flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
        n_classes = len(np.unique(y))
        for i in flip_idx:
            other_classes = [c for c in range(n_classes) if c != y_adv[i]]
            y_adv[i] = rng.choice(other_classes)

    return X, y_adv, n_original, feat_names
