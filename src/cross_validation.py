"""Train/test splitting, feature scaling, and k-fold CV, all from scratch."""

import numpy as np
from numpy.random import Generator
import pandas as pd
from pandas import DataFrame
from typing import Tuple

from .model import LogisticRegressionScratch
from .metrics import accuracy, precision, recall, f1_score, roc_auc


def train_test_split_stratified(X: np.ndarray, y: np.ndarray, 
                               test_size: float = 0.2, 
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split preserving class balance.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target vector of shape (n_samples,).
        test_size: Fraction of data for test set.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    rng: Generator = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_test = int(len(idx_pos) * test_size)
    n_neg_test = int(len(idx_neg) * test_size)
    test_idx: np.ndarray = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx: np.ndarray = np.setdiff1d(np.arange(len(y)), test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standard_scale(X_train: np.ndarray, 
                   X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit scaling on train only, to avoid leaking test-set statistics.
    
    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, mean, std).
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def k_fold_cv(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42, 
              **model_kwargs) -> DataFrame:
    """Perform k-fold cross-validation.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target vector of shape (n_samples,).
        k: Number of folds.
        seed: Random seed for reproducibility.
        **model_kwargs: Arguments passed to LogisticRegressionScratch.
        
    Returns:
        DataFrame with metrics (accuracy, precision, recall, f1, auc) per fold.
    """
    rng: Generator = np.random.default_rng(seed)
    idx: np.ndarray = rng.permutation(len(y))
    folds: list = np.array_split(idx, k)
    results = []

    for i in range(k):
        val_idx = folds[i]
        train_idx: np.ndarray = np.concatenate([folds[j] for j in range(k) if j != i])

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_s, X_val_s, _, _ = standard_scale(X_tr, X_val)

        model = LogisticRegressionScratch(**model_kwargs).fit(X_tr_s, y_tr)
        proba = model.predict_proba(X_val_s)
        pred = (proba >= 0.5).astype(int)

        results.append({
            "fold": i + 1,
            "accuracy": accuracy(y_val, pred),
            "precision": precision(y_val, pred),
            "recall": recall(y_val, pred),
            "f1": f1_score(y_val, pred),
            "auc": roc_auc(y_val, proba),
        })
    return pd.DataFrame(results)
