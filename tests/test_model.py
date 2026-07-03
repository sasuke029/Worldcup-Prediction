"""
Minimal sanity tests for the from-scratch logistic regression model.
Run with: python -m pytest tests/
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import LogisticRegressionScratch
from src.metrics import accuracy, precision, recall, f1_score, roc_auc
from src.cross_validation import train_test_split_stratified, standard_scale


def make_separable_data(n=200, seed=0):
    """A trivially linearly separable dataset the model must fit well."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


def test_sigmoid_bounds():
    model = LogisticRegressionScratch()
    z = np.array([-1000, 0, 1000])
    p = model._sigmoid(z)
    assert np.all(p >= 0) and np.all(p <= 1)
    assert np.isclose(p[1], 0.5)


def test_fits_separable_data_well():
    X, y = make_separable_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, seed=1)
    X_train_s, X_test_s, _, _ = standard_scale(X_train, X_test)

    model = LogisticRegressionScratch(learning_rate=0.5, n_iterations=5000).fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    assert accuracy(y_test, preds) > 0.9


def test_metrics_agree_on_perfect_predictions():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = y_true.copy()
    assert accuracy(y_true, y_pred) == 1.0
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred) == 1.0


def test_auc_of_random_guessing_near_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=2000).astype(float)
    y_scores = rng.random(2000)
    auc = roc_auc(y_true, y_scores)
    assert 0.45 < auc < 0.55
