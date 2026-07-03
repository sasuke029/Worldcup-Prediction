"""From-scratch evaluation metrics for binary classification."""

import numpy as np
from typing import Tuple, Dict


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute confusion matrix counts (TP, TN, FP, FN).
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Tuple of (tp, tn, fp, fn).
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy: (TP + TN) / total."""
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision: TP / (TP + FP)."""
    tp, _, fp, _ = confusion_counts(y_true, y_pred)
    return float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall (sensitivity): TP / (TP + FN)."""
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    return float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score: 2 * (precision * recall) / (precision + recall)."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2 * p * r / (p + r) if (p + r) > 0 else 0.0)


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney U) formulation.
    
    Args:
        y_true: True labels.
        y_scores: Predicted probabilities or scores.
        
    Returns:
        ROC AUC score between 0 and 1.
    """
    order: np.ndarray = np.argsort(y_scores)
    ranks: np.ndarray = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float(np.nan)
    sum_pos_ranks = np.sum(ranks[y_true == 1])
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: np.ndarray) -> Dict[str, float]:
    """Convenience wrapper returning every metric as a dict.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        
    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc(y_true, y_proba),
    }
