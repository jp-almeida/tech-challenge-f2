"""Metrics shared by training and evaluation."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    root_mean_squared_error,
)


def calculate_binary_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate six classification and probability-quality metrics."""
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "rmse": float(root_mean_squared_error(target, probabilities)),
    }
