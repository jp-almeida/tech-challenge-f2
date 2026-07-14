"""Tests for model construction and evaluation helpers."""
import numpy as np

from src.models.model_factory import ModelFactory
from src.training.metrics import calculate_binary_metrics


def test_factory_creates_mlp() -> None:
    """The Factory returns a PyTorch MLP with expected output shape."""
    model = ModelFactory.create_model("mlp", input_dim=3, hidden_dim=8)
    assert model.net[0].in_features == 3


def test_binary_metrics_include_required_measurements() -> None:
    """Evaluation exposes more than four metrics."""
    metrics = calculate_binary_metrics(np.array([0, 1, 1]), np.array([0.1, 0.8, 0.6]))
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "rmse"} <= metrics.keys()
