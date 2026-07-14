"""Factory for creating recommendation models."""
from typing import Any
import torch.nn as nn
from sklearn.base import BaseEstimator
from src.models.mlp import RecommenderMLP
from src.models.baseline import BaselineModel

class ModelFactory:
    """Factory class to instantiate models (Factory Pattern)."""

    @staticmethod
    def create_model(model_type: str, **kwargs: Any) -> nn.Module | BaseEstimator:
        """Create a model based on the type.

        Args:
            model_type (str): Type of the model ('mlp' or 'baseline').
            **kwargs: Arguments to pass to the model.

        Returns:
            Model instance.
        """
        if model_type == "mlp":
            return RecommenderMLP(**kwargs)
        if model_type == "baseline":
            return BaselineModel(**kwargs)
        raise ValueError(f"Unknown model type: {model_type}")
