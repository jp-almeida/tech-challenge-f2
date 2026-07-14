"""Baseline model using Scikit-Learn."""
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class BaselineModel(BaseEstimator):
    """Random Forest classifier used as a recommendation baseline."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        """Initialize the baseline with deterministic parameters."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def fit(self, features: Any, target: Any) -> "BaselineModel":
        """Fit the classifier."""
        self.model.fit(features, target)
        return self

    def predict(self, features: Any) -> Any:
        """Predict purchase labels."""
        return self.model.predict(features)

    def predict_proba(self, features: Any) -> Any:
        """Predict purchase probabilities."""
        return self.model.predict_proba(features)[:, 1]
