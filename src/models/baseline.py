"""Baseline model using Scikit-Learn."""
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class BaselineModel(BaseEstimator):
    """Baseline wrapper for classical Scikit-Learn classifiers."""

    def __init__(
        self,
        algorithm: str = "random_forest",
        random_state: int = 42,
        **kwargs,
    ) -> None:

        if algorithm == "random_forest":
            self.model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=kwargs.get("n_estimators", 100),
                n_jobs=-1,
            )

        elif algorithm == "logistic_regression":
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
            )

        else:
            raise ValueError(f"Unknown baseline algorithm: {algorithm}")

    def fit(self, features: Any, target: Any):
        self.model.fit(features, target)
        return self

    def predict(self, features: Any):
        return self.model.predict(features)

    def predict_proba(self, features: Any):
        return self.model.predict_proba(features)[:, 1]