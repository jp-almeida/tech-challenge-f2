"""Train and register recommendation models with MLflow."""
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient

from src.configs.settings import settings
from src.models.model_factory import ModelFactory
from src.training.data import load_split_data
from src.training.metrics import calculate_binary_metrics
from src.training.mlp_trainer import train_mlp

MODEL_NAME = "EcommerceRecommender"
MLP_CONFIGURATIONS = (32, 64)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for supported random generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_baseline(features: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Train and track the Random Forest baseline."""
    with mlflow.start_run(run_name="random_forest_baseline"):
        model = ModelFactory.create_model("baseline", random_state=settings.random_seed)
        model.fit(features, target)
        metrics = calculate_binary_metrics(target, model.predict_proba(features))
        mlflow.log_params({"model": "random_forest", "n_estimators": 100})
        mlflow.log_metrics(metrics)
    return metrics


def run_mlp(hidden_dim: int, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> tuple[torch.nn.Module, dict[str, float], str]:
    """Train, track, and return one MLP configuration."""
    with mlflow.start_run(run_name=f"mlp_hidden_{hidden_dim}") as run:
        model = ModelFactory.create_model("mlp", input_dim=3, hidden_dim=hidden_dim)
        model, train_loss = train_mlp(
            model,
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32).view(-1, 1),
        )
        probabilities = model(torch.tensor(test_x, dtype=torch.float32)).detach().numpy().ravel()
        metrics = calculate_binary_metrics(test_y, probabilities)
        mlflow.log_params({"model": "mlp", "hidden_dim": hidden_dim, "early_stopping_patience": 5})
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")
    return model, metrics, run.info.run_id


def register_production_model(run_id: str) -> None:
    """Register the selected run and promote it to Production."""
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    MlflowClient().transition_model_version_stage(
        name=MODEL_NAME, version=version.version, stage="Production", archive_existing_versions=True
    )


def save_report(metrics: dict[str, float], hidden_dim: int) -> None:
    """Persist metrics for DVC reporting."""
    path = Path("reports/training_metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path("models/model_config.json").write_text(
        json.dumps({"hidden_dim": hidden_dim}), encoding="utf-8"
    )


def main() -> None:
    """Execute baseline comparison, MLP selection, and model registration."""
    set_seed(settings.random_seed)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    train_x, test_x, train_y, test_y = load_split_data(settings.features_path, settings.random_seed)
    baseline_metrics = run_baseline(test_x, test_y)
    candidates = [run_mlp(hidden_dim, train_x, train_y, test_x, test_y) for hidden_dim in MLP_CONFIGURATIONS]
    best_model, metrics, run_id = max(candidates, key=lambda candidate: candidate[1]["f1"])
    metrics["baseline_f1"] = baseline_metrics["f1"]
    Path(settings.model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), settings.model_path)
    hidden_dim = next(size for size, candidate in zip(MLP_CONFIGURATIONS, candidates) if candidate[2] == run_id)
    save_report(metrics, hidden_dim)
    register_production_model(run_id)
    print("Training complete. Best MLP registered in Production.")


if __name__ == "__main__":
    main()
