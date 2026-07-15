"""Train and register recommendation models with MLflow."""
import itertools
import json
import os
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

# Nome do experimento no MLflow, definido pela variável de ambiente
# PIPELINE_VERSION. Isso permite rodar o mesmo comando em cada branch
# (v1-baseline / v2-melhorado) e ter os runs automaticamente separados
# em experimentos distintos na UI do MLflow, sem editar este arquivo.
#
# Uso:
#   PIPELINE_VERSION=v1-baseline python -m src.training.train
#   PIPELINE_VERSION=v2-melhorado python -m src.training.train
#
# Se a variável não for definida, assume "default".
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "default")
EXPERIMENT_NAME = f"recommender-{PIPELINE_VERSION}"

# Grade de busca de hiperparâmetros (Fase 9 / Etapa 6).
# 3 * 2 * 2 * 2 = 24 combinações de MLP, cada uma vira um run no MLflow.
HIDDEN_DIM_OPTIONS = (64, 128, 256)
LEARNING_RATE_OPTIONS = (0.001, 0.0005)
DROPOUT_OPTIONS = (0.2, 0.3)
BATCH_SIZE_OPTIONS = (128, 256)
EMBEDDING_DIM = 32


def set_seed(seed: int) -> None:
    """Set deterministic seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_embedding_config() -> dict[str, int]:
    """Load number of unique users/items produced by feature engineering."""
    return json.loads(Path("models/embedding_config.json").read_text(encoding="utf-8"))


def run_baseline(
    algorithm: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    """Train and evaluate a baseline model."""

    with mlflow.start_run(run_name=algorithm):

        model = ModelFactory.create_model(
            "baseline",
            algorithm=algorithm,
            random_state=settings.random_seed,
        )

        model.fit(train_x, train_y)

        probabilities = model.predict_proba(test_x)

        metrics = calculate_binary_metrics(
            test_y,
            probabilities,
        )

        mlflow.log_param("algorithm", algorithm)
        mlflow.log_metrics(metrics)

    return metrics


def run_mlp(
    hidden_dim: int,
    learning_rate: float,
    dropout: float,
    batch_size: int,
    num_users: int,
    num_items: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[torch.nn.Module, dict[str, float], str, dict]:
    """Train one MLP configuration (uma combinação da grade de busca)."""

    num_continuous_features = train_x.shape[1] - 2  # exclui user_idx e item_idx

    run_name = f"mlp_h{hidden_dim}_lr{learning_rate}_do{dropout}_bs{batch_size}"

    with mlflow.start_run(run_name=run_name) as run:

        model = ModelFactory.create_model(
            "mlp",
            num_users=num_users,
            num_items=num_items,
            num_continuous_features=num_continuous_features,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        model, train_loss = train_mlp(
            model,
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32).view(-1, 1),
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(test_x, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).numpy().ravel()

        metrics = calculate_binary_metrics(
            test_y,
            probabilities,
        )

        hyperparameters = {
            "model": "mlp",
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "batch_size": batch_size,
            "embedding_dim": EMBEDDING_DIM,
            "num_users": num_users,
            "num_items": num_items,
            "num_continuous_features": num_continuous_features,
        }

        mlflow.log_params(hyperparameters)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metrics(metrics)

        mlflow.pytorch.log_model(model, "model")

    return model, metrics, run.info.run_id, hyperparameters


def register_production_model(run_id: str) -> None:
    """Register best MLP."""

    version = mlflow.register_model(
        f"runs:/{run_id}/model",
        MODEL_NAME,
    )

    MlflowClient().transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )


def save_report(metrics: dict[str, float], hyperparameters: dict) -> None:
    """Save metrics and winning hyperparameters for DVC/evaluate.py."""

    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    Path("reports/training_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    Path("models/model_config.json").write_text(
        json.dumps(hyperparameters, indent=2),
        encoding="utf-8",
    )


def main() -> None:

    set_seed(settings.random_seed)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Running pipeline version: {PIPELINE_VERSION} (experiment: {EXPERIMENT_NAME})")

    train_x, test_x, train_y, test_y = load_split_data(
        settings.features_path,
        settings.random_seed,
    )

    embedding_config = load_embedding_config()
    num_users = embedding_config["num_users"]
    num_items = embedding_config["num_items"]

    logistic_metrics = run_baseline(
        "logistic_regression",
        train_x,
        train_y,
        test_x,
        test_y,
    )

    random_forest_metrics = run_baseline(
        "random_forest",
        train_x,
        train_y,
        test_x,
        test_y,
    )

    search_space = itertools.product(
        HIDDEN_DIM_OPTIONS,
        LEARNING_RATE_OPTIONS,
        DROPOUT_OPTIONS,
        BATCH_SIZE_OPTIONS,
    )

    candidates = [
        run_mlp(
            hidden_dim,
            learning_rate,
            dropout,
            batch_size,
            num_users,
            num_items,
            train_x,
            train_y,
            test_x,
            test_y,
        )
        for hidden_dim, learning_rate, dropout, batch_size in search_space
    ]

    best_model, mlp_metrics, run_id, best_hyperparameters = max(
        candidates,
        key=lambda candidate: candidate[1]["f1"],
    )

    final_metrics = dict(mlp_metrics)
    final_metrics["random_forest_f1"] = random_forest_metrics["f1"]
    final_metrics["logistic_regression_f1"] = logistic_metrics["f1"]

    torch.save(best_model.state_dict(), settings.model_path)

    save_report(
        final_metrics,
        best_hyperparameters,
    )

    register_production_model(run_id)

    # Etapa 9: registra reports/ e models/ como artefatos do MLflow,
    # garantindo que cada execução fique completa e rastreável.
    with mlflow.start_run(run_name="final_artifacts"):
        mlflow.log_artifacts("reports", artifact_path="reports")
        mlflow.log_artifacts("models", artifact_path="models")

    print("Training complete.")
    print(f"Best hyperparameters: {best_hyperparameters}")


if __name__ == "__main__":
    main()
    