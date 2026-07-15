"""Evaluate the selected MLP in a separate DVC stage."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

from src.configs.settings import settings
from src.models.model_factory import ModelFactory
from src.training.data import load_split_data
from src.training.metrics import calculate_binary_metrics, calculate_ranking_metrics

REPORTS_DIR = Path("reports")


def save_confusion_matrix(target, predictions) -> None:
    matrix = confusion_matrix(target, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()


def save_roc_curve(target, probabilities) -> None:
    RocCurveDisplay.from_predictions(target, probabilities)
    plt.title("ROC Curve")
    plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
    plt.close()


def save_precision_recall_curve(target, probabilities) -> None:
    PrecisionRecallDisplay.from_predictions(target, probabilities)
    plt.title("Precision-Recall Curve")
    plt.savefig(REPORTS_DIR / "precision_recall.png", bbox_inches="tight")
    plt.close()


def save_classification_report(target, predictions) -> None:
    report = classification_report(target, predictions, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    """Load the selected model and persist holdout metrics, plots and ranking metrics."""
    _, test_features, _, test_target = load_split_data(settings.features_path, settings.random_seed)

    configuration = json.loads(Path("models/model_config.json").read_text(encoding="utf-8"))

    model = ModelFactory.create_model(
        "mlp",
        num_users=configuration["num_users"],
        num_items=configuration["num_items"],
        num_continuous_features=configuration["num_continuous_features"],
        embedding_dim=configuration["embedding_dim"],
        hidden_dim=configuration["hidden_dim"],
        dropout=configuration["dropout"],
    )
    model.load_state_dict(torch.load(settings.model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(test_features, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).numpy().ravel()

    predictions = (probabilities >= 0.5).astype(int)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = calculate_binary_metrics(test_target, probabilities)

    # user_idx é a primeira coluna da matriz de features (ver src/training/data.py)
    user_ids = test_features[:, 0].astype(int)
    ranking_metrics = calculate_ranking_metrics(user_ids, test_target, probabilities, k=10)
    metrics.update(ranking_metrics)

    save_confusion_matrix(test_target, predictions)
    save_roc_curve(test_target, probabilities)
    save_precision_recall_curve(test_target, probabilities)
    save_classification_report(test_target, predictions)

    report_path = REPORTS_DIR / "evaluation_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()