"""Evaluate the selected MLP in a separate DVC stage."""
import json
from pathlib import Path

import torch

from src.configs.settings import settings
from src.models.model_factory import ModelFactory
from src.training.data import load_split_data
from src.training.metrics import calculate_binary_metrics


def main() -> None:
    """Load the selected model and persist holdout metrics."""
    _, test_features, _, test_target = load_split_data(settings.features_path, settings.random_seed)
    configuration = json.loads(Path("models/model_config.json").read_text(encoding="utf-8"))
    model = ModelFactory.create_model("mlp", input_dim=3, hidden_dim=configuration["hidden_dim"])
    model.load_state_dict(torch.load(settings.model_path, map_location="cpu"))
    model.eval()
    probabilities = model(torch.tensor(test_features, dtype=torch.float32)).detach().numpy().ravel()
    metrics = calculate_binary_metrics(test_target, probabilities)
    report_path = Path("reports/evaluation_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
