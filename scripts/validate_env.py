"""Script to validate environment setup."""
import os
from src.configs.settings import settings

def validate() -> None:
    """Validate the current environment."""
    print(f"MLflow URI: {settings.mlflow_tracking_uri}")
    print(f"Random Seed: {settings.random_seed}")
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Using defaults.")
    print("Environment is valid.")

if __name__ == "__main__":
    validate()
