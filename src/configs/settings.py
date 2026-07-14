"""Pydantic settings configuration."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings and environment variables."""
    
    data_path: str = "data/raw/dataset.csv"
    processed_data_path: str = "data/processed/dataset.csv"
    features_path: str = "data/features/dataset.csv"
    model_path: str = "models/model.pt"
    
    mlflow_tracking_uri: str = "http://localhost:5000"
    random_seed: int = 42

    class Config:
        """Pydantic config."""
        env_file = ".env"

settings = Settings()
