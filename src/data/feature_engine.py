"""Feature engineering module."""
import os
import pandas as pd
from src.configs.settings import settings

def normalize_feature(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalize a column using max scaling."""
    df[f"{col}_norm"] = df[col] / df[col].max()
    return df

def create_features() -> None:
    """Create and save features for the model."""
    df = pd.read_csv(settings.processed_data_path)
    df = normalize_feature(df, "interaction_time")
    
    os.makedirs(os.path.dirname(settings.features_path), exist_ok=True)
    df.to_csv(settings.features_path, index=False)
    print("Features engineered successfully.")

if __name__ == "__main__":
    create_features()
