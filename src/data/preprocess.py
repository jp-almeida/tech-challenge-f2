"""Data preprocessing module."""
import os
import pandas as pd
import numpy as np
from src.configs.settings import settings

def generate_dummy_data(path: str) -> None:
    """Generate dummy dataset for demonstration."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.random.seed(settings.random_seed)
    
    data = {
        "user_id": np.random.randint(0, 1000, 12000),
        "item_id": np.random.randint(0, 5000, 12000),
        "interaction_time": np.random.rand(12000) * 100,
        "purchased": np.random.randint(0, 2, 12000),
    }
    pd.DataFrame(data).to_csv(path, index=False)

def preprocess_data() -> None:
    """Clean and preprocess raw data."""
    generate_dummy_data(settings.data_path)
    df = pd.read_csv(settings.data_path)
    
    df.fillna(0, inplace=True)
    
    os.makedirs(os.path.dirname(settings.processed_data_path), exist_ok=True)
    df.to_csv(settings.processed_data_path, index=False)
    print("Data preprocessed successfully.")

if __name__ == "__main__":
    preprocess_data()
