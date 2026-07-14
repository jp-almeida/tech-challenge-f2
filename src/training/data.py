"""Dataset loading utilities for model training."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["user_id", "item_id", "interaction_time_norm"]
TARGET_COLUMN = "purchased"


def load_split_data(path: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load feature data and create a reproducible stratified split."""
    dataframe = pd.read_csv(path)
    features = dataframe[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = dataframe[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return train_test_split(features, target, test_size=0.2, random_state=seed, stratify=target)
