"""Dataset loading utilities for model training."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "user_id",
    "item_id",
    "user_interactions",
    "item_popularity",
    "user_mean_rating",
    "item_mean_rating",
    "hour",
    "day_of_week",
    "is_weekend",
    "timestamp_norm",
]

TARGET_COLUMN = "target"


def load_split_data(
    path: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load engineered features and return a reproducible train/test split."""

    dataframe = pd.read_csv(path)

    missing = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in dataset: {missing}"
        )

    features = dataframe[FEATURE_COLUMNS].astype(np.float32).values
    target = dataframe[TARGET_COLUMN].astype(np.int64).values

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
        stratify=target,
        shuffle=True,
    )

    return train_x, test_x, train_y, test_y