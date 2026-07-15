"""Dataset loading utilities for model training."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# As duas primeiras colunas são índices categóricos (user_idx, item_idx),
# consumidos pelas camadas nn.Embedding do MLP (src/models/mlp.py).
# Elas NÃO devem ser escalonadas.
ID_COLUMNS = [
    "user_idx",
    "item_idx",
]

# Features contínuas: escalonadas com StandardScaler.
CONTINUOUS_COLUMNS = [
    "user_interactions",
    "item_popularity",
    "user_mean_rating",
    "item_mean_rating",
    "hour",
    "day_of_week",
    "is_weekend",
    "timestamp_seconds",
]

FEATURE_COLUMNS = ID_COLUMNS + CONTINUOUS_COLUMNS

TARGET_COLUMN = "target"


def load_split_data(
    path: str,
    seed: int,
    scaler_path: str = "models/scaler.pkl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load engineered features and return a reproducible train/test split.

    As colunas contínuas são padronizadas com StandardScaler. O scaler é
    ajustado (fit) apenas no conjunto de treino e aplicado (transform) no
    conjunto de teste, evitando vazamento de dados. O scaler ajustado é
    persistido em disco para reutilização na etapa de avaliação/inferência.
    """

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

    n_id_columns = len(ID_COLUMNS)

    scaler = StandardScaler()
    train_x[:, n_id_columns:] = scaler.fit_transform(train_x[:, n_id_columns:])
    test_x[:, n_id_columns:] = scaler.transform(test_x[:, n_id_columns:])

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return train_x, test_x, train_y, test_y