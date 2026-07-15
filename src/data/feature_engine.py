"""Feature engineering module."""

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.configs.settings import settings


def create_features() -> None:

    df = pd.read_csv(
        settings.processed_data_path,
        parse_dates=["timestamp"],
    )

    # Codificação de user_id e item_id como índices contíguos (0..N-1).
    # Necessário para as camadas de embedding do MLP (src/models/mlp.py):
    # nn.Embedding exige índices contíguos, e não os IDs originais.
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_idx"] = user_encoder.fit_transform(df["user_id"])
    df["item_idx"] = item_encoder.fit_transform(df["item_id"])

    # Número de interações do usuário
    user_interactions = (
        df.groupby("user_id")
        .size()
        .rename("user_interactions")
    )

    df = df.merge(
        user_interactions,
        on="user_id",
    )

    # Popularidade do item
    item_popularity = (
        df.groupby("item_id")
        .size()
        .rename("item_popularity")
    )

    df = df.merge(
        item_popularity,
        on="item_id",
    )

    # Média de avaliações do usuário
    user_mean_rating = (
        df.groupby("user_id")["rating"]
        .mean()
        .rename("user_mean_rating")
    )

    df = df.merge(
        user_mean_rating,
        on="user_id",
    )

    # Média de avaliações do item
    item_mean_rating = (
        df.groupby("item_id")["rating"]
        .mean()
        .rename("item_mean_rating")
    )

    df = df.merge(
        item_mean_rating,
        on="item_id",
    )

    # Hora do dia
    df["hour"] = df["timestamp"].dt.hour

    # Dia da semana
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Final de semana
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Timestamp em segundos, SEM normalização aqui.
    # Antes: timestamp_norm = timestamp / max(timestamp) (min-max manual).
    # Agora: o escalonamento (StandardScaler) é feito em
    # src/training/data.py, ajustado apenas no conjunto de treino,
    # para evitar vazamento de informação do conjunto de teste.
    df["timestamp_seconds"] = (
        df["timestamp"].astype("int64") // 10**9
    )

    Path(settings.features_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        settings.features_path,
        index=False,
    )

    # Salva o número de usuários/itens únicos. Necessário para dimensionar
    # corretamente as camadas nn.Embedding do MLP em src/training/train.py.
    embedding_config = {
        "num_users": int(df["user_idx"].nunique()),
        "num_items": int(df["item_idx"].nunique()),
    }

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models/embedding_config.json").write_text(
        json.dumps(embedding_config, indent=2),
        encoding="utf-8",
    )

    print("Feature engineering finished.")


if __name__ == "__main__":
    create_features()