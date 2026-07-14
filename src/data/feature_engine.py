"""Feature engineering module."""

from pathlib import Path

import pandas as pd

from src.configs.settings import settings


def create_features() -> None:

    df = pd.read_csv(
        settings.processed_data_path,
        parse_dates=["timestamp"],
    )

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

    # Timestamp normalizado
    df["timestamp_seconds"] = (
        df["timestamp"].astype("int64") // 10**9
    )

    df["timestamp_norm"] = (
        df["timestamp_seconds"]
        - df["timestamp_seconds"].min()
    ) / (
        df["timestamp_seconds"].max()
        - df["timestamp_seconds"].min()
    )

    Path(settings.features_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        settings.features_path,
        index=False,
    )

    print("Feature engineering finished.")


if __name__ == "__main__":
    create_features()