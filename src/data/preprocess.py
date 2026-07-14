"""Data preprocessing module."""

from pathlib import Path
from zipfile import ZipFile
import urllib.request

import pandas as pd

from src.configs.settings import settings

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_dataset() -> Path:
    """Download MovieLens 100K if it does not exist."""

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "ml-100k.zip"
    extract_dir = raw_dir / "ml-100k"

    if not extract_dir.exists():

        if not zip_path.exists():
            print("Downloading MovieLens 100K...")
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)

        print("Extracting dataset...")
        with ZipFile(zip_path) as zip_file:
            zip_file.extractall(raw_dir)

    return extract_dir


def preprocess_data() -> None:
    """Load and preprocess MovieLens dataset."""

    dataset_path = download_dataset()

    ratings = pd.read_csv(
        dataset_path / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    ratings.drop_duplicates(inplace=True)

    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
    )

    ratings["target"] = (ratings["rating"] >= 4).astype(int)

    Path(settings.processed_data_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ratings.to_csv(
        settings.processed_data_path,
        index=False,
    )

    print(f"{len(ratings)} interactions processed.")


if __name__ == "__main__":
    preprocess_data()