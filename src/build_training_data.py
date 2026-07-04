from pathlib import Path
import json

import pandas as pd
import torch

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

with open(MODEL_DIR / "user_to_idx.json") as f:
    user_to_idx = json.load(f)

with open(MODEL_DIR / "movie_to_idx.json") as f:
    movie_to_idx = json.load(f)


def convert_split(split_name):

    df = pd.read_parquet(
        DATA_DIR / f"{split_name}_implicit.parquet"
    )

    df["user_idx"] = (
        df["userId"]
        .astype(str)
        .map(user_to_idx)
    )

    df["movie_idx"] = (
        df["movieId"]
        .astype(str)
        .map(movie_to_idx)
    )

    # Drop users/movies not present in training vocabulary.
    df = df.dropna(
        subset=["user_idx", "movie_idx"]
    )

    df["user_idx"] = df["user_idx"].astype("int64")
    df["movie_idx"] = df["movie_idx"].astype("int64")

    output = DATA_DIR / f"{split_name}_indexed.parquet"

    df[
        ["user_idx", "movie_idx", "timestamp"]
    ].to_parquet(
        output,
        index=False
    )

    print(split_name)
    print("Rows:", len(df))
    print("Saved:", output)


if __name__ == "__main__":
    for split in [
        "train",
        "validation",
        "test",
    ]:
        convert_split(split)