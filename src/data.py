"""
Shared data loading for training and evaluation.

Interactions are stored as one flat, chronologically ordered array of movie
indices per user (CSR layout), which keeps everything on the GPU and avoids
per-row pandas access during training.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
CACHE_FILE = DATA_DIR / "sequences.npz"


@dataclass
class Sequences:
    items: np.ndarray        # [num_train_interactions] movie_idx, grouped by user, oldest first
    offsets: np.ndarray      # [num_users + 1] start of each user's slice in `items`
    val_item: np.ndarray     # [num_users] held-out validation movie, -1 if none
    test_item: np.ndarray    # [num_users] held-out test movie, -1 if none
    num_movies: int

    @property
    def num_users(self) -> int:
        return len(self.offsets) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def item_counts(self) -> np.ndarray:
        return np.bincount(self.items, minlength=self.num_movies)


def build_sequences(data_dir: Path = DATA_DIR) -> Sequences:
    train = pd.read_parquet(
        data_dir / "train_next_indexed.parquet",
        columns=["user_idx", "movie_idx", "timestamp"],
    )
    validation = pd.read_parquet(
        data_dir / "validation_next_indexed.parquet",
        columns=["user_idx", "movie_idx"],
    )
    test = pd.read_parquet(
        data_dir / "test_next_indexed.parquet",
        columns=["user_idx", "movie_idx"],
    )
    genres = np.load(data_dir / "movie_genres.npy", mmap_mode="r")
    num_movies = genres.shape[0]
    num_users = int(
        max(train.user_idx.max(), validation.user_idx.max(), test.user_idx.max())
    ) + 1

    train = train.sort_values(["user_idx", "timestamp"], kind="stable")
    counts = np.bincount(train.user_idx.to_numpy(), minlength=num_users)
    offsets = np.zeros(num_users + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    val_item = np.full(num_users, -1, dtype=np.int64)
    val_item[validation.user_idx.to_numpy()] = validation.movie_idx.to_numpy()
    test_item = np.full(num_users, -1, dtype=np.int64)
    test_item[test.user_idx.to_numpy()] = test.movie_idx.to_numpy()

    return Sequences(
        items=train.movie_idx.to_numpy().astype(np.int64),
        offsets=offsets,
        val_item=val_item,
        test_item=test_item,
        num_movies=num_movies,
    )


def load_sequences(use_cache: bool = True) -> Sequences:
    if use_cache and CACHE_FILE.exists():
        z = np.load(CACHE_FILE)
        return Sequences(
            items=z["items"],
            offsets=z["offsets"],
            val_item=z["val_item"],
            test_item=z["test_item"],
            num_movies=int(z["num_movies"]),
        )

    seqs = build_sequences()
    if use_cache:
        np.savez(
            CACHE_FILE,
            items=seqs.items,
            offsets=seqs.offsets,
            val_item=seqs.val_item,
            test_item=seqs.test_item,
            num_movies=seqs.num_movies,
        )
    return seqs


def load_movie_features(data_dir: Path = DATA_DIR):
    genres = np.load(data_dir / "movie_genres.npy").astype(np.float32)
    numerical = np.load(data_dir / "movie_numerical_scaled.npy").astype(np.float32)
    return genres, numerical
