from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.negative_sampling import NegativeSampler


class ImplicitMovieDataset(Dataset):

    def __init__(
        self,
        interactions_path,
        num_movies,
        movie_counts,
        num_negatives=4,
        seed=42,
    ):
        df = pd.read_parquet(interactions_path)

        self.users = df["userId"].astype(int).tolist()
        self.movies = df["movieId"].astype(int).tolist()

        # Convert MovieLens IDs to contiguous IDs later.
        self.num_movies = num_movies
        self.num_negatives = num_negatives

        # Build user -> set(movie) interaction mapping.
        positive_by_user = defaultdict(set)

        for user_id, movie_id in zip(self.users, self.movies):
            positive_by_user[user_id].add(movie_id)

        self.positive_by_user = positive_by_user

        self.movie_counts = movie_counts

        self.user_ids = list(self.users)
        self.movie_ids = list(self.movies)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.movie_ids[idx],
        )