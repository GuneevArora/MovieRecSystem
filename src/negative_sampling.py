from typing import Dict, Set, List

import numpy as np


class NegativeSampler:

    def __init__(
        self,
        positive_by_user: Dict[int, Set[int]],
        movie_counts: Dict[int, int],
        num_movies: int,
        power: float = 0.75,
        seed: int = 42,
    ):
        self.positive_by_user = positive_by_user
        self.num_movies = num_movies

        self.rng = np.random.default_rng(seed)

        # -----------------------------------------
        # Build popularity-weighted probabilities
        # -----------------------------------------

        weights = np.zeros(
            num_movies,
            dtype=np.float64
        )

        for movie_idx, count in movie_counts.items():

            movie_idx = int(movie_idx)

            if 0 <= movie_idx < num_movies:
                weights[movie_idx] = (
                    float(count) ** power
                )

        total = weights.sum()

        if total <= 0:
            raise ValueError(
                "No valid movie sampling weights found."
            )

        self.probabilities = (
            weights / total
        )

    def sample(
        self,
        user_idx: int,
        num_samples: int,
    ) -> List[int]:

        seen = self.positive_by_user.get(
            user_idx,
            set()
        )

        negatives = set()

        # We keep sampling until we have enough
        # movies the user has never interacted with.
        while len(negatives) < num_samples:

            candidates = self.rng.choice(
                self.num_movies,
                size=max(
                    num_samples * 2,
                    10
                ),
                replace=True,
                p=self.probabilities,
            )

            for candidate in candidates:

                candidate = int(candidate)

                # Don't use a movie the user already
                # interacted with.
                if candidate in seen:
                    continue

                negatives.add(candidate)

                if len(negatives) >= num_samples:
                    break

        return list(negatives)