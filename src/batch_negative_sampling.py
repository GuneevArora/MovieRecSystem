from typing import Dict, Set

import numpy as np


class BatchNegativeSampler:

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

        self.rng = np.random.default_rng(
            seed
        )

        # -----------------------------------------
        # Popularity-weighted distribution
        # P(movie) ∝ popularity^0.75
        # -----------------------------------------

        weights = np.zeros(
            num_movies,
            dtype=np.float64
        )

        for movie_idx, count in (
            movie_counts.items()
        ):

            movie_idx = int(movie_idx)

            if 0 <= movie_idx < num_movies:
                weights[movie_idx] = (
                    float(count) ** power
                )

        total = weights.sum()

        if total <= 0:
            raise ValueError(
                "Movie sampling weights are empty."
            )

        self.probabilities = (
            weights / total
        )

    def sample_batch(
        self,
        user_indices,
        num_negatives: int,
    ):
        """
        Sample negatives for an entire batch.

        Returns:
            np.ndarray with shape:
            [batch_size, num_negatives]
        """

        user_indices = np.asarray(
            user_indices,
            dtype=np.int64
        )

        batch_size = len(
            user_indices
        )

        negatives = np.empty(
            (
                batch_size,
                num_negatives
            ),
            dtype=np.int64
        )

        filled = np.zeros(
            batch_size,
            dtype=np.int64
        )

        # Keep filling rows until each user
        # has enough valid negative movies.
        while np.any(
            filled < num_negatives
        ):

            active_rows = np.where(
                filled < num_negatives
            )[0]

            candidates = self.rng.choice(
                self.num_movies,
                size=len(active_rows),
                p=self.probabilities,
            )

            for row_position, candidate in zip(
                active_rows,
                candidates,
            ):

                user_idx = int(
                    user_indices[row_position]
                )

                candidate = int(candidate)

                # User already interacted with
                # this movie.
                if candidate in (
                    self.positive_by_user[
                        user_idx
                    ]
                ):
                    continue

                current_count = filled[
                    row_position
                ]

                # Avoid duplicate negatives
                # for this particular user.
                existing = negatives[
                    row_position,
                    :current_count
                ]

                if candidate in existing:
                    continue

                negatives[
                    row_position,
                    current_count
                ] = candidate

                filled[
                    row_position
                ] += 1

        return negatives