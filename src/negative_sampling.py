import random
from collections import defaultdict

import numpy as np


class NegativeSampler:
    def __init__(
        self,
        positive_by_user,
        movie_counts,
        num_movies,
        power=0.75,
        seed=42,
    ):
        self.positive_by_user = positive_by_user
        self.num_movies = num_movies
        self.rng = random.Random(seed)

        counts = np.zeros(num_movies, dtype=np.float64)

        for movie_idx, count in movie_counts.items():
            counts[movie_idx] = count

        weights = np.power(counts, power)

        # Movies with zero training interactions shouldn't be sampled.
        weights[counts == 0] = 0

        total = weights.sum()

        if total == 0:
            raise ValueError("No valid movie sampling weights found.")

        self.probabilities = weights / total

    def sample(self, user_idx, num_samples):
        seen = self.positive_by_user[user_idx]

        results = []

        while len(results) < num_samples:
            candidates = np.random.choice(
                self.num_movies,
                size=num_samples * 2,
                replace=True,
                p=self.probabilities,
            )

            for candidate in candidates:
                candidate = int(candidate)

                if candidate in seen:
                    continue

                if candidate in results:
                    continue

                results.append(candidate)

                if len(results) == num_samples:
                    break

        return results