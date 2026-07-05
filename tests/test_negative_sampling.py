import pandas as pd

from src.interaction_utils import (
    build_positive_lookup,
    build_movie_counts,
)

from src.negative_sampling import (
    NegativeSampler,
)


def test_negative_sampler():

    data = pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1],
        "movie_idx": [1, 2, 3, 2, 4],
    })

    positive_by_user = build_positive_lookup(
        data
    )

    movie_counts = build_movie_counts(
        data
    )

    sampler = NegativeSampler(
        positive_by_user=positive_by_user,
        movie_counts=movie_counts,
        num_movies=5,
        seed=42,
    )

    negatives = sampler.sample(
        user_idx=0,
        num_samples=1,
    )

    for movie in negatives:
        assert movie not in positive_by_user[0]