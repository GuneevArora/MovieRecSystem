import pandas as pd
import numpy as np

from src.interaction_utils import (
    build_positive_lookup,
    build_movie_counts,
)

from src.batch_negative_sampling import (
    BatchNegativeSampler,
)


def test_batch_negative_sampler():

    data = pd.DataFrame({
        "user_idx": [
            0, 0, 0,
            1, 1,
        ],
        "movie_idx": [
            1, 2, 3,
            2, 4,
        ],
    })

    positive_by_user = (
        build_positive_lookup(data)
    )

    movie_counts = (
        build_movie_counts(data)
    )

    sampler = BatchNegativeSampler(
        positive_by_user=
            positive_by_user,
        movie_counts=
            movie_counts,
        num_movies=5,
        seed=42,
    )

    users = np.array([
        0,
        1,
    ])

    negatives = sampler.sample_batch(
        users,
        num_negatives=1,
    )

    assert negatives.shape == (
        2,
        1
    )

    for row_idx, user_idx in enumerate(
        users
    ):

        user_negatives = set(
            negatives[row_idx]
        )

        assert (
            user_negatives
            .isdisjoint(
                positive_by_user[
                    int(user_idx)
                ]
            )
        )