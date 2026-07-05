from collections import defaultdict
from typing import Dict, Set

import pandas as pd


def build_positive_lookup(
    interactions: pd.DataFrame,
) -> Dict[int, Set[int]]:

    positive_by_user = defaultdict(set)

    for row in interactions.itertuples(index=False):

        user_idx = int(row.user_idx)
        movie_idx = int(row.movie_idx)

        positive_by_user[user_idx].add(
            movie_idx
        )

    return dict(positive_by_user)


def build_movie_counts(
    interactions: pd.DataFrame,
) -> Dict[int, int]:

    counts = (
        interactions["movie_idx"]
        .value_counts()
        .to_dict()
    )

    return {
        int(movie_idx): int(count)
        for movie_idx, count in counts.items()
    }