from pathlib import Path

import numpy as np
import pandas as pd


# ============================================
# Configuration
# ============================================

DATA_DIR = Path("data/processed")

K_VALUES = [10, 20]

EVAL_USERS = 1000

MIN_TRAIN_INTERACTIONS = 5

RANDOM_SEED = 42


# ============================================
# Load development training data
# ============================================

dev_train = pd.read_parquet(
    DATA_DIR /
    "dev_train_indexed.parquet"
)

validation = pd.read_parquet(
    DATA_DIR /
    "validation_indexed.parquet"
)


print(
    "Development training rows:",
    len(dev_train)
)

print(
    "Validation rows:",
    len(validation)
)


# ============================================
# Build popularity ranking
# ============================================

popular_movies = (
    dev_train["movie_idx"]
    .value_counts()
    .index
    .tolist()
)


print(
    "Movies in popularity ranking:",
    len(popular_movies)
)


# ============================================
# Build training history
# ============================================

seen_by_user = {}

train_counts = {}

for row in dev_train.itertuples(
    index=False
):

    user_idx = int(
        row.user_idx
    )

    movie_idx = int(
        row.movie_idx
    )

    if user_idx not in seen_by_user:

        seen_by_user[user_idx] = set()

    seen_by_user[user_idx].add(
        movie_idx
    )

    train_counts[user_idx] = (
        train_counts.get(
            user_idx,
            0
        ) + 1
    )


# ============================================
# Warm users
# ============================================

warm_users = {
    user_idx
    for user_idx, count
    in train_counts.items()
    if count >= MIN_TRAIN_INTERACTIONS
}


print(
    "Warm users:",
    len(warm_users)
)


# ============================================
# Validation positives
# ============================================

validation_by_user = (
    validation
    .groupby("user_idx")["movie_idx"]
    .apply(set)
    .to_dict()
)


validation_users = [
    int(user_idx)
    for user_idx in validation_by_user
    if user_idx in warm_users
]


# ============================================
# Sample same users as model evaluation
# ============================================

rng = np.random.default_rng(
    RANDOM_SEED
)

if len(validation_users) > EVAL_USERS:

    validation_users = rng.choice(
        validation_users,
        size=EVAL_USERS,
        replace=False,
    )

validation_users = [
    int(user_idx)
    for user_idx in validation_users
]


print(
    "Evaluation users:",
    len(validation_users)
)


# ============================================
# Evaluate popularity
# ============================================

def evaluate_popularity(k):

    recall_total = 0.0
    ndcg_total = 0.0

    evaluated_users = 0

    for user_idx in validation_users:

        relevant = (
            validation_by_user.get(
                user_idx,
                set()
            )
        )

        if not relevant:
            continue

        seen = seen_by_user.get(
            user_idx,
            set()
        )

        recommendations = []

        for movie_idx in popular_movies:

            if movie_idx in seen:
                continue

            recommendations.append(
                movie_idx
            )

            if len(recommendations) == k:
                break

        # --------------------------------
        # Hits
        # --------------------------------

        hits = [
            int(
                movie_idx in relevant
            )
            for movie_idx in recommendations
        ]

        num_hits = sum(hits)

        num_relevant = len(
            relevant
        )

        # --------------------------------
        # Recall
        # --------------------------------

        recall = (
            num_hits /
            num_relevant
        )

        # --------------------------------
        # DCG
        # --------------------------------

        dcg = 0.0

        for rank, hit in enumerate(
            hits,
            start=1
        ):

            if hit:

                dcg += (
                    1.0 /
                    np.log2(rank + 1)
                )

        # --------------------------------
        # IDCG
        # --------------------------------

        ideal_hits = min(
            num_relevant,
            k
        )

        idcg = sum(
            1.0 /
            np.log2(rank + 1)
            for rank in range(
                1,
                ideal_hits + 1
            )
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

        recall_total += recall
        ndcg_total += ndcg

        evaluated_users += 1

    if evaluated_users == 0:

        raise ValueError(
            "No warm users could be evaluated."
        )

    return (
        recall_total /
        evaluated_users,

        ndcg_total /
        evaluated_users,
    )


# ============================================
# Results
# ============================================

print(
    "\nWarm-start popularity results:"
)

for k in K_VALUES:

    recall, ndcg = evaluate_popularity(
        k
    )

    print(
        f"Recall@{k}: "
        f"{recall:.6f}"
    )

    print(
        f"NDCG@{k}: "
        f"{ndcg:.6f}"
    )


print(
    "\nBaseline evaluation complete."
)