from pathlib import Path
import json

import pandas as pd


# ============================================
# Configuration
# ============================================

DATA_DIR = Path(
    "data/processed"
)

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# Load FINAL training data
# ============================================

train_path = (
    DATA_DIR /
    "train_next.parquet"
)

if not train_path.exists():

    raise FileNotFoundError(
        f"Could not find: {train_path}"
    )


train = pd.read_parquet(
    train_path
)


print(
    "Training interactions:",
    len(train)
)


# ============================================
# Build user mapping
# ============================================

users = sorted(
    train["userId"]
    .unique()
)

user_to_idx = {
    str(user_id): idx
    for idx, user_id
    in enumerate(users)
}


idx_to_user = {
    str(idx): int(user_id)
    for user_id, idx
    in user_to_idx.items()
}


# ============================================
# Build movie mapping
# ============================================

movies = sorted(
    train["movieId"]
    .unique()
)

movie_to_idx = {
    str(movie_id): idx
    for idx, movie_id
    in enumerate(movies)
}


idx_to_movie = {
    str(idx): int(movie_id)
    for movie_id, idx
    in movie_to_idx.items()
}


# ============================================
# Save mappings
# ============================================

with open(
    MODEL_DIR /
    "user_to_idx.json",
    "w"
) as f:

    json.dump(
        user_to_idx,
        f
    )


with open(
    MODEL_DIR /
    "idx_to_user.json",
    "w"
) as f:

    json.dump(
        idx_to_user,
        f
    )


with open(
    MODEL_DIR /
    "movie_to_idx.json",
    "w"
) as f:

    json.dump(
        movie_to_idx,
        f
    )


with open(
    MODEL_DIR /
    "idx_to_movie.json",
    "w"
) as f:

    json.dump(
        idx_to_movie,
        f
    )


# ============================================
# Statistics
# ============================================

print(
    "\nMapping statistics:"
)

print(
    "Users:",
    len(user_to_idx)
)

print(
    "Movies:",
    len(movie_to_idx)
)

print(
    "First user mapping:",
    list(user_to_idx.items())[:3]
)

print(
    "First movie mapping:",
    list(movie_to_idx.items())[:3]
)


print(
    "\nSaved:"
)

print(
    MODEL_DIR /
    "user_to_idx.json"
)

print(
    MODEL_DIR /
    "idx_to_user.json"
)

print(
    MODEL_DIR /
    "movie_to_idx.json"
)

print(
    MODEL_DIR /
    "idx_to_movie.json"
)


print(
    "\nFinal mapping creation complete."
)