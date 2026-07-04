from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

train = pd.read_parquet(
    DATA_DIR / "train_implicit.parquet"
)

users = sorted(train["userId"].unique())
movies = sorted(train["movieId"].unique())

user_to_idx = {
    str(user_id): idx
    for idx, user_id in enumerate(users)
}

movie_to_idx = {
    str(movie_id): idx
    for idx, movie_id in enumerate(movies)
}

idx_to_user = {
    str(idx): int(user_id)
    for user_id, idx in user_to_idx.items()
}

idx_to_movie = {
    str(idx): int(movie_id)
    for movie_id, idx in movie_to_idx.items()
}


with open(MODEL_DIR / "user_to_idx.json", "w") as f:
    json.dump(user_to_idx, f)

with open(MODEL_DIR / "movie_to_idx.json", "w") as f:
    json.dump(movie_to_idx, f)

with open(MODEL_DIR / "idx_to_user.json", "w") as f:
    json.dump(idx_to_user, f)

with open(MODEL_DIR / "idx_to_movie.json", "w") as f:
    json.dump(idx_to_movie, f)

print("Users:", len(user_to_idx))
print("Movies:", len(movie_to_idx))