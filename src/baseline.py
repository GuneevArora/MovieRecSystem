from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

train = pd.read_parquet(DATA_DIR / "train.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

# Number of ratings per movie
popular = (
    train.groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        average_rating=("rating", "mean")
    )
    .sort_values(
        ["rating_count", "average_rating"],
        ascending=False
    )
)

popular_movies = popular.index.tolist()

print("Most popular movies:")
print(popular.head(20))

# Save ranking
popular.reset_index().to_csv(
    DATA_DIR / "popularity_baseline.csv",
    index=False
)