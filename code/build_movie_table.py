from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/ml-32m")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

movies = pd.read_csv(RAW_DIR / "movies.csv")
links = pd.read_csv(RAW_DIR / "links.csv")
tmdb = pd.read_csv(PROCESSED_DIR / "tmdb_movies.csv")

print("MovieLens movies:", len(movies))
print("Links:", len(links))
print("TMDB metadata:", len(tmdb))

# Connect MovieLens movies -> TMDB IDs
movie_table = movies.merge(
    links[["movieId", "tmdbId"]],
    on="movieId",
    how="left"
)

# Connect TMDB metadata
movie_table = movie_table.merge(
    tmdb,
    left_on="tmdbId",
    right_on="tmdb_id",
    how="left"
)

output = PROCESSED_DIR / "movies_enriched.csv"
movie_table.to_csv(output, index=False)

print("\nSaved:", output)
print("Rows:", len(movie_table))
print("\nTMDB coverage:")
print(movie_table["tmdb_id"].notna().mean())

print("\nSample:")
print(movie_table.head())