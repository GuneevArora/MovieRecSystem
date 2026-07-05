from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd


# ============================================
# Configuration
# ============================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MOVIES_FILE = (
    DATA_DIR /
    "movies_enriched.csv"
)


# ============================================
# Load data
# ============================================

movies = pd.read_csv(
    MOVIES_FILE
)

print(
    "Loaded movies:",
    len(movies)
)

print(
    "Columns:",
    movies.columns.tolist()
)


# ============================================
# Load MovieLens -> contiguous ID mapping
# ============================================

with open(
    MODEL_DIR / "movie_to_idx.json"
) as f:

    movie_to_idx = json.load(f)


num_movies = len(
    movie_to_idx
)

print(
    "Movies in mapping:",
    num_movies
)


# ============================================
# Validate required columns
# ============================================

required_columns = [
    "movieId",
    "genres_x",
    "release_year",
    "vote_average",
    "vote_count",
    "runtime",
]

missing = [
    column
    for column in required_columns
    if column not in movies.columns
]

if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================
# Map MovieLens IDs -> movie_idx
# ============================================

movies["movie_idx"] = (
    movies["movieId"]
    .astype(str)
    .map(movie_to_idx)
)

movies = movies.dropna(
    subset=["movie_idx"]
).copy()

movies["movie_idx"] = (
    movies["movie_idx"]
    .astype(int)
)


# ============================================
# Check for duplicate movie_idx values
# ============================================

duplicate_count = (
    movies["movie_idx"]
    .duplicated()
    .sum()
)

if duplicate_count > 0:

    raise ValueError(
        f"Found {duplicate_count} duplicate "
        "movie_idx values."
    )


# ============================================
# Sort EXACTLY by movie_idx
# ============================================

movies = movies.sort_values(
    "movie_idx"
).reset_index(
    drop=True
)


# ============================================
# Verify exact sequential alignment
# ============================================

expected_indices = np.arange(
    len(movies)
)

actual_indices = (
    movies["movie_idx"]
    .to_numpy()
)

if not np.array_equal(
    actual_indices,
    expected_indices
):

    raise ValueError(
        "movie_idx is not sequential after "
        "sorting. Feature alignment would "
        "be unsafe."
    )


if len(movies) != num_movies:

    raise ValueError(
        f"Feature movie count "
        f"({len(movies)}) does not match "
        f"mapping movie count "
        f"({num_movies})."
    )


print(
    "Movie feature rows:",
    len(movies)
)


# ============================================
# Genre parser
# ============================================

def parse_genres(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):

        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    value = str(value).strip()

    if not value:
        return []

    # Python-list representation:
    # ['Action', 'Drama']
    try:

        parsed = ast.literal_eval(
            value
        )

        if isinstance(parsed, list):

            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except (
        ValueError,
        SyntaxError
    ):
        pass

    # Pipe-separated:
    # Action|Drama|Sci-Fi
    if "|" in value:

        return [
            x.strip()
            for x in value.split("|")
            if x.strip()
        ]

    # Comma-separated:
    # Action, Drama, Sci-Fi
    if "," in value:

        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    return [value]


# ============================================
# Parse genres_x
# ============================================

movies["parsed_genres"] = (
    movies["genres_x"]
    .apply(parse_genres)
)


print(
    "\nExample parsed genres:"
)

for value in (
    movies["parsed_genres"]
    .head(10)
):

    print(value)


# ============================================
# Build genre vocabulary
# ============================================

genre_set = set()

for genre_list in (
    movies["parsed_genres"]
):

    for genre in genre_list:

        genre_set.add(
            genre
        )


genres = sorted(
    genre_set
)

genre_to_idx = {
    genre: idx
    for idx, genre in enumerate(
        genres
    )
}


print(
    "\nNumber of genres:",
    len(genres)
)

print(
    "Genres:",
    genres
)


with open(
    MODEL_DIR / "genre_to_idx.json",
    "w"
) as f:

    json.dump(
        genre_to_idx,
        f,
        indent=2
    )


# ============================================
# Build genre matrix
# ============================================

genre_matrix = np.zeros(
    (
        num_movies,
        len(genres)
    ),
    dtype=np.float32
)


for row_idx, genre_list in enumerate(
    movies["parsed_genres"]
):

    for genre in genre_list:

        if genre in genre_to_idx:

            genre_matrix[
                row_idx,
                genre_to_idx[genre]
            ] = 1.0


# ============================================
# Save genre matrix
# ============================================

genre_path = (
    DATA_DIR /
    "movie_genres.npy"
)

np.save(
    genre_path,
    genre_matrix
)


# ============================================
# Numerical features
# ============================================

movies["release_year"] = pd.to_numeric(
    movies["release_year"],
    errors="coerce"
).fillna(2000)

movies["vote_average"] = pd.to_numeric(
    movies["vote_average"],
    errors="coerce"
).fillna(0)

movies["vote_count"] = pd.to_numeric(
    movies["vote_count"],
    errors="coerce"
).fillna(0)

movies["runtime"] = pd.to_numeric(
    movies["runtime"],
    errors="coerce"
).fillna(0)


movies["log_vote_count"] = np.log1p(
    movies["vote_count"]
)


numerical_features = movies[
    [
        "release_year",
        "vote_average",
        "log_vote_count",
        "runtime",
    ]
].to_numpy(
    dtype=np.float32
)


# ============================================
# Save numerical features
# ============================================

numerical_path = (
    DATA_DIR /
    "movie_numerical.npy"
)

np.save(
    numerical_path,
    numerical_features
)


# ============================================
# Final alignment checks
# ============================================

print(
    "\nFinal feature shapes:"
)

print(
    "Genre matrix:",
    genre_matrix.shape
)

print(
    "Numerical matrix:",
    numerical_features.shape
)


# Every feature row must correspond
# exactly to movie_idx.
print(
    "\nAlignment check:"
)

print(
    "First movie_idx:",
    movies.iloc[0]["movie_idx"]
)

print(
    "Last movie_idx:",
    movies.iloc[-1]["movie_idx"]
)

print(
    "Expected last movie_idx:",
    num_movies - 1
)


if (
    genre_matrix.shape[0]
    != num_movies
):

    raise ValueError(
        "Genre matrix is misaligned."
    )


if (
    numerical_features.shape[0]
    != num_movies
):

    raise ValueError(
        "Numerical feature matrix "
        "is misaligned."
    )


print(
    "\nSUCCESS:"
)

print(
    "Every feature row is aligned "
    "with movie_idx."
)

print(
    "\nSaved:"
)

print(
    genre_path
)

print(
    numerical_path
)