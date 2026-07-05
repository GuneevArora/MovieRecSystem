from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================
# Configuration
# ============================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MOVIES_FILE = (
    DATA_DIR /
    "movies_enriched.csv"
)

OUTPUT_FILE = (
    DATA_DIR /
    "movie_text_embeddings.npy"
)

MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 256

RANDOM_SEED = 42


# ============================================
# Load MovieLens -> model movie mapping
# ============================================

mapping_file = (
    MODEL_DIR /
    "movie_to_idx.json"
)

if not mapping_file.exists():

    raise FileNotFoundError(
        f"Could not find mapping file: "
        f"{mapping_file}"
    )


with open(
    mapping_file,
    "r"
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
# Load enriched movie data
# ============================================

if not MOVIES_FILE.exists():

    raise FileNotFoundError(
        f"Could not find movie data: "
        f"{MOVIES_FILE}"
    )


movies = pd.read_csv(
    MOVIES_FILE
)

print(
    "Movies loaded:",
    len(movies)
)


# ============================================
# Validate required columns
# ============================================

required_columns = [
    "movieId",
    "title_x",
    "overview",
]

missing_columns = [
    column
    for column in required_columns
    if column not in movies.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================
# Map MovieLens movie IDs
# to contiguous model IDs
# ============================================

movies["movie_idx"] = (
    movies["movieId"]
    .astype(str)
    .map(movie_to_idx)
)


# Remove movies that aren't in the
# model's movie vocabulary.
movies = movies.dropna(
    subset=["movie_idx"]
).copy()


movies["movie_idx"] = (
    movies["movie_idx"]
    .astype(int)
)


print(
    "Movies after mapping:",
    len(movies)
)


# ============================================
# Check duplicate movie IDs
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
# Sort by movie_idx
# ============================================
#
# This is critical.
#
# The embedding at row 0 must correspond
# to movie_idx 0.
#
# The embedding at row 1 must correspond
# to movie_idx 1.
#
# etc.
# ============================================

movies = (
    movies
    .sort_values("movie_idx")
    .reset_index(drop=True)
)


# ============================================
# Verify perfect alignment
# ============================================

expected_indices = np.arange(
    len(movies),
    dtype=np.int64
)

actual_indices = (
    movies["movie_idx"]
    .to_numpy(
        dtype=np.int64
    )
)


if not np.array_equal(
    expected_indices,
    actual_indices
):

    raise ValueError(
        "Movie rows are not perfectly "
        "aligned with movie_idx."
    )


if len(movies) != num_movies:

    raise ValueError(
        f"Movie feature rows "
        f"({len(movies)}) do not match "
        f"movie mapping size "
        f"({num_movies})."
    )


print(
    "Movie alignment verified."
)


# ============================================
# Build text documents
# ============================================

def clean_text(value):
    """
    Convert a movie field into clean text.
    """

    if pd.isna(value):

        return ""

    return (
        str(value)
        .strip()
    )


documents = []

for row in movies.itertuples(
    index=False
):

    title = clean_text(
        row.title_x
    )

    overview = clean_text(
        row.overview
    )

    # Use both title and overview.
    #
    # We intentionally do not include
    # genres here because genre information
    # is already used separately by the
    # recommendation model.
    text = (
        f"Title: {title}. "
        f"Overview: {overview}"
    )

    documents.append(
        text
    )


# ============================================
# Show examples
# ============================================

print(
    "\nExample documents:"
)

for i in range(
    min(3, len(documents))
):

    print(
        f"\n[{i}]"
    )

    print(
        documents[i]
    )


# ============================================
# Load embedding model
# ============================================

print(
    "\nLoading embedding model:"
)

print(
    MODEL_NAME
)

encoder = SentenceTransformer(
    MODEL_NAME
)


# ============================================
# Generate embeddings
# ============================================

print(
    "\nGenerating movie text embeddings..."
)

embeddings = encoder.encode(
    documents,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


# ============================================
# Convert to float32
# ============================================

embeddings = embeddings.astype(
    np.float32
)


# ============================================
# Validate embedding shape
# ============================================

print(
    "\nEmbedding shape:",
    embeddings.shape
)


if embeddings.shape[0] != num_movies:

    raise ValueError(
        f"Embedding count "
        f"({embeddings.shape[0]}) does not "
        f"match movie count "
        f"({num_movies})."
    )


embedding_dimension = (
    embeddings.shape[1]
)

print(
    "Embedding dimension:",
    embedding_dimension
)


# ============================================
# Save embeddings
# ============================================

np.save(
    OUTPUT_FILE,
    embeddings
)


print(
    "\nSaved text embeddings to:"
)

print(
    OUTPUT_FILE
)


# ============================================
# Reload and verify
# ============================================

loaded_embeddings = np.load(
    OUTPUT_FILE
)


print(
    "\nReloaded embedding shape:",
    loaded_embeddings.shape
)


if not np.array_equal(
    loaded_embeddings,
    embeddings
):

    raise ValueError(
        "Saved embeddings do not match "
        "the original embeddings."
    )


# ============================================
# Final validation
# ============================================

print(
    "\n================================"
)

print(
    "SUCCESS"
)

print(
    "Movie text embeddings generated."
)

print(
    "Movies:",
    embeddings.shape[0]
)

print(
    "Dimensions:",
    embeddings.shape[1]
)

print(
    "Aligned with movie_idx: YES"
)

print(
    "================================"
)