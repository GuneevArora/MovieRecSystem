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


# ============================================
# Load mappings
# ============================================

with open(
    MODEL_DIR /
    "user_to_idx.json"
) as f:

    user_to_idx = json.load(f)


with open(
    MODEL_DIR /
    "movie_to_idx.json"
) as f:

    movie_to_idx = json.load(f)


print(
    "Users in training mapping:",
    len(user_to_idx)
)

print(
    "Movies in training mapping:",
    len(movie_to_idx)
)


# ============================================
# Process one split
# ============================================

def process_split(
    split_name
):

    input_path = (
        DATA_DIR /
        f"{split_name}_next.parquet"
    )

    output_path = (
        DATA_DIR /
        f"{split_name}_next_indexed.parquet"
    )

    print(
        f"\nProcessing {split_name}..."
    )

    # ----------------------------------------
    # Check input
    # ----------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            f"Could not find: {input_path}"
        )

    # ----------------------------------------
    # Load
    # ----------------------------------------

    df = pd.read_parquet(
        input_path
    )

    print(
        "Original rows:",
        len(df)
    )

    # ----------------------------------------
    # Map users
    # ----------------------------------------

    df["user_idx"] = (
        df["userId"]
        .astype(str)
        .map(user_to_idx)
    )

    # ----------------------------------------
    # Map movies
    # ----------------------------------------

    df["movie_idx"] = (
        df["movieId"]
        .astype(str)
        .map(movie_to_idx)
    )

    # ----------------------------------------
    # Count missing mappings
    # ----------------------------------------

    unknown_users = (
        int(
            df["user_idx"]
            .isna()
            .sum()
        )
    )

    unknown_movies = (
        int(
            df["movie_idx"]
            .isna()
            .sum()
        )
    )

    print(
        "Unknown users:",
        unknown_users
    )

    print(
        "Unknown movies:",
        unknown_movies
    )

    # ----------------------------------------
    # Remove users not present in training
    # ----------------------------------------
    #
    # A user embedding cannot be created for
    # someone who never appeared in training.
    #
    # These rows cannot be evaluated by our
    # warm-start recommender.
    # ----------------------------------------

    if unknown_users > 0:

        print(
            f"Removing {unknown_users} rows "
            "with users not present in the "
            "training mapping."
        )

        df = df.dropna(
            subset=["user_idx"]
        ).copy()

    # ----------------------------------------
    # Remove movies not present in training
    # ----------------------------------------
    #
    # A movie without a training mapping
    # cannot use the learned movie ID embedding.
    #
    # We therefore exclude these from this
    # indexed evaluation dataset.
    # ----------------------------------------

    if unknown_movies > 0:

        print(
            f"Removing {unknown_movies} rows "
            "with movies not present in the "
            "training mapping."
        )

        df = df.dropna(
            subset=["movie_idx"]
        ).copy()

    # ----------------------------------------
    # Convert mapped IDs to integers
    # ----------------------------------------

    df["user_idx"] = (
        df["user_idx"]
        .astype("int64")
    )

    df["movie_idx"] = (
        df["movie_idx"]
        .astype("int64")
    )

    # ----------------------------------------
    # Keep useful columns
    # ----------------------------------------

    df = df[
        [
            "userId",
            "movieId",
            "rating",
            "timestamp",
            "user_idx",
            "movie_idx",
        ]
    ]

    # ----------------------------------------
    # Save
    # ----------------------------------------

    df.to_parquet(
        output_path,
        index=False
    )

    # ----------------------------------------
    # Statistics
    # ----------------------------------------

    print(
        "Indexed rows:",
        len(df)
    )

    print(
        "Unique users:",
        df["user_idx"].nunique()
    )

    print(
        "Unique movies:",
        df["movie_idx"].nunique()
    )

    print(
        "Saved:",
        output_path
    )


# ============================================
# Process all final splits
# ============================================

for split in [
    "train",
    "validation",
    "test",
]:

    process_split(
        split
    )


print(
    "\n================================"
)

print(
    "Final split indexing complete."
)

print(
    "================================"
)