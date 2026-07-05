from pathlib import Path

import duckdb


# ============================================
# Configuration
# ============================================

RAW_RATINGS = (
    "data/raw/ml-32m/ratings.csv"
)

PROCESSED_DIR = Path(
    "data/processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# Output paths
# ============================================

TRAIN_PATH = (
    PROCESSED_DIR /
    "train_next.parquet"
)

VALIDATION_PATH = (
    PROCESSED_DIR /
    "validation_next.parquet"
)

TEST_PATH = (
    PROCESSED_DIR /
    "test_next.parquet"
)


# ============================================
# Connect DuckDB
# ============================================

con = duckdb.connect()


# ============================================
# Explain
# ============================================

print(
    "Creating chronological next-item splits..."
)

print(
    "Positive interaction definition:"
)

print(
    "rating >= 4.0"
)


# ============================================
# Create positive interactions
# ============================================

query = f"""
WITH positives AS (

    SELECT
        userId,
        movieId,
        rating,
        timestamp

    FROM read_csv_auto(
        '{RAW_RATINGS}'
    )

    WHERE rating >= 4.0
),

ranked AS (

    SELECT

        userId,
        movieId,
        rating,
        timestamp,

        ROW_NUMBER() OVER (
            PARTITION BY userId
            ORDER BY timestamp DESC
        ) AS reverse_rank,

        COUNT(*) OVER (
            PARTITION BY userId
        ) AS positive_count

    FROM positives
)

SELECT

    userId,
    movieId,
    rating,
    timestamp,

    CASE

        WHEN reverse_rank = 1
            THEN 'test'

        WHEN reverse_rank = 2
            THEN 'validation'

        ELSE 'train'

    END AS split

FROM ranked

WHERE positive_count >= 3
"""


print(
    "\nRunning split query..."
)


df = con.execute(
    query
).fetchdf()


print(
    "\nTotal positive interactions:",
    len(df)
)


# ============================================
# Split
# ============================================

train = df[
    df["split"] == "train"
].copy()

validation = df[
    df["split"] == "validation"
].copy()

test = df[
    df["split"] == "test"
].copy()


# ============================================
# Save
# ============================================

train.to_parquet(
    TRAIN_PATH,
    index=False
)

validation.to_parquet(
    VALIDATION_PATH,
    index=False
)

test.to_parquet(
    TEST_PATH,
    index=False
)


# ============================================
# Statistics
# ============================================

print(
    "\nTraining interactions:",
    len(train)
)

print(
    "Validation interactions:",
    len(validation)
)

print(
    "Test interactions:",
    len(test)
)


print(
    "\nTraining users:",
    train["userId"].nunique()
)

print(
    "Validation users:",
    validation["userId"].nunique()
)

print(
    "Test users:",
    test["userId"].nunique()
)


# ============================================
# Close
# ============================================

con.close()


print(
    "\nSaved:"
)

print(
    TRAIN_PATH
)

print(
    VALIDATION_PATH
)

print(
    TEST_PATH
)

print(
    "\nSplit creation complete."
)