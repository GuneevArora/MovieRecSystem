from pathlib import Path
import duckdb

RAW_RATINGS = "data/raw/ml-32m/ratings.csv"
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("Creating chronological splits...")

query = f"""
WITH ranked AS (
    SELECT
        userId,
        movieId,
        rating,
        timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY userId
            ORDER BY timestamp
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY userId
        ) AS user_count
    FROM read_csv_auto('{RAW_RATINGS}')
)

SELECT
    userId,
    movieId,
    rating,
    timestamp,
    CASE
        WHEN rn >= user_count - 1 THEN 'test'
        WHEN rn = user_count - 2 THEN 'validation'
        ELSE 'train'
    END AS split
FROM ranked
WHERE user_count >= 5
"""

df = con.execute(query).fetchdf()

print("Rows:", len(df))

train = df[df["split"] == "train"]
validation = df[df["split"] == "validation"]
test = df[df["split"] == "test"]

train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
validation.to_parquet(PROCESSED_DIR / "validation.parquet", index=False)
test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

con.close()