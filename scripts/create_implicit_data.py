from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

POSITIVE_THRESHOLD = 4.0


def process_split(name: str):
    path = DATA_DIR / f"{name}.parquet"

    df = pd.read_parquet(path)

    print(f"{name}: original rows = {len(df):,}")

    positives = df[df["rating"] >= POSITIVE_THRESHOLD].copy()

    # We don't need the actual rating during implicit training.
    positives = positives[["userId", "movieId", "timestamp"]]

    output = DATA_DIR / f"{name}_implicit.parquet"

    positives.to_parquet(output, index=False)

    print(f"{name}: positive rows = {len(positives):,}")
    print(f"Saved: {output}")

    return positives


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        process_split(split)