from pathlib import Path
import pickle

import pandas as pd


# ============================================
# Configuration
# ============================================

DATA_DIR = Path(
    "data/processed"
)

INPUT_FILE = (
    DATA_DIR /
    "train_next_indexed.parquet"
)

OUTPUT_FILE = (
    DATA_DIR /
    "user_histories_final.pkl"
)


# ============================================
# Load
# ============================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Could not find: {INPUT_FILE}"
    )


print(
    "Loading final training data..."
)

train = pd.read_parquet(
    INPUT_FILE
)

print(
    "Rows:",
    len(train)
)


# ============================================
# Sort chronologically
# ============================================

print(
    "\nSorting training interactions..."
)

train = train.sort_values(
    [
        "user_idx",
        "timestamp",
    ]
)


# ============================================
# Build history
# ============================================

print(
    "\nBuilding user histories..."
)

user_histories = {}

for row in train.itertuples(
    index=False
):

    user_idx = int(
        row.user_idx
    )

    movie_idx = int(
        row.movie_idx
    )

    if user_idx not in user_histories:

        user_histories[user_idx] = []

    user_histories[user_idx].append(
        movie_idx
    )


# ============================================
# Save
# ============================================

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        user_histories,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================
# Statistics
# ============================================

history_lengths = [
    len(history)
    for history
    in user_histories.values()
]


print(
    "\nUsers:",
    len(user_histories)
)

print(
    "Minimum history:",
    min(history_lengths)
)

print(
    "Maximum history:",
    max(history_lengths)
)

print(
    "Average history:",
    sum(history_lengths)
    / len(history_lengths)
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nFinal user history creation complete."
)