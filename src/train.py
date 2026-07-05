from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import TwoTowerModel

from src.dataset import (
    RetrievalDataset,
    retrieval_collate_fn,
)

from src.interaction_utils import (
    build_positive_lookup,
    build_movie_counts,
)

from src.batch_negative_sampling import (
    BatchNegativeSampler,
)


# ============================================
# Configuration
# ============================================

DATA_DIR = Path(
    "data/processed"
)

MODEL_DIR = Path(
    "models"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------
# Training configuration
# --------------------------------------------

NUM_NEGATIVES = 4

BATCH_SIZE = 512

EPOCHS = 3

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

MAX_HISTORY_LENGTH = 20

RANDOM_SEED = 42

MODEL_NAME = (
    "two_tower_final"
)

# --------------------------------------------
# Validation configuration
# --------------------------------------------
#
# Validation is used for model selection.
#
# We do NOT use the test set during training.
#
# To keep training practical, validation checks
# a deterministic subset of users.
# --------------------------------------------

VALIDATION_USERS = 5000

VALIDATION_K = 20

# Number of candidate movies evaluated for each
# validation user.
#
# We include:
#   - the held-out validation movie
#   - sampled negatives
#
# This is a ranking sanity check and model-selection
# signal, not the final reported benchmark.
# --------------------------------------------

VALIDATION_NEGATIVES = 199

# --------------------------------------------
# Runtime options
# --------------------------------------------

NUM_WORKERS = 0

PIN_MEMORY = True

# --------------------------------------------
# Reproducibility
# --------------------------------------------

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================
# Device
# ============================================

print(
    "Device:",
    DEVICE
)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================
# Load mappings
# ============================================

user_mapping_path = (
    MODEL_DIR /
    "user_to_idx.json"
)

movie_mapping_path = (
    MODEL_DIR /
    "movie_to_idx.json"
)

if not user_mapping_path.exists():

    raise FileNotFoundError(
        f"Missing mapping: {user_mapping_path}"
    )

if not movie_mapping_path.exists():

    raise FileNotFoundError(
        f"Missing mapping: {movie_mapping_path}"
    )


with open(
    user_mapping_path
) as f:

    user_to_idx = json.load(f)


with open(
    movie_mapping_path
) as f:

    movie_to_idx = json.load(f)


num_users = len(
    user_to_idx
)

num_movies = len(
    movie_to_idx
)


print(
    "Users:",
    num_users
)

print(
    "Movies:",
    num_movies
)


# ============================================
# Load final training interactions
# ============================================

train_path = (
    DATA_DIR /
    "train_next_indexed.parquet"
)

if not train_path.exists():

    raise FileNotFoundError(
        f"Missing final training set: "
        f"{train_path}"
    )


train = pd.read_parquet(
    train_path
)


print(
    "\nTraining interactions:",
    len(train)
)


# ============================================
# Validate training data
# ============================================

required_columns = {
    "userId",
    "movieId",
    "rating",
    "timestamp",
    "user_idx",
    "movie_idx",
}

missing_columns = (
    required_columns -
    set(train.columns)
)

if missing_columns:

    raise ValueError(
        "Training data is missing columns: "
        f"{sorted(missing_columns)}"
    )


if train["user_idx"].isna().any():

    raise ValueError(
        "Training set contains missing user_idx values."
    )


if train["movie_idx"].isna().any():

    raise ValueError(
        "Training set contains missing movie_idx values."
    )


if (
    train["user_idx"].min() < 0
    or
    train["user_idx"].max() >= num_users
):

    raise ValueError(
        "Training user indices are outside "
        "the user mapping range."
    )


if (
    train["movie_idx"].min() < 0
    or
    train["movie_idx"].max() >= num_movies
):

    raise ValueError(
        "Training movie indices are outside "
        "the movie mapping range."
    )


print(
    "Training data validation passed."
)


# ============================================
# Load validation interactions
# ============================================

validation_path = (
    DATA_DIR /
    "validation_next_indexed.parquet"
)

if not validation_path.exists():

    raise FileNotFoundError(
        f"Missing validation set: "
        f"{validation_path}"
    )


validation = pd.read_parquet(
    validation_path
)


print(
    "Validation interactions:",
    len(validation)
)


# ============================================
# Load user histories
# ============================================

history_path = (
    DATA_DIR /
    "user_histories_final.pkl"
)

if not history_path.exists():

    raise FileNotFoundError(
        f"History file not found: "
        f"{history_path}\n"
        "Run build_user_histories first."
    )


print(
    "Loading final user histories..."
)


# ============================================
# Build positive lookup
# ============================================
#
# This is built from the full training set.
#
# It is used by the negative sampler to ensure
# sampled negative movies are not movies the user
# already positively interacted with.
# ============================================

print(
    "\nBuilding positive interaction lookup..."
)

positive_by_user = (
    build_positive_lookup(
        train
    )
)

print(
    "Users with positive interactions:",
    len(positive_by_user)
)


# ============================================
# Build popularity counts
# ============================================

print(
    "Building movie popularity counts..."
)

movie_counts = (
    build_movie_counts(
        train
    )
)

print(
    "Movies with positive interactions:",
    len(movie_counts)
)


# ============================================
# Load movie features
# ============================================
#
# Text embeddings are intentionally NOT loaded.
#
# They belong to the later RAG / explanation
# layer rather than the retrieval model.
# ============================================

genres = torch.tensor(
    np.load(
        DATA_DIR /
        "movie_genres.npy"
    ),
    dtype=torch.float32,
    device=DEVICE,
)

numerical = torch.tensor(
    np.load(
        DATA_DIR /
        "movie_numerical_scaled.npy"
    ),
    dtype=torch.float32,
    device=DEVICE,
)


print(
    "\nGenre features:",
    tuple(genres.shape)
)

print(
    "Numerical features:",
    tuple(numerical.shape)
)


# ============================================
# Validate feature alignment
# ============================================

if genres.ndim != 2:

    raise ValueError(
        "Genre feature matrix must be 2-dimensional."
    )


if numerical.ndim != 2:

    raise ValueError(
        "Numerical feature matrix must be 2-dimensional."
    )


if genres.shape[0] != num_movies:

    raise ValueError(
        "Genre features are not aligned "
        "with the final movie mapping.\n"
        f"Expected {num_movies} movies, "
        f"found {genres.shape[0]}."
    )


if numerical.shape[0] != num_movies:

    raise ValueError(
        "Numerical features are not aligned "
        "with the final movie mapping.\n"
        f"Expected {num_movies} movies, "
        f"found {numerical.shape[0]}."
    )


print(
    "Movie feature alignment passed."
)


# ============================================
# Negative sampler
# ============================================

negative_sampler = (
    BatchNegativeSampler(
        positive_by_user=
            positive_by_user,

        movie_counts=
            movie_counts,

        num_movies=
            num_movies,

        power=0.75,

        seed=RANDOM_SEED,
    )
)


# ============================================
# Dataset
# ============================================

dataset = RetrievalDataset(
    interactions=train,
    history_file=str(
        history_path
    ),
    max_history_length=
        MAX_HISTORY_LENGTH,
)


print(
    "\nTraining examples:",
    len(dataset)
)


if len(dataset) == 0:

    raise ValueError(
        "Training dataset is empty."
    )


# ============================================
# DataLoader
# ============================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        PIN_MEMORY
        and DEVICE == "cuda"
    ),
    collate_fn=
        retrieval_collate_fn,
)


print(
    "Batches per epoch:",
    len(loader)
)


# ============================================
# Model dimensions
# ============================================

num_genres = (
    genres.shape[1]
)

numerical_dim = (
    numerical.shape[1]
)


# ============================================
# Model
# ============================================

model = TwoTowerModel(
    num_users=num_users,
    num_movies=num_movies,
    num_genres=num_genres,
    numerical_dim=numerical_dim,
    embedding_dim=128,
    output_dim=128,
).to(DEVICE)


print(
    "\nModel:"
)

print(
    model
)


# ============================================
# Optimizer
# ============================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================
# Validation helper
# ============================================

def evaluate_validation(
    model,
    validation_df,
    history_file,
    num_users,
    num_movies,
    genres,
    numerical,
    positive_by_user,
    max_history_length,
    max_users,
    num_negatives,
    k,
    seed,
):
    """
    Evaluate the model using the chronological validation
    split.

    Each validation row is one held-out positive movie.

    For each user we rank:

        1 positive validation movie
        +
        sampled negative movies

    The reported metric is Recall@K / Hit@K because
    there is one held-out target per user.

    This is used ONLY for model selection.

    The final test benchmark will use a separate
    evaluation script and the untouched test split.
    """

    model.eval()

    rng = np.random.default_rng(
        seed
    )

    # ----------------------------------------
    # Select deterministic validation users
    # ----------------------------------------

    validation_users = (
        validation_df[
            "user_idx"
        ]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    if len(validation_users) > max_users:

        validation_users = (
            rng.choice(
                validation_users,
                size=max_users,
                replace=False,
            )
        )

        validation_users.sort()

    # ----------------------------------------
    # Candidate movie matrix
    # ----------------------------------------
    #
    # Movie vectors are computed once for the
    # complete movie catalog.
    #
    # 54,838 x 128 is small enough to keep on GPU.
    # ----------------------------------------

    with torch.no_grad():

        all_movie_indices = torch.arange(
            num_movies,
            device=DEVICE,
            dtype=torch.long,
        )

        all_movie_vectors = []

        chunk_size = 4096

        for start in range(
            0,
            num_movies,
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                num_movies,
            )

            movie_chunk = (
                all_movie_indices[
                    start:end
                ]
            )

            movie_vectors = (
                model.encode_movies(
                    movie_chunk,
                    genres[
                        movie_chunk
                    ],
                    numerical[
                        movie_chunk
                    ],
                )
            )

            all_movie_vectors.append(
                movie_vectors
            )

        all_movie_vectors = torch.cat(
            all_movie_vectors,
            dim=0,
        )

    # ----------------------------------------
    # Load histories
    # ----------------------------------------

    import pickle

    with open(
        history_file,
        "rb",
    ) as f:

        histories = pickle.load(f)

    hits_at_k = []

    ndcg_at_k = []

    evaluated = 0

    # ----------------------------------------
    # Evaluate users
    # ----------------------------------------

    for user_idx in validation_users:

        rows = validation_df[
            validation_df["user_idx"]
            == user_idx
        ]

        if len(rows) == 0:

            continue

        target_movie = int(
            rows.iloc[0]["movie_idx"]
        )

        # ------------------------------------
        # User history
        # ------------------------------------

        history = histories.get(
            int(user_idx),
            [],
        )

        history = list(
            history[-max_history_length:]
        )

        history = [
            int(movie_idx)
            for movie_idx in history
            if int(movie_idx) >= 0
        ]

        if len(history) == 0:

            continue

        history_tensor = torch.tensor(
            history,
            dtype=torch.long,
            device=DEVICE,
        )

        history_mask = torch.ones(
            len(history),
            dtype=torch.bool,
            device=DEVICE,
        )

        # ------------------------------------
        # Encode history movies
        # ------------------------------------

        history_movie_vectors = (
            model.encode_movies(
                history_tensor,
                genres[
                    history_tensor
                ],
                numerical[
                    history_tensor
                ],
            )
        )

        # [1, history_length, output_dim]

        history_movie_vectors = (
            history_movie_vectors
            .unsqueeze(0)
        )

        history_mask = (
            history_mask
            .unsqueeze(0)
        )

        user_tensor = torch.tensor(
            [int(user_idx)],
            dtype=torch.long,
            device=DEVICE,
        )

        user_vector = (
            model.encode_users_from_history(
                user_tensor,
                history_movie_vectors,
                history_mask,
            )
        )

        # ------------------------------------
        # Sample candidate negatives
        # ------------------------------------

        sampled_negatives = []

        seen_movies = set(
            positive_by_user.get(
                int(user_idx),
                set(),
            )
        )

        seen_movies.add(
            target_movie
        )

        # ------------------------------------
        # Deterministic candidate sampling
        # ------------------------------------

        attempts = 0

        max_attempts = (
            num_negatives * 20
        )

        while (
            len(sampled_negatives)
            < num_negatives
            and attempts
            < max_attempts
        ):

            candidate = int(
                rng.integers(
                    0,
                    num_movies,
                )
            )

            attempts += 1

            if candidate in seen_movies:

                continue

            sampled_negatives.append(
                candidate
            )

        if len(sampled_negatives) < num_negatives:

            continue

        candidate_movies = (
            [target_movie]
            +
            sampled_negatives
        )

        candidate_tensor = torch.tensor(
            candidate_movies,
            dtype=torch.long,
            device=DEVICE,
        )

        candidate_user = (
            user_vector
            .expand(
                len(candidate_movies),
                -1,
            )
        )

        candidate_scores = (
            candidate_user
            *
            all_movie_vectors[
                candidate_tensor
            ]
        ).sum(
            dim=-1
        )

        # ------------------------------------
        # Rank candidates
        # ------------------------------------

        ranked_indices = torch.argsort(
            candidate_scores,
            descending=True,
        )

        ranked_movie_ids = [
            candidate_movies[
                int(index)
            ]
            for index in ranked_indices
        ]

        # ------------------------------------
        # Hit@K
        # ------------------------------------

        top_k = ranked_movie_ids[
            :k
        ]

        if target_movie in top_k:

            hits_at_k.append(
                1.0
            )

            rank = (
                top_k.index(
                    target_movie
                )
                + 1
            )

            ndcg_at_k.append(
                1.0 /
                np.log2(
                    rank + 1
                )
            )

        else:

            hits_at_k.append(
                0.0
            )

            ndcg_at_k.append(
                0.0
            )

        evaluated += 1

    # ----------------------------------------
    # Aggregate
    # ----------------------------------------

    if evaluated == 0:

        return {
            "users": 0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    return {
        "users": evaluated,
        "recall_at_k": float(
            np.mean(
                hits_at_k
            )
        ),
        "ndcg_at_k": float(
            np.mean(
                ndcg_at_k
            )
        ),
    }


# ============================================
# Best checkpoint tracking
# ============================================

best_recall = -1.0

best_ndcg = -1.0

best_epoch = -1


best_model_path = (
    MODEL_DIR /
    f"{MODEL_NAME}_best.pt"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================
# Training
# ============================================

for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    total_hit_rate = 0.0

    progress = tqdm(
        loader,
        desc=(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        ),
        unit="batch",
    )

    for batch_number, batch in enumerate(
        progress,
        start=1,
    ):

        (
            users,
            positive_movies,
            history_movies,
            history_mask,
        ) = batch

        # ------------------------------------
        # Move tensors to device
        # ------------------------------------

        users = users.to(
            DEVICE,
            dtype=torch.long,
            non_blocking=True,
        )

        positive_movies = (
            positive_movies.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=True,
            )
        )

        history_movies = (
            history_movies.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=True,
            )
        )

        history_mask = (
            history_mask.to(
                DEVICE,
                non_blocking=True,
            )
        )

        # ------------------------------------
        # Sample negatives
        # ------------------------------------

        user_indices = (
            users
            .detach()
            .cpu()
            .numpy()
        )

        negative_movie_indices = (
            negative_sampler.sample_batch(
                user_indices,
                NUM_NEGATIVES,
            )
        )

        negative_movies = torch.tensor(
            negative_movie_indices,
            dtype=torch.long,
            device=DEVICE,
        )

        # ------------------------------------
        # Build history vectors
        # ------------------------------------
        #
        # Padded history entries use -1.
        # Replace them temporarily with 0
        # for feature indexing.
        # ------------------------------------

        safe_history_movies = (
            history_movies.clamp(
                min=0
            )
        )

        batch_size = (
            users.shape[0]
        )

        history_length = (
            history_movies.shape[1]
        )

        history_flat = (
            safe_history_movies
            .reshape(-1)
        )

        history_genres_flat = (
            genres[
                history_flat
            ]
        )

        history_numeric_flat = (
            numerical[
                history_flat
            ]
        )

        # ------------------------------------
        # Encode history movies
        # ------------------------------------

        history_vectors_flat = (
            model.encode_movies(
                history_flat,
                history_genres_flat,
                history_numeric_flat,
            )
        )

        history_vectors = (
            history_vectors_flat.reshape(
                batch_size,
                history_length,
                -1,
            )
        )

        # ------------------------------------
        # Encode users
        # ------------------------------------

        user_vectors = (
            model.encode_users_from_history(
                users,
                history_vectors,
                history_mask,
            )
        )

        # ------------------------------------
        # Encode positive movies
        # ------------------------------------

        positive_vectors = (
            model.encode_movies(
                positive_movies,
                genres[
                    positive_movies
                ],
                numerical[
                    positive_movies
                ],
            )
        )

        positive_scores = (
            user_vectors
            *
            positive_vectors
        ).sum(
            dim=1
        )

        # ------------------------------------
        # Encode negative movies
        # ------------------------------------

        negative_flat = (
            negative_movies
            .reshape(-1)
        )

        negative_vectors = (
            model.encode_movies(
                negative_flat,
                genres[
                    negative_flat
                ],
                numerical[
                    negative_flat
                ],
            )
        )

        # ------------------------------------
        # Repeat user vectors
        # ------------------------------------

        repeated_user_vectors = (
            user_vectors
            .unsqueeze(1)
            .expand(
                -1,
                NUM_NEGATIVES,
                -1,
            )
            .reshape(
                -1,
                user_vectors.shape[-1],
            )
        )

        negative_scores = (
            repeated_user_vectors
            *
            negative_vectors
        ).sum(
            dim=1
        )

        # ------------------------------------
        # Reshape negative scores
        # ------------------------------------

        negative_scores_reshaped = (
            negative_scores.reshape(
                -1,
                NUM_NEGATIVES,
            )
        )

        # ------------------------------------
        # Pairwise BPR loss
        # ------------------------------------

        score_differences = (
            positive_scores.unsqueeze(1)
            -
            negative_scores_reshaped
        )

        loss = -torch.log(
            torch.sigmoid(
                score_differences
            )
            + 1e-8
        ).mean()

        # ------------------------------------
        # Backpropagation
        # ------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        # ------------------------------------
        # Batch hit rate
        # ------------------------------------
        #
        # This means the positive scored above
        # every sampled negative.
        #
        # It is a training diagnostic only.
        # It is NOT the final Recall@K metric.
        # ------------------------------------

        batch_hit_rate = (
            (
                positive_scores.unsqueeze(1)
                >
                negative_scores_reshaped
            )
            .all(dim=1)
            .float()
            .mean()
            .item()
        )

        total_loss += (
            loss.item()
        )

        total_hit_rate += (
            batch_hit_rate
        )

        # ------------------------------------
        # Progress display
        # ------------------------------------

        running_loss = (
            total_loss /
            batch_number
        )

        running_hit_rate = (
            total_hit_rate /
            batch_number
        )

        progress.set_postfix(
            loss=f"{running_loss:.4f}",
            pairwise_hit=f"{running_hit_rate:.4f}",
        )

        # ------------------------------------
        # First batch diagnostics
        # ------------------------------------

        if batch_number == 1:

            history_mean = (
                history_mask
                .float()
                .sum(dim=1)
                .mean()
                .item()
            )

            print(
                "\n--- Batch diagnostics ---"
            )

            print(
                "History length mean:",
                f"{history_mean:.2f}"
            )

            print(
                "Positive score mean:",
                f"{positive_scores.mean().item():.4f}"
            )

            print(
                "Negative score mean:",
                f"{negative_scores.mean().item():.4f}"
            )

            print(
                "Pairwise hit rate:",
                f"{batch_hit_rate:.4f}"
            )

            print(
                "--------------------------\n"
            )

    # ========================================
    # Epoch summary
    # ========================================

    avg_loss = (
        total_loss /
        len(loader)
    )

    avg_hit_rate = (
        total_hit_rate /
        len(loader)
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Average loss: {avg_loss:.4f}"
    )

    print(
        f"Pairwise hit rate: {avg_hit_rate:.4f}"
    )

    # ========================================
    # Validation
    # ========================================

    print(
        "\nRunning validation..."
    )

    validation_metrics = (
        evaluate_validation(
            model=model,
            validation_df=validation,
            history_file=str(
                history_path
            ),
            num_users=num_users,
            num_movies=num_movies,
            genres=genres,
            numerical=numerical,
            positive_by_user=
                positive_by_user,
            max_history_length=
                MAX_HISTORY_LENGTH,
            max_users=
                VALIDATION_USERS,
            num_negatives=
                VALIDATION_NEGATIVES,
            k=VALIDATION_K,
            seed=
                RANDOM_SEED
                + epoch,
        )
    )

    print(
        "Validation users:",
        validation_metrics["users"]
    )

    print(
        f"Validation Recall@{VALIDATION_K}:",
        f"{validation_metrics['recall_at_k']:.6f}"
    )

    print(
        f"Validation NDCG@{VALIDATION_K}:",
        f"{validation_metrics['ndcg_at_k']:.6f}"
    )

    # ========================================
    # Save best checkpoint
    # ========================================

    current_recall = (
        validation_metrics[
            "recall_at_k"
        ]
    )

    current_ndcg = (
        validation_metrics[
            "ndcg_at_k"
        ]
    )

    is_better = (
        current_recall > best_recall
        or
        (
            current_recall == best_recall
            and
            current_ndcg > best_ndcg
        )
    )

    if is_better:

        best_recall = (
            current_recall
        )

        best_ndcg = (
            current_ndcg
        )

        best_epoch = (
            epoch + 1
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "num_users":
                    num_users,

                "num_movies":
                    num_movies,

                "num_genres":
                    num_genres,

                "numerical_dim":
                    numerical_dim,

                "embedding_dim":
                    128,

                "output_dim":
                    128,

                "max_history_length":
                    MAX_HISTORY_LENGTH,

                "model_name":
                    MODEL_NAME,

                "epoch":
                    epoch + 1,

                "train_rows":
                    len(train),

                "validation_users":
                    validation_metrics[
                        "users"
                    ],

                "validation_recall_at_20":
                    current_recall,

                "validation_ndcg_at_20":
                    current_ndcg,
            },
            best_model_path,
        )

        print(
            "\nNew best model saved:"
        )

        print(
            best_model_path
        )


# ============================================
# Save final checkpoint
# ============================================

final_model_path = (
    MODEL_DIR /
    f"{MODEL_NAME}_last.pt"
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "num_users":
            num_users,

        "num_movies":
            num_movies,

        "num_genres":
            num_genres,

        "numerical_dim":
            numerical_dim,

        "embedding_dim":
            128,

        "output_dim":
            128,

        "max_history_length":
            MAX_HISTORY_LENGTH,

        "model_name":
            MODEL_NAME,

        "epochs":
            EPOCHS,

        "train_rows":
            len(train),

        "best_epoch":
            best_epoch,

        "best_validation_recall_at_20":
            best_recall,

        "best_validation_ndcg_at_20":
            best_ndcg,
    },
    final_model_path,
)


# ============================================
# Final summary
# ============================================

print(
    "\n============================================"
)

print(
    "Final training complete."
)

print(
    "============================================"
)

print(
    "Training rows:",
    len(train)
)

print(
    "Users:",
    num_users
)

print(
    "Movies:",
    num_movies
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best Validation Recall@{VALIDATION_K}:",
    f"{best_recall:.6f}"
)

print(
    f"Best Validation NDCG@{VALIDATION_K}:",
    f"{best_ndcg:.6f}"
)

print(
    "Best checkpoint:",
    best_model_path
)

print(
    "Last checkpoint:",
    final_model_path
)

print(
    "============================================"
)