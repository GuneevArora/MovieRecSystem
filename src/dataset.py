from pathlib import Path
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):

    def __init__(
        self,
        interactions: pd.DataFrame,
        history_file: str,
        max_history_length: int = 20,
    ):
        self.interactions = (
            interactions.reset_index(
                drop=True
            )
        )

        self.max_history_length = (
            max_history_length
        )

        history_path = Path(
            history_file
        )

        if not history_path.exists():

            raise FileNotFoundError(
                f"History file not found: "
                f"{history_path}"
            )

        with open(
            history_path,
            "rb"
        ) as f:

            self.user_histories = (
                pickle.load(f)
            )

    def __len__(self):
        return len(
            self.interactions
        )

    def __getitem__(
        self,
        index
    ):

        row = self.interactions.iloc[
            index
        ]

        user_idx = int(
            row["user_idx"]
        )

        positive_movie_idx = int(
            row["movie_idx"]
        )

        history = self.user_histories.get(
            user_idx,
            []
        )

        # ------------------------------------
        # Remove the current positive movie
        # from the history input.
        #
        # Otherwise the model would be told
        # directly that the target movie is
        # part of the user's input history.
        # ------------------------------------

        history = [
            movie_idx
            for movie_idx in history
            if movie_idx != positive_movie_idx
        ]

        # ------------------------------------
        # Keep only the most recent history
        # ------------------------------------

        history = history[
            -self.max_history_length:
        ]

        return {
            "user": user_idx,

            "positive_movie":
                positive_movie_idx,

            "history":
                history,
        }


def retrieval_collate_fn(
    batch
):
    """
    Convert a list of examples into padded
    tensors.

    History sequences can have different
    lengths, so they are padded with -1.
    """

    users = torch.tensor(
        [
            item["user"]
            for item in batch
        ],
        dtype=torch.long,
    )

    positive_movies = torch.tensor(
        [
            item["positive_movie"]
            for item in batch
        ],
        dtype=torch.long,
    )

    max_history = max(
        [
            len(item["history"])
            for item in batch
        ],
        default=1,
    )

    max_history = max(
        max_history,
        1,
    )

    history_movies = torch.full(
        (
            len(batch),
            max_history,
        ),
        fill_value=-1,
        dtype=torch.long,
    )

    history_mask = torch.zeros(
        (
            len(batch),
            max_history,
        ),
        dtype=torch.bool,
    )

    for row_idx, item in enumerate(
        batch
    ):

        history = item["history"]

        if not history:
            continue

        length = len(history)

        history_movies[
            row_idx,
            :length
        ] = torch.tensor(
            history,
            dtype=torch.long,
        )

        history_mask[
            row_idx,
            :length
        ] = True

    return (
        users,
        positive_movies,
        history_movies,
        history_mask,
    )