"""
In-process recommender used by the API.

The catalog is small (about 55K movies x 256 dims), so retrieval is an exact
dot product against every movie vector rather than an approximate index:
it takes a few milliseconds on CPU and has no recall loss.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model import TwoTowerModel


class Recommender:
    def __init__(self, model: TwoTowerModel, movies: pd.DataFrame, device: str = "cpu"):
        """
        movies: one row per movie_idx, in order, with at least the columns
        movie_id, title, year, genres, popularity.
        """
        self.device = device
        self.model = model.to(device).eval()
        self.movies = movies.reset_index(drop=True)
        self.num_movies = len(self.movies)
        if self.num_movies != model.num_movies:
            raise ValueError(
                f"Movie table has {self.num_movies} rows but the model expects "
                f"{model.num_movies}."
            )

        self.id_to_idx = {
            int(movie_id): idx for idx, movie_id in enumerate(self.movies["movie_id"])
        }
        self._records = self.movies.drop(columns=["movie_idx"], errors="ignore").to_dict(
            "records"
        )
        self._titles_lower = self.movies["title"].fillna("").str.lower().to_numpy()

        popularity = self.movies["popularity"].to_numpy()
        self._popular_order = np.argsort(-popularity, kind="stable")
        self._popularity_rank = np.empty(self.num_movies, dtype=np.int64)
        self._popularity_rank[self._popular_order] = np.arange(self.num_movies)

        with torch.inference_mode():
            self.item_vectors = self.model.item_vectors().contiguous()
            self.item_vectors_normalized = F.normalize(self.item_vectors, dim=-1)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path, device: str = "cpu"):
        artifact_dir = Path(artifact_dir)
        bundle = torch.load(
            artifact_dir / "model.pt", map_location="cpu", weights_only=False
        )
        model = TwoTowerModel(
            genres=bundle["genres"], numerical=bundle["numerical"], **bundle["config"]
        )
        model.load_state_dict(bundle["model_state_dict"])
        movies = pd.read_parquet(artifact_dir / "movies.parquet")
        return cls(model, movies, device=device)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def movie(self, idx: int, score: float | None = None) -> dict:
        record = dict(self._records[idx])
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
        if score is not None:
            record["score"] = float(score)
        return record

    def get(self, movie_id: int) -> dict | None:
        idx = self.id_to_idx.get(int(movie_id))
        return None if idx is None else self.movie(idx)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Case-insensitive title search, most popular matches first."""
        query = query.strip().lower()
        if not query:
            return []
        matches = np.flatnonzero(np.char.find(self._titles_lower.astype(str), query) >= 0)
        if len(matches) == 0:
            return []
        starts = np.char.startswith(self._titles_lower[matches].astype(str), query)
        order = np.lexsort((self._popularity_rank[matches], ~starts))
        return [self.movie(int(i)) for i in matches[order][:limit]]

    def popular(self, k: int = 10, exclude: set[int] | None = None) -> list[dict]:
        exclude = exclude or set()
        out = []
        for idx in self._popular_order:
            if int(idx) in exclude:
                continue
            out.append(self.movie(int(idx)))
            if len(out) == k:
                break
        return out

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def to_indices(self, movie_ids: list[int]) -> tuple[list[int], list[int]]:
        """Map MovieLens ids to model indices. Returns (indices, unknown_ids)."""
        indices, unknown = [], []
        for movie_id in movie_ids:
            idx = self.id_to_idx.get(int(movie_id))
            if idx is None:
                unknown.append(int(movie_id))
            else:
                indices.append(idx)
        return indices, unknown

    @torch.inference_mode()
    def recommend(self, movie_ids: list[int], k: int = 10) -> dict:
        """
        movie_ids: movies the user liked, oldest first.

        Falls back to popularity when none of the ids are known.
        """
        indices, unknown = self.to_indices(movie_ids)
        if not indices:
            return {"items": self.popular(k), "strategy": "popularity", "unknown_ids": unknown}

        items = self.recommend_batch([(indices, k)])[0]
        return {"items": items, "strategy": "two_tower", "unknown_ids": unknown}

    @torch.inference_mode()
    def recommend_batch(self, requests: list[tuple[list[int], int]]) -> list[list[dict]]:
        """
        Score several users in one forward pass.

        requests: (history as model indices, oldest first and non-empty, k).
        Returns the top-k movies for each request, already-seen movies removed.
        """
        max_len = self.model.max_len
        length = min(max_len, max(len(indices) for indices, _ in requests))
        history = torch.full(
            (len(requests), length), self.model.pad_idx, dtype=torch.long
        )
        seen_rows, seen_cols = [], []
        for row, (indices, _) in enumerate(requests):
            recent = indices[-length:]
            history[row, length - len(recent) :] = torch.tensor(recent)
            seen_rows.extend([row] * len(indices))
            seen_cols.extend(indices)

        user_vectors = self.model.encode_users(history.to(self.device), self.item_vectors)
        scores = user_vectors @ self.item_vectors.T
        scores[seen_rows, seen_cols] = float("-inf")

        max_k = min(max(k for _, k in requests), self.num_movies)
        top = torch.topk(scores, max_k, dim=1)
        top_indices, top_scores = top.indices.tolist(), top.values.tolist()

        results = []
        for row, (_, k) in enumerate(requests):
            results.append(
                [
                    self.movie(idx, score)
                    for idx, score in zip(top_indices[row][:k], top_scores[row][:k])
                    if score != float("-inf")
                ]
            )
        return results

    @torch.inference_mode()
    def similar(self, movie_id: int, k: int = 10) -> list[dict] | None:
        """Nearest movies by cosine similarity in the movie-tower space."""
        idx = self.id_to_idx.get(int(movie_id))
        if idx is None:
            return None
        scores = self.item_vectors_normalized @ self.item_vectors_normalized[idx]
        scores[idx] = float("-inf")
        top = torch.topk(scores, min(k, self.num_movies - 1))
        return [
            self.movie(int(i), s) for i, s in zip(top.indices.tolist(), top.values.tolist())
        ]
