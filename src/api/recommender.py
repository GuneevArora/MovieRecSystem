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
    def __init__(
        self,
        model: TwoTowerModel,
        movies: pd.DataFrame,
        device: str = "cpu",
        extra_item_vectors: torch.Tensor | None = None,
    ):
        """
        movies: one row per movie, in index order, with at least the columns
        movie_id, title, year, genres, popularity.

        extra_item_vectors: vectors for cold-start movies that the model has
        no ID embedding for (see src/cold_start.py). Their rows come after
        the model's own movies in `movies`.
        """
        self.device = device
        self.model = model.to(device).eval()
        self.movies = movies.reset_index(drop=True)
        self.num_movies = len(self.movies)
        num_extra = 0 if extra_item_vectors is None else len(extra_item_vectors)
        if self.num_movies != model.num_movies + num_extra:
            raise ValueError(
                f"Movie table has {self.num_movies} rows but the model has "
                f"{model.num_movies} movies plus {num_extra} cold-start vectors."
            )

        self.id_to_idx = {
            int(movie_id): idx for idx, movie_id in enumerate(self.movies["movie_id"])
        }
        # TMDB ids let media servers (Jellyfin, Jellyseerr) talk to the model
        # without knowing MovieLens ids. Where two rows share a TMDB id the
        # first (a rated film, not a cold-start row) wins.
        self.tmdb_to_idx: dict[int, int] = {}
        if "tmdb_id" in self.movies:
            for idx, tmdb_id in enumerate(self.movies["tmdb_id"]):
                if pd.notna(tmdb_id):
                    self.tmdb_to_idx.setdefault(int(tmdb_id), idx)
        self._records = self.movies.drop(columns=["movie_idx"], errors="ignore").to_dict(
            "records"
        )
        self._titles_lower = self.movies["title"].fillna("").str.lower().to_numpy()

        popularity = self.movies["popularity"].to_numpy()
        self._popular_order = np.argsort(-popularity, kind="stable")

        # Search ordering. Cold-start movies have no ratings, so the table can
        # carry a `search_weight` that places them among comparable films.
        weight = (
            self.movies["search_weight"].to_numpy()
            if "search_weight" in self.movies
            else popularity
        )
        self._popularity_rank = np.empty(self.num_movies, dtype=np.int64)
        self._popularity_rank[np.argsort(-weight, kind="stable")] = np.arange(
            self.num_movies
        )

        with torch.inference_mode():
            item_vectors = self.model.item_vectors()
            if num_extra:
                item_vectors = torch.cat(
                    [item_vectors, extra_item_vectors.to(device, item_vectors.dtype)]
                )
            self.item_vectors = item_vectors.contiguous().clone()
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
        return cls(
            model,
            movies,
            device=device,
            extra_item_vectors=bundle.get("extra_item_vectors"),
        )

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

    def tmdb_to_indices(self, tmdb_ids: list[int]) -> tuple[list[int], list[int]]:
        """Map TMDB ids to model indices. Returns (indices, unknown_ids)."""
        indices, unknown = [], []
        for tmdb_id in tmdb_ids:
            idx = self.tmdb_to_idx.get(int(tmdb_id))
            if idx is None:
                unknown.append(int(tmdb_id))
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
    def recommend_batch(self, requests: list[tuple]) -> list[list[dict]]:
        """
        Score several users in one forward pass.

        requests: (history, k) or (history, k, candidates, exclude).
        history is model indices, oldest first and non-empty. candidates, if
        given, restricts the ranking to those indices (for example the films
        in someone's library); exclude removes indices from it. Returns the
        top-k movies for each request, already-seen movies removed.
        """
        requests = [(tuple(request) + (None, None))[:4] for request in requests]
        max_len = self.model.max_len
        length = min(max_len, max(len(request[0]) for request in requests))
        # padding index = one past the last movie (see TwoTowerModel.encode_sequence)
        history = torch.full((len(requests), length), self.num_movies, dtype=torch.long)
        seen_rows, seen_cols = [], []
        for row, (indices, _, _, exclude) in enumerate(requests):
            recent = indices[-length:]
            history[row, length - len(recent) :] = torch.tensor(recent)
            hidden = list(indices) + list(exclude or [])
            seen_rows.extend([row] * len(hidden))
            seen_cols.extend(hidden)

        user_vectors = self.model.encode_users(history.to(self.device), self.item_vectors)
        scores = user_vectors @ self.item_vectors.T
        for row, (_, _, candidates, _) in enumerate(requests):
            if candidates is not None:
                allowed = torch.zeros(self.num_movies, dtype=torch.bool, device=scores.device)
                allowed[torch.tensor(list(candidates), dtype=torch.long)] = True
                scores[row, ~allowed] = float("-inf")
        scores[seen_rows, seen_cols] = float("-inf")

        max_k = min(max(request[1] for request in requests), self.num_movies)
        top = torch.topk(scores, max_k, dim=1)
        top_indices, top_scores = top.indices.tolist(), top.values.tolist()

        results = []
        for row, (_, k, _, _) in enumerate(requests):
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
