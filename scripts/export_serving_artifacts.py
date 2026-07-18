"""
Bundle everything the API needs into `artifacts/` so the service does not
depend on the training data directory.

    artifacts/model.pt        model config + weights + movie content features
    artifacts/movies.parquet  one row per movie_idx: ids, title, metadata, popularity

Usage
-----
    python -m scripts.export_serving_artifacts
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import load_movie_features, load_sequences

YEAR_PATTERN = re.compile(r"\s*\((\d{4})\)\s*$")


def split_pipe(value) -> list[str]:
    if not isinstance(value, str) or not value or value.startswith("(no genres"):
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def load_poster_paths(tmdb_ids, raw_dir: Path = Path("data/tmdb/raw")) -> list:
    """Poster path from the raw TMDB responses, None where unavailable."""
    posters = []
    for tmdb_id in tmdb_ids:
        path = None
        if pd.notna(tmdb_id):
            raw_file = raw_dir / f"{int(tmdb_id)}.json"
            if raw_file.exists():
                try:
                    path = json.loads(raw_file.read_text(encoding="utf-8")).get("poster_path")
                except (json.JSONDecodeError, OSError):
                    path = None
        posters.append(path or None)
    return posters


def build_movie_table(seqs) -> pd.DataFrame:
    with open("models/idx_to_movie.json") as f:
        idx_to_movie = {int(k): int(v) for k, v in json.load(f).items()}

    enriched = pd.read_csv("data/processed/movies_enriched.csv")
    enriched = enriched.drop_duplicates("movieId").set_index("movieId")

    movie_ids = [idx_to_movie[i] for i in range(seqs.num_movies)]
    rows = enriched.reindex(movie_ids)

    ml_title = rows["title_x"].fillna("").astype(str)
    year_from_title = ml_title.str.extract(YEAR_PATTERN)[0].astype(float)
    clean_ml_title = ml_title.str.replace(YEAR_PATTERN, "", regex=True)

    table = pd.DataFrame(
        {
            "movie_idx": np.arange(seqs.num_movies),
            "movie_id": movie_ids,
            "title": rows["title_y"].where(rows["title_y"].notna(), clean_ml_title).to_numpy(),
            "year": rows["release_year"].fillna(year_from_title).to_numpy(),
            "genres": [split_pipe(v) for v in rows["genres_x"]],
            "tmdb_id": rows["tmdbId"].to_numpy(),
            "poster_path": load_poster_paths(rows["tmdbId"]),
            "overview": rows["overview"].fillna("").to_numpy(),
            "director": rows["director"].fillna("").to_numpy(),
            "cast": [split_pipe(v)[:5] for v in rows["cast"]],
            "vote_average": rows["vote_average"].to_numpy(),
            "runtime": rows["runtime"].to_numpy(),
            "popularity": seqs.item_counts(),
        }
    )
    table["year"] = table["year"].astype("Int64")
    table["tmdb_id"] = table["tmdb_id"].astype("Int64")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/two_tower_seq_best.pt")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    genres, numerical = load_movie_features()
    torch.save(
        {
            "config": checkpoint["config"],
            "model_state_dict": checkpoint["model_state_dict"],
            "genres": torch.from_numpy(genres),
            "numerical": torch.from_numpy(numerical),
            "validation": checkpoint.get("validation"),
        },
        output_dir / "model.pt",
    )

    table = build_movie_table(load_sequences())
    table.to_parquet(output_dir / "movies.parquet", index=False)

    for path in sorted(output_dir.iterdir()):
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
    print(
        f"Movies: {len(table):,}  with overview: {(table.overview != '').mean():.1%}  "
        f"with poster: {table.poster_path.notna().mean():.1%}"
    )


if __name__ == "__main__":
    main()
