"""
Cold-start vectors for films that have no ratings.

The retrieval model places a film mainly through its ID embedding, which is
learned from interactions. Films released after the MovieLens cut-off have
no interactions, so this module learns a mapper

    content (text embedding + genres + TMDB numericals)  ->  movie vector

on the films the model already knows, then applies it to new releases so
they can be searched, picked and recommended. Each prediction is then
anchored to its nearest real catalog films (see `anchor_to_catalog`).

How well this works is measured by hiding 10% of well-rated catalog films
from the mapper, replacing their real vectors with predicted ones, and
re-running the test-split evaluation for users whose held-out film is one
of them.

Usage
-----
    python -m scripts.ingest_tmdb_new_releases
    python -m src.cold_start
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import DATA_DIR, load_movie_features, load_sequences
from src.evaluate import EvalData, evaluate, load_model

TEXT_MODEL = "all-MiniLM-L6-v2"
CATALOG_TEXT_FILE = DATA_DIR / "text_embeddings_catalog.npy"
NEW_RELEASES_FILE = Path("data/tmdb/new_releases.json")
TMDB_RAW_DIR = Path("data/tmdb/raw")

# New films get ids above every MovieLens id so the two can never collide.
COLD_START_ID_OFFSET = 10_000_000

# TMDB genre names that differ from the MovieLens vocabulary the model uses.
TMDB_TO_MOVIELENS_GENRE = {
    "Science Fiction": "Sci-Fi",
    "Family": "Children",
    "Music": "Musical",
}


def movie_text(title: str, overview: str) -> str:
    return f"Title: {title}. Overview: {overview}"


def embed_texts(texts: list[str], device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(TEXT_MODEL, device=device)
    return encoder.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


class ContentMapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mapper(x, y, train_idx, device, epochs=40, seed=0) -> ContentMapper:
    torch.manual_seed(seed)
    mapper = ContentMapper(x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=1e-3, weight_decay=0.01)
    steps = epochs * (len(train_idx) // 512 + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps)

    for _ in range(epochs):
        mapper.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(order), 512):
            batch = order[start : start + 512]
            predicted = mapper(x[batch])
            loss = F.mse_loss(predicted, y[batch]) + (
                1 - F.cosine_similarity(predicted, y[batch], dim=-1)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return mapper.eval()


def anchor_to_catalog(predicted, bank, k: int = 10, temperature: float = 0.05):
    """
    Replace each predicted vector with a weighted average of the real
    vectors of its k nearest catalog films.

    Raw mapper outputs are regression estimates: they shrink toward the mean
    and sit close to one another, so new films end up being neighbours of
    other new films rather than of the films they resemble. Averaging real
    vectors puts them back among films the model learned from ratings. The
    result is rescaled to the neighbours' average length, because vector
    length carries how strongly a film is recommended.
    """
    similarity = F.normalize(predicted, dim=-1) @ F.normalize(bank, dim=-1).T
    top = similarity.topk(k, dim=1)
    weights = torch.softmax(top.values / temperature, dim=1).unsqueeze(-1)
    neighbours = bank[top.indices]                      # [N, k, dim]
    anchored = (weights * neighbours).sum(dim=1)
    target_norm = (weights.squeeze(-1) * neighbours.norm(dim=-1)).sum(dim=1, keepdim=True)
    return F.normalize(anchored, dim=-1) * target_norm


# ----------------------------------------------------------------------
# Catalog and new-release content
# ----------------------------------------------------------------------


def catalog_texts(num_movies: int) -> list[str]:
    with open("models/idx_to_movie.json") as f:
        idx_to_movie = {int(k): int(v) for k, v in json.load(f).items()}
    enriched = pd.read_csv(
        DATA_DIR / "movies_enriched.csv",
        usecols=["movieId", "title_x", "title_y", "overview"],
    ).drop_duplicates("movieId").set_index("movieId")
    rows = enriched.reindex([idx_to_movie[i] for i in range(num_movies)])
    titles = rows["title_y"].where(rows["title_y"].notna(), rows["title_x"]).fillna("")
    overviews = rows["overview"].fillna("")
    return [movie_text(t, o) for t, o in zip(titles, overviews)]


def load_new_releases() -> pd.DataFrame:
    tmdb_ids = json.loads(NEW_RELEASES_FILE.read_text())["tmdb_ids"]
    records = []
    for tmdb_id in tmdb_ids:
        raw = json.loads((TMDB_RAW_DIR / f"{tmdb_id}.json").read_text(encoding="utf-8"))
        credits = raw.get("credits") or {}
        directors = [
            person["name"] for person in credits.get("crew", []) if person.get("job") == "Director"
        ]
        genres = []
        for genre in raw.get("genres", []):
            name = TMDB_TO_MOVIELENS_GENRE.get(genre["name"], genre["name"])
            if name not in genres:
                genres.append(name)
        records.append(
            {
                "movie_id": COLD_START_ID_OFFSET + int(tmdb_id),
                "title": raw.get("title") or "",
                "year": int(raw["release_date"][:4]),
                "genres": genres,
                "tmdb_id": int(tmdb_id),
                "poster_path": raw.get("poster_path"),
                "overview": raw.get("overview") or "",
                "director": directors[0] if directors else "",
                "cast": [person["name"] for person in credits.get("cast", [])[:5]],
                "vote_average": float(raw.get("vote_average") or 0.0),
                "vote_count": float(raw.get("vote_count") or 0.0),
                "runtime": float(raw.get("runtime") or 0.0),
            }
        )
    return pd.DataFrame.from_records(records)


def new_release_features(new: pd.DataFrame, max_catalog_year: float):
    """Genre multi-hot and scaled numericals, in the catalog's feature space."""
    with open("models/genre_to_idx.json") as f:
        genre_to_idx = json.load(f)
    genres = np.zeros((len(new), len(genre_to_idx)), dtype=np.float32)
    for row, names in enumerate(new["genres"]):
        known = [genre_to_idx[name] for name in names if name in genre_to_idx]
        genres[row, known or [genre_to_idx["(no genres listed)"]]] = 1.0

    # The mapper never saw a release year past the catalog's last one, so
    # newer years are clipped instead of extrapolated.
    numerical = np.stack(
        [
            np.minimum(new["year"].to_numpy(dtype=np.float64), max_catalog_year),
            new["vote_average"].to_numpy(dtype=np.float64),
            np.log1p(new["vote_count"].to_numpy(dtype=np.float64)),
            new["runtime"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    scaler = joblib.load("models/movie_feature_scaler.joblib")
    return genres, scaler.transform(numerical).astype(np.float32)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/two_tower_seq_best.pt")
    parser.add_argument("--min-ratings", type=int, default=20,
                        help="Only learn from films with at least this many train ratings.")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = load_sequences()
    data = EvalData(seqs, device)
    model, checkpoint = load_model(Path(args.checkpoint), device)
    max_len = checkpoint["config"]["max_len"]
    with torch.inference_mode():
        item_vectors = model.item_vectors().float().clone()

    # ---------------- content features for the catalog ----------------
    if CATALOG_TEXT_FILE.exists() and len(np.load(CATALOG_TEXT_FILE, mmap_mode="r")) == seqs.num_movies:
        text = np.load(CATALOG_TEXT_FILE)
    else:
        text = embed_texts(catalog_texts(seqs.num_movies), device)
        np.save(CATALOG_TEXT_FILE, text)
    genres, numerical = load_movie_features()
    x = torch.from_numpy(np.concatenate([text, genres, numerical], axis=1)).to(device)

    # ---------------- train / held-out split over films ----------------
    counts = seqs.item_counts()
    eligible = np.flatnonzero(counts >= args.min_ratings)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible)
    num_held = int(len(eligible) * args.holdout)
    held = torch.from_numpy(np.sort(eligible[:num_held])).to(device)
    train_idx = torch.from_numpy(eligible[num_held:]).to(device)
    print(f"Films to learn from: {len(train_idx):,}   held out: {len(held):,}")

    mapper = train_mapper(x, item_vectors, train_idx, device, seed=args.seed)

    # ---------------- evaluation on held-out films ----------------
    # Neighbours are drawn only from films the mapper was trained on, so the
    # held-out films stay hidden.
    bank = item_vectors[train_idx]
    with torch.inference_mode():
        raw_held = mapper(x[held])
        predicted_held = anchor_to_catalog(raw_held, bank)
    cosine = F.cosine_similarity(predicted_held, item_vectors[held], dim=-1)

    test_users = data.eval_users("test", None)
    is_held = torch.zeros(seqs.num_movies, dtype=torch.bool, device=device)
    is_held[held] = True
    users = test_users[is_held[data.test_item[test_users]]]

    def score_with(matrix):
        def score_fn(batch_users, hist, mask):
            return model.encode_users(hist, matrix) @ matrix.T
        return score_fn

    predicted_matrix = item_vectors.clone()
    predicted_matrix[held] = predicted_held
    raw_matrix = item_vectors.clone()
    raw_matrix[held] = raw_held
    mean_matrix = item_vectors.clone()
    mean_matrix[held] = item_vectors[train_idx].mean(dim=0)

    results = {
        "protocol": (
            "10% of films with enough ratings are hidden from the mapper and their "
            "vectors replaced by content-predicted ones; test-split users whose "
            "held-out film is one of them are evaluated with full-catalog ranking. "
            "The retrieval model itself did see these films during training, so "
            "this measures the mapper, not a fully unseen film."
        ),
        "held_out_films": int(len(held)),
        "mean_cosine_to_true_vector": float(cosine.mean()),
        "true_vectors": evaluate(score_with(item_vectors), data, "test", max_len=max_len, users=users),
        "content_predicted_vectors": evaluate(score_with(predicted_matrix), data, "test", max_len=max_len, users=users),
        "mapper_without_anchoring": evaluate(score_with(raw_matrix), data, "test", max_len=max_len, users=users),
        "mean_vector_baseline": evaluate(score_with(mean_matrix), data, "test", max_len=max_len, users=users),
    }
    for name in ("true_vectors", "content_predicted_vectors", "mapper_without_anchoring", "mean_vector_baseline"):
        m = results[name]
        print(f"{name:<28} users={m['users']:>6}  Recall@10={m['recall@10']:.4f}  "
              f"NDCG@10={m['ndcg@10']:.4f}  Recall@50={m['recall@50']:.4f}")
    print(f"Mean cosine to true vector: {results['mean_cosine_to_true_vector']:.3f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/cold_start_metrics.json").write_text(json.dumps(results, indent=2))

    # ---------------- vectors for the new releases ----------------
    if not NEW_RELEASES_FILE.exists():
        print("No new releases file; run scripts.ingest_tmdb_new_releases first.")
        return

    new = load_new_releases()
    max_year = float(np.load(DATA_DIR / "movie_numerical.npy")[:, 0].max())
    new_genres, new_numerical = new_release_features(new, max_year)
    new_text = embed_texts(
        [movie_text(t, o) for t, o in zip(new["title"], new["overview"])], device
    )
    new_x = torch.from_numpy(
        np.concatenate([new_text, new_genres, new_numerical], axis=1)
    ).to(device)
    # For real new releases every well-rated catalog film can be a neighbour.
    all_rated = torch.from_numpy(np.flatnonzero(counts >= args.min_ratings)).to(device)
    with torch.inference_mode():
        new_vectors = anchor_to_catalog(mapper(new_x), item_vectors[all_rated])
        new_vectors = new_vectors.cpu().numpy().astype(np.float32)

    new.to_parquet(DATA_DIR / "cold_start_movies.parquet", index=False)
    np.save(DATA_DIR / "cold_start_vectors.npy", new_vectors)
    print(f"Saved vectors for {len(new):,} new releases "
          f"({int(new.year.min())}-{int(new.year.max())}).")


if __name__ == "__main__":
    main()
