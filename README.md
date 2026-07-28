# MovieRecSystem

A movie recommender built end to end: a sequential two-tower retrieval model trained on MovieLens 32M, a FastAPI service that ranks the whole catalog in a few milliseconds, and a Next.js demo.

Pick a few films you like, in order, and it ranks every film in the catalog for what to watch next.

## Results

Leave-one-out, next-item evaluation on the held-out test split. Each user's most recent liked film is hidden and ranked against the **full catalog** (54,732 films, no sampled negatives), with already-seen films removed. 199,899 users.

| Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---|---|---|---|
| Popularity baseline | 0.0457 | 0.0232 | 0.0782 | 0.0313 |
| Sequential two-tower | **0.2369** | **0.1345** | **0.3316** | **0.1583** |

That is 5.8× the baseline's NDCG@10. Numbers are written to `results/metrics.json` by `python -m src.evaluate --split test`.

## How it works

**Model** (`src/model.py`)
- *Movie tower:* ID embedding plus an MLP over genres and TMDB numerical features.
- *User tower:* a causal Transformer over the user's last 50 liked films, each represented by the movie tower. There is no user-ID embedding, so it works for anyone with at least one pick.
- *Scoring:* dot product between the two, so the catalog can be scored with one matrix multiply.

**Training** (`src/train.py`): next-film prediction with a full softmax over the catalog, on 15.5M positive interactions (rating ≥ 4) from 200K users. About 37 minutes on one consumer GPU.

**New releases** (`src/cold_start.py`): MovieLens ratings stop in October 2023, so later films have nothing to learn from. A small mapper predicts a film's vector from its description, genres and TMDB stats, then anchors it to its nearest rated films. On rated films hidden from the mapper this recovers about half of full quality (Recall@10 0.106 vs 0.201 with learned vectors; see `results/cold_start_metrics.json`). The demo marks these films as "New".

**Serving** (`src/api`): FastAPI with exact top-k retrieval and dynamic micro-batching, so concurrent requests share a forward pass.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The data is not in this repository. Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) into `data/raw/ml-32m/`, put a TMDB API token in `.env` as `TMDB_API_TOKEN`, then run the scripts in `scripts/` to build the splits and features.

```bash
python -m src.train                           # train
python -m src.evaluate --split test           # evaluate
python -m scripts.export_serving_artifacts    # bundle model + catalog for the API
uvicorn src.api.main:app --port 8000          # API, docs at /docs
cd web && npm install && npm run dev          # demo at http://localhost:3000
python -m pytest tests                        # tests
```

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Data pipeline: splits, TMDB ingest, features, artifact export |
| `src/` | Model, training, evaluation, cold start |
| `src/api/` | FastAPI service |
| `web/` | Next.js demo |
| `infra/terraform/` | Google Cloud Run infrastructure |
| `.github/workflows/` | CI (tests, lint, build) and deploy |
| `tests/` | Model, metric and API tests |

## Data and attribution

- Ratings: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/). F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5, 4. Used for non-commercial research; this project is not endorsed by the University of Minnesota or GroupLens.
- Film metadata and posters: [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.
