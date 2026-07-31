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

**Jellyfin and Jellyseerr** (`jellyfin-plugin/`): a server plugin that reads each user's watch history, keeps a "Recommended for you" playlist of unwatched films from their own library, and lists the best films the library lacks with a one-click Jellyseerr request. It talks to the API by TMDB id through `POST /recommend/tmdb`. See [jellyfin-plugin/README.md](jellyfin-plugin/README.md).

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The data is not in this repository (see [Data](#data)). Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) into `data/raw/ml-32m/`, put a TMDB API token in `.env` as `TMDB_API_TOKEN`, then run the scripts in `scripts/` to build the splits and features.

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
| `jellyfin-plugin/` | Jellyfin server plugin (C#) |
| `infra/terraform/` | Google Cloud Run infrastructure |
| `.github/workflows/` | CI (tests, lint, build) and deploy |
| `tests/` | Model, metric and API tests |

## Data

The model is trained on two sources, neither of which is redistributed here.

**Ratings: MovieLens 32M.** 32 million ratings from about 200,000 users, collected between January 1995 and October 2023. A rating of 4 stars or higher counts as a positive interaction, which leaves 15.9M positives across 200,143 users and 54,732 films. For each user the most recent positive is the test item, the one before it is the validation item, and the remaining 15.5M are used for training. Splitting by time per user means the model is always evaluated on something that happened after everything it trained on for that user.

**Film metadata: TMDB.** Overview, genres, release year, runtime, vote statistics, director, cast and poster path for each film, fetched through the TMDB API and matched through the MovieLens `links.csv` file. The same API supplies the 1,598 films released since mid-2023 that MovieLens has no ratings for.

**What is in this repository**

| In the repository | Not in the repository |
|---|---|
| Code for every step from raw files to a served model | Raw MovieLens files (`data/raw/`) |
| Index mappings between MovieLens ids and model indices (`models/*.json`) | TMDB responses and the merged metadata table (`data/tmdb/`, `data/processed/`) |
| The fitted feature scaler and genre vocabulary | Trained model weights (`models/*.pt`, `artifacts/`) |
| Measured results (`results/*.json`) | Your TMDB API token (`.env`) |

To reproduce the data, download MovieLens 32M yourself and run the scripts in `scripts/` with your own TMDB token.

## Licence and attribution

The code is released under the [MIT licence](LICENSE). The licence covers the code only, not the datasets:

- **MovieLens 32M** is provided by GroupLens for research use and may not be used for commercial or revenue-bearing purposes without their permission. F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5, 4: 19:1–19:19. This project is not endorsed by the University of Minnesota or the GroupLens Research Group.
- **TMDB** metadata and images are used under the TMDB API terms. This product uses the TMDB API but is not endorsed or certified by TMDB.

A model trained on this data inherits the non-commercial restriction, so this project is a research and portfolio demo, not a commercial service.
