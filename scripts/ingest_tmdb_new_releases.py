"""
Download films released after the MovieLens 32M cut-off (October 2023).

MovieLens has no ratings for these, so they cannot be learned from
interactions. They are added to the catalog as cold-start items whose
vectors are predicted from content (see src/cold_start.py).

Uses TMDB's discover endpoint to list films by release date, then the same
detail + credits fetch and on-disk cache as scripts/ingest_tmdb.py.

Usage
-----
    python -m scripts.ingest_tmdb_new_releases
    python -m scripts.ingest_tmdb_new_releases --since 2023-07-01 --min-votes 100
"""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.ingest_tmdb import (
    REQUEST_DELAY_SECONDS,
    TMDB_BASE_URL,
    fetch_movie,
    load_cached_movie,
    save_cached_movie,
    session,
)

OUTPUT_FILE = Path("data/tmdb/new_releases.json")
LINKS_FILE = Path("data/raw/ml-32m/links.csv")
MAX_PAGES = 500  # TMDB caps discover at 500 pages of 20


def discover(since: str, until: str, min_votes: int) -> list[int]:
    ids: list[int] = []
    page, total_pages = 1, 1
    while page <= min(total_pages, MAX_PAGES):
        response = session.get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "language": "en-US",
                "include_adult": "false",
                "include_video": "false",
                "sort_by": "vote_count.desc",
                "primary_release_date.gte": since,
                "primary_release_date.lte": until,
                "vote_count.gte": min_votes,
                "page": page,
            },
            timeout=30,
        )
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", 2)))
            continue
        response.raise_for_status()
        body = response.json()
        total_pages = body["total_pages"]
        ids.extend(int(movie["id"]) for movie in body["results"])
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return list(dict.fromkeys(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2023-07-01")
    parser.add_argument("--until", default=dt.date.today().isoformat())
    parser.add_argument(
        "--min-votes",
        type=int,
        default=100,
        help="Minimum TMDB vote count; filters out obscure and unreleased titles.",
    )
    args = parser.parse_args()

    known = set(pd.read_csv(LINKS_FILE)["tmdbId"].dropna().astype(int))

    print(f"Discovering films released {args.since} .. {args.until}")
    discovered = discover(args.since, args.until, args.min_votes)
    new_ids = [tmdb_id for tmdb_id in discovered if tmdb_id not in known]
    print(f"Discovered: {len(discovered):,}  not in MovieLens: {len(new_ids):,}")

    kept = []
    for tmdb_id in tqdm(new_ids, desc="Downloading details"):
        movie = load_cached_movie(tmdb_id)
        if movie is None:
            movie = fetch_movie(tmdb_id)
            if movie is None:
                continue
            save_cached_movie(tmdb_id, movie)
            time.sleep(REQUEST_DELAY_SECONDS)
        if movie.get("overview") and movie.get("release_date"):
            kept.append(tmdb_id)

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "since": args.since,
                "until": args.until,
                "min_votes": args.min_votes,
                "tmdb_ids": kept,
            }
        )
    )
    print(f"Saved {len(kept):,} new releases to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
