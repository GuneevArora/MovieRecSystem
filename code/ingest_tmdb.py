from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm



PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINKS_FILE = PROJECT_ROOT/"data"/"raw"/"ml-32m"/"links.csv"
CACHE_DIR = PROJECT_ROOT/"data"/"tmdb"/"raw"
ERROR_FILE = PROJECT_ROOT/"data"/"tmdb"/"errors.jsonl"

TMDB_BASE_URL = "https://api.themoviedb.org/3"


MAX_MOVIES = None

# How many times to retry a failed request.
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 0.05

load_dotenv()

TMDB_TOKEN = os.getenv("TMDB_API_TOKEN")

if not TMDB_TOKEN:
    raise RuntimeError(
        "TMDB_API_TOKEN is missing. "
        "Add it to your .env file."
    )

CACHE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)


HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "Accept": "application/json",
}


session = requests.Session()
session.headers.update(HEADERS)


def cache_path(tmdb_id: int) -> Path:
    """Return the cache filename for a TMDB movie."""
    return CACHE_DIR / f"{tmdb_id}.json"


def load_cached_movie(tmdb_id: int) -> dict[str, Any] | None:
    """Load a previously downloaded movie from disk."""
    path = cache_path(tmdb_id)

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: invalid cache file: {path}")
        return None


def save_cached_movie(tmdb_id: int, data: dict[str, Any]) -> None:
    """Save TMDB response to disk."""
    path = cache_path(tmdb_id)

    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def record_error(tmdb_id: int, error: str) -> None:
    """Append a failed movie to the error log."""
    record = {
        "tmdb_id": tmdb_id,
        "error": error,
        "timestamp": time.time(),
    }

    with ERROR_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")



def fetch_movie(tmdb_id: int) -> dict[str, Any] | None:
    """
    Fetch movie details from TMDB.

    We request credits in the same call using append_to_response
    so that we don't need a separate /credits request.
    """

    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"

    params = {
        "language": "en-US",
        "append_to_response": "credits",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url,
                params=params,
                timeout=30,
            )

            # Rate limit response.
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    wait_seconds = float(retry_after)
                else:
                    wait_seconds = 2 ** attempt

                print(
                    f"\nRate limited for TMDB {tmdb_id}. "
                    f"Waiting {wait_seconds:.1f}s..."
                )

                time.sleep(wait_seconds)
                continue

            # Movie does not exist / bad TMDB ID.
            if response.status_code == 404:
                record_error(tmdb_id, "Movie not found")
                return None

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                record_error(
                    tmdb_id,
                    f"Request failed after retries: {exc}",
                )
                return None

            wait_seconds = 2 ** attempt

            print(
                f"\nRequest failed for TMDB {tmdb_id}: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    return None



def main() -> None:
    if not LINKS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {LINKS_FILE}"
        )

    print("Loading MovieLens links.csv...")

    links = pd.read_csv(LINKS_FILE)


    links = links.dropna(subset=["tmdbId"])

    links["tmdbId"] = links["tmdbId"].astype(int)

    tmdb_ids = links["tmdbId"].drop_duplicates().tolist()

    if MAX_MOVIES is not None:
        tmdb_ids = tmdb_ids[:MAX_MOVIES]

    print(f"Movies to process: {len(tmdb_ids)}")

    cached_count = 0
    downloaded_count = 0
    failed_count = 0

    for tmdb_id in tqdm(tmdb_ids, desc="Downloading TMDB data"):

        cached = load_cached_movie(tmdb_id)

        if cached is not None:
            cached_count += 1
            continue

        movie = fetch_movie(tmdb_id)

        if movie is None:
            failed_count += 1
            continue

        save_cached_movie(tmdb_id, movie)

        downloaded_count += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nDone.")
    print(f"Already cached: {cached_count}")
    print(f"Downloaded:     {downloaded_count}")
    print(f"Failed:         {failed_count}")
    print(f"Cache location: {CACHE_DIR}")
    print(f"Error log:      {ERROR_FILE}")


if __name__ == "__main__":
    main()