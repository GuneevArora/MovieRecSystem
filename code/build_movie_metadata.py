from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = PROJECT_ROOT/"data"/"tmdb" /"raw"
OUTPUT_FILE = PROJECT_ROOT/"data"/"processed"/"tmdb_movies.csv"


def extract_movie(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        movie = json.load(f)

    credits = movie.get("credits", {})

    directors = [
        person["name"]
        for person in credits.get("crew", [])
        if person.get("job") == "Director"
    ]

    cast = [
        person["name"]
        for person in credits.get("cast", [])[:10]
    ]

    genres = [
        genre["name"]
        for genre in movie.get("genres", [])
    ]

    release_date = movie.get("release_date") or ""

    release_year = (
        int(release_date[:4])
        if len(release_date) >= 4
        else None
    )

    return {
        "tmdb_id": movie.get("id"),
        "title": movie.get("title"),
        "overview": movie.get("overview"),
        "release_year": release_year,
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "runtime": movie.get("runtime"),
        "original_language": movie.get("original_language"),
        "genres": "|".join(genres),
        "director": "|".join(directors),
        "cast": "|".join(cast),
    }


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    files = list(CACHE_DIR.glob("*.json"))

    print(f"Found {len(files)} TMDB cached movies.")

    for path in files:
        try:
            rows.append(extract_movie(path))
        except Exception as exc:
            print(f"Failed to parse {path.name}: {exc}")

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved {len(df)} movies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()