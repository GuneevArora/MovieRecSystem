import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TMDB_API_TOKEN")

if not TOKEN:
    raise RuntimeError("TMDB_API_TOKEN is not set")

movie_id = 603

url = f"https://api.themoviedb.org/3/movie/{movie_id}"

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

response = requests.get(url, headers=headers, timeout=30)

response.raise_for_status()

movie = response.json()

print("Title:", movie.get("title"))
print("Overview:", movie.get("overview"))
print("Genres:", movie.get("genres"))