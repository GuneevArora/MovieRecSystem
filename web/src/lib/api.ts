/**
 * Data access for the demo.
 *
 * By default the model runs in the visitor's browser (see engine.ts), so the
 * site is plain static files. Set NEXT_PUBLIC_API_URL to use the FastAPI
 * service in src/api instead; both return the same shapes.
 */

import type { Engine } from "@/lib/engine";
import { getEngine } from "@/lib/local";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

/** True when ranking happens in the browser rather than on a server. */
export const runsInBrowser = API_URL === "";

export type Movie = {
  movie_id: number;
  title: string | null;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  poster_path: string | null;
  director: string | null;
  score: number | null;
  /** True for films released after the ratings data ends; placed by content. */
  cold_start?: boolean;
};

export type Recommendations = {
  items: Movie[];
  /** Time spent ranking, in milliseconds: in the browser, or as reported by the API. */
  serverMs: number | null;
};

// ---------------------------------------------------------------------
// In the browser
// ---------------------------------------------------------------------

function toMovie(engine: Engine, index: number, score: number | null = null): Movie {
  const c = engine.catalog;
  return {
    movie_id: c.movie_id[index],
    title: c.title[index],
    year: c.year[index],
    genres: c.genres[index].map((genre) => c.genre_names[genre]),
    tmdb_id: c.tmdb_id[index],
    poster_path: c.poster_path[index],
    director: c.director[index],
    score,
    cold_start: c.cold_start[index] === 1,
  };
}

// ---------------------------------------------------------------------
// Through the API
// ---------------------------------------------------------------------

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    throw new Error(`API responded with ${response.status}`);
  }
  return response;
}

// ---------------------------------------------------------------------

export async function searchMovies(
  query: string,
  signal?: AbortSignal,
): Promise<Movie[]> {
  if (runsInBrowser) {
    const engine = await getEngine();
    return engine.search(query, 8).map((index) => toMovie(engine, index));
  }
  const params = new URLSearchParams({ q: query, limit: "8" });
  const response = await request(`/movies/search?${params}`, { signal });
  return (await response.json()).items;
}

export async function getMovie(movieId: number): Promise<Movie> {
  if (runsInBrowser) {
    const engine = await getEngine();
    const index = engine.indexOf(movieId);
    if (index === undefined) {
      throw new Error(`Unknown film ${movieId}`);
    }
    return toMovie(engine, index);
  }
  const response = await request(`/movies/${movieId}`);
  return response.json();
}

export async function recommend(
  movieIds: number[],
  k = 12,
  signal?: AbortSignal,
): Promise<Recommendations> {
  if (runsInBrowser) {
    const engine = await getEngine();
    const history = movieIds
      .map((id) => engine.indexOf(id))
      .filter((index): index is number => index !== undefined);
    const started = performance.now();
    const top = engine.recommend(history, k);
    const serverMs = performance.now() - started;
    return {
      items: top.map((entry) => toMovie(engine, entry.index, entry.score)),
      serverMs,
    };
  }
  const response = await request("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ movie_ids: movieIds, k }),
    signal,
  });
  const serverMs = Number(response.headers.get("X-Process-Time-Ms"));
  return {
    items: (await response.json()).items,
    serverMs: Number.isFinite(serverMs) && serverMs > 0 ? serverMs : null,
  };
}

export function posterUrl(movie: Movie, size: "w185" | "w342" = "w342") {
  return movie.poster_path
    ? `https://image.tmdb.org/t/p/${size}${movie.poster_path}`
    : null;
}
