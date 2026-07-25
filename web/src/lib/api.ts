export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export type Movie = {
  movie_id: number;
  title: string | null;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  poster_path: string | null;
  overview: string | null;
  director: string | null;
  cast: string[];
  vote_average: number | null;
  runtime: number | null;
  popularity: number | null;
  score: number | null;
  /** True for films released after the ratings data ends; placed by content. */
  cold_start?: boolean;
};

export type Recommendations = {
  items: Movie[];
  strategy: string;
  unknown_ids: number[];
  /** Server-side processing time reported by the API, in milliseconds. */
  serverMs: number | null;
};

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    throw new Error(`API responded with ${response.status}`);
  }
  return response;
}

export async function searchMovies(
  query: string,
  signal?: AbortSignal,
): Promise<Movie[]> {
  const params = new URLSearchParams({ q: query, limit: "8" });
  const response = await request(`/movies/search?${params}`, { signal });
  return (await response.json()).items;
}

export async function getMovie(movieId: number): Promise<Movie> {
  const response = await request(`/movies/${movieId}`);
  return response.json();
}

export async function recommend(
  movieIds: number[],
  k = 12,
  signal?: AbortSignal,
): Promise<Recommendations> {
  const response = await request("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ movie_ids: movieIds, k }),
    signal,
  });
  const serverMs = Number(response.headers.get("X-Process-Time-Ms"));
  return {
    ...(await response.json()),
    serverMs: Number.isFinite(serverMs) && serverMs > 0 ? serverMs : null,
  };
}

export function posterUrl(movie: Movie, size: "w185" | "w342" = "w342") {
  return movie.poster_path
    ? `https://image.tmdb.org/t/p/${size}${movie.poster_path}`
    : null;
}
