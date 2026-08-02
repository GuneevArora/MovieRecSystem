"use client";

import { useEffect, useRef, useState } from "react";
import { Gauge, LoaderCircle, Search, Sparkles, X } from "lucide-react";
import {
  type Movie,
  type Recommendations,
  getMovie,
  recommend,
  runsInBrowser,
  searchMovies,
} from "@/lib/api";
import { getEngine, onModelProgress } from "@/lib/local";
import { catalogSize, presets } from "@/lib/project";
import { MovieCard, Poster } from "@/components/MovieCard";

const MAX_PICKS = 50;

const UNAVAILABLE = runsInBrowser
  ? "The model could not be downloaded. Check your connection and try again."
  : "The recommender API is not reachable right now. If the service was idle it may be waking up, so try again in a few seconds.";

export function Recommender() {
  const [picks, setPicks] = useState<Movie[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Movie[]>([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const [recs, setRecs] = useState<Recommendations | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const searchBox = useRef<HTMLDivElement>(null);

  // Download progress of the in-browser model (stays at 0 when using the API).
  useEffect(() => onModelProgress(setProgress), []);
  const downloading = runsInBrowser && progress > 0 && progress < 1;

  // Start the download as soon as someone shows interest, not on page load.
  function warmUp() {
    if (runsInBrowser) {
      getEngine().catch(() => setError(UNAVAILABLE));
    }
  }

  // Debounced title search.
  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      searchMovies(trimmed, controller.signal)
        .then((items) => {
          setResults(items);
          setError(null);
        })
        .catch((reason) => {
          if (reason.name !== "AbortError") {
            setError(UNAVAILABLE);
          }
        });
    }, 180);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  // Re-rank whenever the picks change.
  useEffect(() => {
    if (picks.length === 0) {
      return;
    }
    const controller = new AbortController();
    recommend(
      picks.map((movie) => movie.movie_id),
      12,
      controller.signal,
    )
      .then((response) => {
        setRecs(response);
        setError(null);
        setLoading(false);
      })
      .catch((reason) => {
        if (reason.name !== "AbortError") {
          setError(UNAVAILABLE);
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [picks]);

  // Close the search dropdown on outside click.
  useEffect(() => {
    function onPointerDown(event: PointerEvent) {
      if (!searchBox.current?.contains(event.target as Node)) {
        setSearchOpen(false);
      }
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, []);

  function updatePicks(next: Movie[]) {
    setPicks(next);
    if (next.length === 0) {
      setRecs(null);
      setLoading(false);
    } else {
      setLoading(true);
    }
  }

  function addPick(movie: Movie) {
    if (picks.some((pick) => pick.movie_id === movie.movie_id)) {
      return;
    }
    updatePicks([...picks, movie].slice(-MAX_PICKS));
    setQuery("");
    setResults([]);
    setSearchOpen(false);
  }

  function removePick(movieId: number) {
    updatePicks(picks.filter((pick) => pick.movie_id !== movieId));
  }

  async function applyPreset(ids: number[]) {
    setLoading(true);
    try {
      updatePicks(await Promise.all(ids.map(getMovie)));
      setError(null);
    } catch {
      setError(UNAVAILABLE);
      setLoading(false);
    }
  }

  const showResults = searchOpen && query.trim().length >= 2 && results.length > 0;

  return (
    <div className="grid gap-12 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-16">
      {/* Picks */}
      <div>
        <p className="mb-4 text-[10px] font-bold uppercase tracking-[.2em] text-black/40">
          Step 1 / Films you love
        </p>

        <div ref={searchBox} className="relative">
          <label className="flex items-center gap-3 rounded-full border border-black/15 bg-white px-4 py-3 transition-colors focus-within:border-black/50">
            <Search size={16} className="shrink-0 text-black/40" />
            <input
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setSearchOpen(true);
              }}
              onFocus={() => {
                setSearchOpen(true);
                warmUp();
              }}
              placeholder="Search 56,330 films"
              aria-label="Search films"
              className="w-full bg-transparent text-sm outline-none placeholder:text-black/35"
            />
          </label>

          {showResults && (
            <ul className="absolute inset-x-0 top-full z-20 mt-2 max-h-80 overflow-y-auto rounded-2xl border border-black/10 bg-white p-1.5 shadow-xl">
              {results.map((movie) => (
                <li key={movie.movie_id}>
                  <button
                    type="button"
                    onClick={() => addPick(movie)}
                    className="flex w-full items-center gap-3 rounded-xl p-2 text-left transition-colors hover:bg-[#f4f4f0]"
                  >
                    <Poster movie={movie} sizes="36px" className="h-12 w-9 shrink-0 rounded-xs" />
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold tracking-[-.02em]">
                        {movie.title}
                      </span>
                      <span className="block truncate text-[11px] text-black/45">
                        {[movie.year, ...movie.genres.slice(0, 2), movie.cold_start ? "New" : null]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {presets.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => applyPreset(preset.ids)}
              className="rounded-full border border-black/10 px-3 py-1.5 text-[10px] font-semibold text-black/55 transition-colors hover:border-black/40 hover:text-black"
            >
              {preset.label}
            </button>
          ))}
        </div>

        <ul className="mt-8 border-t border-black/15">
          {picks.length === 0 && (
            <li className="py-5 text-sm leading-6 text-black/45">
              Nothing picked yet. Search for a film, or start from one of the
              sets above.
            </li>
          )}
          {picks.map((movie, index) => (
            <li
              key={movie.movie_id}
              className="flex items-center gap-3 border-b border-black/10 py-3"
            >
              <span className="w-5 text-[10px] font-bold tracking-[.16em] text-black/35">
                {String(index + 1).padStart(2, "0")}
              </span>
              <Poster movie={movie} sizes="36px" className="h-12 w-9 shrink-0 rounded-xs" />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-semibold tracking-[-.02em]">
                  {movie.title}
                </span>
                <span className="block text-[11px] text-black/45">{movie.year}</span>
              </span>
              <button
                type="button"
                onClick={() => removePick(movie.movie_id)}
                aria-label={`Remove ${movie.title}`}
                className="rounded-full p-2 text-black/35 transition-colors hover:bg-[#f4f4f0] hover:text-black"
              >
                <X size={14} />
              </button>
            </li>
          ))}
        </ul>
        {downloading && picks.length === 0 && (
          <div className="mt-4">
            <ModelProgress fraction={progress} />
          </div>
        )}
        {picks.length > 1 && (
          <p className="mt-3 text-[11px] leading-5 text-black/40">
            Order matters: the model reads these oldest to newest, so later
            picks weigh more.
          </p>
        )}
      </div>

      {/* Recommendations */}
      <div>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <p className="text-[10px] font-bold uppercase tracking-[.2em] text-black/40">
            Step 2 / Your next watch
          </p>
          {recs?.serverMs != null && picks.length > 0 && (
            <p className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[.16em] text-black/40">
              {loading ? (
                <LoaderCircle size={13} className="animate-spin" />
              ) : (
                <Gauge size={13} />
              )}
              {catalogSize.toLocaleString("en-US")} films ranked in{" "}
              {recs.serverMs.toFixed(0)} ms{runsInBrowser ? " in your browser" : ""}
            </p>
          )}
        </div>

        {error && (
          <div className="rounded-sm border border-black/10 bg-[#f4f4f0] p-5 text-sm leading-6 text-black/60">
            {error}
          </div>
        )}

        {!error && picks.length === 0 && (
          <div className="flex min-h-80 flex-col items-center justify-center rounded-sm border border-dashed border-black/15 p-8 text-center">
            <Sparkles size={20} className="text-black/25" />
            <p className="mt-4 max-w-xs text-sm leading-6 text-black/45">
              Pick a few films and recommendations appear here, ranked against
              the whole catalog.
            </p>
          </div>
        )}

        {!error && picks.length > 0 && recs && (
          <div
            className={`grid grid-cols-2 gap-4 transition-opacity duration-300 sm:grid-cols-3 xl:grid-cols-4 ${
              loading ? "opacity-50" : "opacity-100"
            }`}
          >
            {recs.items.map((movie, index) => (
              <MovieCard
                key={movie.movie_id}
                movie={movie}
                rank={index + 1}
                onAdd={addPick}
              />
            ))}
          </div>
        )}

        {!error && picks.length > 0 && !recs && (
          <div className="flex min-h-80 flex-col items-center justify-center gap-4 text-black/35">
            <LoaderCircle size={22} className="animate-spin" />
            {downloading && <ModelProgress fraction={progress} />}
          </div>
        )}
      </div>
    </div>
  );
}

function ModelProgress({ fraction }: { fraction: number }) {
  return (
    <div className="w-full max-w-56" role="status">
      <p className="text-[10px] font-bold uppercase tracking-[.16em] text-black/40">
        Downloading the model · {Math.round(fraction * 100)}%
      </p>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-black/10">
        <div
          className="h-full rounded-full bg-[#171816] transition-[width] duration-200"
          style={{ width: `${fraction * 100}%` }}
        />
      </div>
    </div>
  );
}
