import Image from "next/image";
import { Plus } from "lucide-react";
import { type Movie, posterUrl } from "@/lib/api";

// Soft tints in the portfolio palette, used when a film has no poster.
const tints = ["#e9f5b4", "#d6e8f3", "#f1ead9", "#e6e3f1", "#dcefe2"];

export function Poster({
  movie,
  sizes,
  className = "",
  showTitle = false,
}: {
  movie: Movie;
  sizes: string;
  className?: string;
  /** Print the title on the tint when there is no poster (large tiles only). */
  showTitle?: boolean;
}) {
  const url = posterUrl(movie);
  return (
    <div
      className={`relative overflow-hidden ${className}`}
      style={{ backgroundColor: tints[movie.movie_id % tints.length] }}
    >
      {url ? (
        <Image
          src={url}
          alt=""
          fill
          sizes={sizes}
          className="poster-image object-cover"
        />
      ) : showTitle ? (
        <span className="display absolute inset-0 flex items-center justify-center p-3 text-center text-lg leading-tight text-black/45">
          {movie.title}
        </span>
      ) : null}
    </div>
  );
}

export function MovieCard({
  movie,
  rank,
  onAdd,
}: {
  movie: Movie;
  rank: number;
  onAdd: (movie: Movie) => void;
}) {
  return (
    <article className="movie-card group overflow-hidden rounded-sm border border-black/10 bg-white/40">
      <div className="relative">
        <Poster
          movie={movie}
          sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 25vw"
          className="aspect-[2/3]"
          showTitle
        />
        <div className="pointer-events-none absolute inset-0 bg-linear-to-t from-black/70 via-black/0 to-transparent" />
        <span className="absolute left-3 top-3 text-[10px] font-bold uppercase tracking-[.18em] text-white/85 drop-shadow">
          {String(rank).padStart(2, "0")}
          {movie.cold_start && (
            <span
              title="Released after the ratings data ends, so it is placed by its description rather than by viewing history."
              className="ml-2 rounded-full bg-[#d8f45d] px-2 py-0.5 tracking-[.14em] text-[#171816] drop-shadow-none"
            >
              New
            </span>
          )}
        </span>
        <button
          type="button"
          onClick={() => onAdd(movie)}
          aria-label={`Add ${movie.title} to your picks`}
          title="I like this one too"
          className="absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-full bg-white text-black transition-transform duration-500 hover:bg-[#d8f45d] group-hover:rotate-90"
        >
          <Plus size={16} />
        </button>
        <div className="absolute inset-x-3 bottom-3 text-white">
          <h3 className="text-base font-semibold leading-[1.1] tracking-[-.03em]">
            {movie.title}
          </h3>
          <p className="mt-1 text-[10px] font-bold uppercase tracking-[.16em] text-white/65">
            {[movie.year, movie.director].filter(Boolean).join(" · ")}
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 p-3">
        {movie.genres.slice(0, 3).map((genre) => (
          <span
            key={genre}
            className="rounded-full border border-black/10 px-2.5 py-1 text-[10px] font-semibold text-black/55"
          >
            {genre}
          </span>
        ))}
      </div>
    </article>
  );
}
