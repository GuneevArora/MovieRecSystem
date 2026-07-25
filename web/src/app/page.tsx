import { ArrowDown, ArrowUpRight } from "lucide-react";
import { Navigation } from "@/components/Navigation";
import { Recommender } from "@/components/Recommender";
import { Reveal } from "@/components/Reveal";
import { metrics, repoUrl, stack, steps } from "@/lib/project";

export default function Home() {
  return (
    <>
      <Navigation />
      <main>
        {/* Hero */}
        <section
          id="home"
          className="relative overflow-hidden px-6 pb-20 pt-40 sm:px-10 lg:px-16 lg:pb-28 lg:pt-52"
        >
          <div className="ambient-orb pointer-events-none absolute -left-32 top-10 h-[34rem] w-[34rem] rounded-full" />
          <div className="ambient-orb ambient-orb-two pointer-events-none absolute -right-40 top-40 h-[38rem] w-[38rem] rounded-full" />
          <div className="relative mx-auto max-w-6xl">
            <p className="mb-6 text-[11px] font-bold uppercase tracking-[.2em] text-black/40">
              Project / Recommender system
            </p>
            <h1 className="max-w-4xl text-5xl leading-[.95] tracking-[-.055em] sm:text-7xl lg:text-8xl">
              Tell it what you love.{" "}
              <span className="display tracking-[-.02em] text-black/45">
                It finds what&apos;s next.
              </span>
            </h1>
            <div className="mt-10 flex flex-col justify-between gap-8 sm:flex-row sm:items-end">
              <p className="max-w-md text-sm leading-6 text-black/55">
                A sequential two-tower model trained on 15.5 million MovieLens
                ratings. It reads the films you pick, in order, and ranks all
                56,330 titles in a few milliseconds.
              </p>
              <a
                href="#demo"
                className="flex w-fit items-center gap-2 rounded-full bg-[#171816] px-5 py-3 text-xs font-semibold text-white transition-transform hover:scale-[1.03]"
              >
                Try the demo <ArrowDown size={14} />
              </a>
            </div>
          </div>
        </section>

        {/* Demo */}
        <section id="demo" className="px-6 py-20 sm:px-10 lg:px-16 lg:py-28">
          <div className="mx-auto max-w-6xl">
            <Reveal>
              <div className="mb-14 flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
                <div>
                  <p className="mb-5 text-[11px] font-bold uppercase tracking-[.2em] text-black/40">
                    01 / Live demo
                  </p>
                  <h2 className="text-4xl tracking-[-.055em] sm:text-6xl">
                    Build a watchlist.
                  </h2>
                </div>
                <p className="max-w-55 text-sm leading-6 text-black/50">
                  Every pick re-ranks the catalog. Tap + on a result to feed it
                  back in.
                </p>
              </div>
            </Reveal>
            <Recommender />
          </div>
        </section>

        {/* How it works */}
        <section id="how" className="px-6 py-20 sm:px-10 lg:px-16 lg:py-28">
          <div className="mx-auto max-w-6xl">
            <Reveal>
              <p className="mb-5 text-[11px] font-bold uppercase tracking-[.2em] text-black/40">
                02 / How it works
              </p>
              <h2 className="max-w-2xl text-4xl tracking-[-.055em] sm:text-6xl">
                Three steps, one dot product.
              </h2>
            </Reveal>
            <div className="mt-14 grid gap-4 md:grid-cols-3">
              {steps.map((step) => (
                <Reveal key={step.number}>
                  <article className="flex min-h-52 flex-col justify-between border-t border-black/15 py-5">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[.16em] text-black/35">
                        {step.number}
                      </p>
                      <h3 className="mt-3 text-xl font-semibold tracking-[-.04em]">
                        {step.title}
                      </h3>
                    </div>
                    <p className="mt-6 text-xs leading-5 text-black/50">{step.body}</p>
                  </article>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* Results */}
        <section id="results" className="px-6 py-20 sm:px-10 lg:px-16 lg:py-28">
          <div className="mx-auto max-w-6xl">
            <Reveal>
              <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
                <div>
                  <p className="mb-5 text-[11px] font-bold uppercase tracking-[.2em] text-black/40">
                    03 / Results
                  </p>
                  <h2 className="text-4xl tracking-[-.055em] sm:text-6xl">
                    Measured, not guessed.
                  </h2>
                </div>
                <p className="max-w-xs text-sm leading-6 text-black/50">
                  Each user&apos;s most recent favourite is hidden, then ranked
                  against the full catalog. 199,899 test users, no sampled
                  shortcuts.
                </p>
              </div>
            </Reveal>
            <Reveal>
              <div className="mt-14 grid grid-cols-2 gap-y-8 border-y border-black/10 py-8 md:grid-cols-4">
                {metrics.map((metric) => (
                  <div key={metric.label}>
                    <p className="text-4xl font-semibold tracking-[-.05em] sm:text-5xl">
                      {metric.value}
                    </p>
                    <p className="mt-2 text-[10px] font-bold uppercase tracking-[.14em] text-black/40">
                      {metric.label}
                    </p>
                  </div>
                ))}
              </div>
            </Reveal>
            <Reveal>
              <div className="mt-8 flex flex-wrap items-center justify-between gap-6">
                <div className="flex flex-wrap gap-2">
                  {stack.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-black/10 px-3 py-1.5 text-[10px] font-semibold text-black/55"
                    >
                      {item}
                    </span>
                  ))}
                </div>
                <a
                  href={repoUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="group flex items-center gap-2 text-xs font-semibold"
                >
                  Read the code
                  <ArrowUpRight
                    size={15}
                    className="transition-transform duration-300 group-hover:-translate-y-1 group-hover:translate-x-1"
                  />
                </a>
              </div>
            </Reveal>
          </div>
        </section>
      </main>

      <footer className="border-t border-black/10 px-6 py-8 sm:px-10 lg:px-16">
        <div className="mx-auto flex max-w-6xl flex-col justify-between gap-3 text-[11px] text-black/40 sm:flex-row">
          <p>Guneev Arora · Movie Recommender</p>
          <p>
            Ratings from MovieLens 32M. Film data and posters from TMDB; this
            product is not endorsed or certified by TMDB.
          </p>
        </div>
      </footer>
    </>
  );
}
