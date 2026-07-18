"""
FastAPI service for the movie recommender.

Run locally
-----------
    python -m scripts.export_serving_artifacts
    uvicorn src.api.main:app --port 8000

Environment
-----------
    ARTIFACT_DIR      directory with model.pt and movies.parquet (default: artifacts)
    DEVICE            cpu or cuda (default: cpu)
    ALLOWED_ORIGINS   comma-separated CORS origins (default: http://localhost:3000)
    TORCH_THREADS     intra-op threads for CPU inference (default: torch's choice)
    MAX_BATCH_SIZE    largest inference batch (default: 64)
"""

import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.batching import MicroBatcher
from src.api.recommender import Recommender

MAX_K = 100
MAX_HISTORY = 500


class Movie(BaseModel):
    movie_id: int
    title: str | None = None
    year: int | None = None
    genres: list[str] = []
    tmdb_id: int | None = None
    poster_path: str | None = None
    overview: str | None = None
    director: str | None = None
    cast: list[str] = []
    vote_average: float | None = None
    runtime: float | None = None
    popularity: int | None = None
    score: float | None = None


class RecommendRequest(BaseModel):
    movie_ids: list[int] = Field(
        default_factory=list,
        max_length=MAX_HISTORY,
        description="MovieLens ids of movies the user liked, oldest first.",
    )
    k: int = Field(default=10, ge=1, le=MAX_K)


class RecommendResponse(BaseModel):
    items: list[Movie]
    strategy: str
    unknown_ids: list[int] = []


class MovieList(BaseModel):
    items: list[Movie]


def create_app(recommender: Recommender | None = None) -> FastAPI:
    """
    Pass a ready `recommender` (tests do this) or leave it None to load one
    from ARTIFACT_DIR at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if recommender is not None:
            app.state.recommender = recommender
        else:
            threads = os.getenv("TORCH_THREADS")
            if threads:
                torch.set_num_threads(int(threads))
            app.state.recommender = Recommender.from_artifacts(
                os.getenv("ARTIFACT_DIR", "artifacts"),
                device=os.getenv("DEVICE", "cpu"),
            )
        r = app.state.recommender
        app.state.batcher = MicroBatcher(
            r.recommend_batch, max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "64"))
        )
        await app.state.batcher.start()
        yield
        await app.state.batcher.stop()

    app = FastAPI(
        title="Movie Recommender API",
        version="0.1.0",
        description="Sequential two-tower retrieval over the MovieLens 32M catalog.",
        lifespan=lifespan,
    )

    origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms"],
    )

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    def rec(request: Request) -> Recommender:
        return request.app.state.recommender

    # Lookups are plain `def` handlers, which FastAPI runs in its thread pool.
    # /recommend is async and hands inference to the micro-batcher, so the
    # model never runs on the event loop and concurrent requests share one
    # forward pass.

    @app.get("/health")
    def health(request: Request):
        r = rec(request)
        batcher = request.app.state.batcher
        return {
            "status": "ok",
            "movies": r.num_movies,
            "device": r.device,
            "batches_run": batcher.batches_run,
            "requests_batched": batcher.items_run,
        }

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(body: RecommendRequest, request: Request):
        r = rec(request)
        indices, unknown = r.to_indices(body.movie_ids)
        if not indices:
            return {"items": r.popular(body.k), "strategy": "popularity", "unknown_ids": unknown}
        items = await request.app.state.batcher.submit((indices, body.k))
        return {"items": items, "strategy": "two_tower", "unknown_ids": unknown}

    @app.get("/movies/search", response_model=MovieList)
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=100),
        limit: int = Query(default=10, ge=1, le=50),
    ):
        return {"items": rec(request).search(q, limit)}

    @app.get("/movies/popular", response_model=MovieList)
    def popular(request: Request, k: int = Query(default=20, ge=1, le=MAX_K)):
        return {"items": rec(request).popular(k)}

    @app.get("/movies/{movie_id}", response_model=Movie)
    def get_movie(movie_id: int, request: Request):
        movie = rec(request).get(movie_id)
        if movie is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        return movie

    @app.get("/movies/{movie_id}/similar", response_model=MovieList)
    def similar(movie_id: int, request: Request, k: int = Query(default=10, ge=1, le=MAX_K)):
        items = rec(request).similar(movie_id, k)
        if items is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        return {"items": items}

    return app


app = create_app()
