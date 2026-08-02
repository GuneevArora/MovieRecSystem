"""
Export the model in a form the browser can run without a server.

The demo site can then be hosted as static files: the browser downloads the
user tower's weights and one vector per film, runs the Transformer in
TypeScript (web/src/lib/engine.ts) and ranks the catalog with a dot product.

Run after export_serving_artifacts:

    python -m scripts.export_web_model

Writes to web/public/model/ (not tracked by git):

    manifest.json   shapes, tensor offsets, counts
    weights.bin     user-tower weights, float16, little endian
    items.bin       one int8 vector per film, row-wise quantised
    scales.bin      float32 scale per film (vector = int8 * scale)
    movies.json     the catalog, column-wise, in the same order as items.bin

and artifacts/web_parity.json: histories with the exact top results from
PyTorch, used by web/scripts/check-parity.mts to verify the TypeScript port.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.api.recommender import Recommender

PARITY_HISTORIES = [
    [2571, 79132, 541, 1196],
    [1, 3114, 4886, 6377],
    [1258, 1219, 593, 1387],
    [318],
    [296, 50, 47, 593, 2959, 858, 1221, 1213, 608, 111],
]


def user_tower_tensors(model) -> list[tuple[str, torch.Tensor]]:
    """The tensors engine.ts needs, under the names it looks them up by."""
    tensors = [("position", model.position_embedding.weight)]
    for i, layer in enumerate(model.encoder.layers):
        tensors += [
            (f"l{i}.norm1.w", layer.norm1.weight),
            (f"l{i}.norm1.b", layer.norm1.bias),
            (f"l{i}.attn.in.w", layer.self_attn.in_proj_weight),
            (f"l{i}.attn.in.b", layer.self_attn.in_proj_bias),
            (f"l{i}.attn.out.w", layer.self_attn.out_proj.weight),
            (f"l{i}.attn.out.b", layer.self_attn.out_proj.bias),
            (f"l{i}.norm2.w", layer.norm2.weight),
            (f"l{i}.norm2.b", layer.norm2.bias),
            (f"l{i}.ff1.w", layer.linear1.weight),
            (f"l{i}.ff1.b", layer.linear1.bias),
            (f"l{i}.ff2.w", layer.linear2.weight),
            (f"l{i}.ff2.b", layer.linear2.bias),
        ]
    tensors += [("out.w", model.output_norm.weight), ("out.b", model.output_norm.bias)]
    return tensors


def column(series: pd.Series) -> list:
    """A JSON-safe list with None for missing values."""
    return [None if pd.isna(value) else value for value in series.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="web/public/model")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recommender = Recommender.from_artifacts(args.artifacts)
    model, movies = recommender.model, recommender.movies

    # ---- user tower weights ------------------------------------------------
    tensors, offset, blobs = {}, 0, []
    for name, tensor in user_tower_tensors(model):
        values = tensor.detach().cpu().numpy().astype("<f2")
        tensors[name] = {"offset": offset, "shape": list(values.shape)}
        offset += values.size
        blobs.append(values.tobytes())
    (out / "weights.bin").write_bytes(b"".join(blobs))

    # ---- film vectors, int8 with one scale per row ---------------------------
    vectors = recommender.item_vectors.cpu().numpy().astype(np.float32)
    scales = np.abs(vectors).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    quantised = np.clip(np.round(vectors / scales[:, None]), -127, 127).astype(np.int8)
    (out / "items.bin").write_bytes(quantised.tobytes())
    (out / "scales.bin").write_bytes(scales.astype("<f4").tobytes())

    # ---- catalog ---------------------------------------------------------------
    genre_names = sorted({genre for genres in movies["genres"] for genre in genres})
    genre_index = {name: i for i, name in enumerate(genre_names)}
    weight = movies["search_weight"] if "search_weight" in movies else movies["popularity"]
    search_rank = np.empty(len(movies), dtype=np.int64)
    search_rank[np.argsort(-weight.to_numpy(), kind="stable")] = np.arange(len(movies))
    catalog = {
        "genre_names": genre_names,
        "movie_id": movies["movie_id"].astype(int).tolist(),
        "title": column(movies["title"]),
        "year": [None if v is None else int(v) for v in column(movies["year"])],
        "genres": [[genre_index[g] for g in genres] for genres in movies["genres"]],
        "poster_path": column(movies["poster_path"]),
        "director": column(movies["director"]),
        "tmdb_id": [None if v is None else int(v) for v in column(movies["tmdb_id"])],
        "cold_start": [int(v) for v in movies["cold_start"].fillna(False)],
        "search_rank": search_rank.tolist(),
    }
    (out / "movies.json").write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    # ---- parity fixture: exact PyTorch results on float32 vectors -----------
    cases = []
    for history in PARITY_HISTORIES:
        indices, _ = recommender.to_indices(history)
        items = recommender.recommend_batch([(indices, 50)])[0]
        cases.append(
            {
                "movie_ids": history,
                "top": [item["movie_id"] for item in items],
                "scores": [item["score"] for item in items],
            }
        )
    Path(args.artifacts, "web_parity.json").write_text(json.dumps(cases), encoding="utf-8")

    manifest = {
        "dim": model.dim,
        "max_len": model.max_len,
        "num_heads": model.num_heads,
        "num_layers": len(model.encoder.layers),
        "num_movies": len(movies),
        "tensors": tensors,
        # uncompressed sizes, so the browser can show download progress
        "bytes": {
            name: (out / name).stat().st_size
            for name in ("weights.bin", "items.bin", "scales.bin", "movies.json")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    for path in sorted(out.iterdir()):
        print(f"{path.name:15s} {path.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
