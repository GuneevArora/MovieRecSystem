"""
Offline evaluation: leave-one-out, next-item, full-catalog ranking.

Protocol
--------
Each user's most recent positive (rating >= 4) is the test item and the one
before it is the validation item. For every evaluated user the model scores
ALL movies in the catalog, movies the user already interacted with are
removed, and we record where the held-out movie lands.

There is exactly one relevant item per user, so Recall@K equals HitRate@K.
No sampled negatives are used: sampled metrics are much easier and are not
comparable to full-ranking numbers.

Usage
-----
    python -m src.evaluate --split test
    python -m src.evaluate --split test --checkpoint models/two_tower_seq_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import Sequences, load_sequences, load_movie_features

K_VALUES = (10, 20, 50)


class EvalData:
    """Holds the interaction sequences on the target device."""

    def __init__(self, seqs: Sequences, device: str):
        self.seqs = seqs
        self.device = device
        self.items = torch.from_numpy(seqs.items).to(device)
        self.offsets = torch.from_numpy(seqs.offsets).to(device)
        self.val_item = torch.from_numpy(seqs.val_item).to(device)
        self.test_item = torch.from_numpy(seqs.test_item).to(device)
        self.num_movies = seqs.num_movies

    def eval_users(self, split: str, max_users: int | None, seed: int = 42):
        target = self.seqs.test_item if split == "test" else self.seqs.val_item
        users = np.flatnonzero((target >= 0) & (self.seqs.lengths() >= 1))
        if max_users is not None and len(users) > max_users:
            rng = np.random.default_rng(seed)
            users = np.sort(rng.choice(users, size=max_users, replace=False))
        return torch.from_numpy(users).to(self.device)

    def history(self, users: torch.Tensor, split: str, max_len: int):
        """
        Most recent `max_len` movies for each user, left-padded.

        For the test split the validation movie is appended, since it
        happened before the test movie and is known at prediction time.

        Returns (hist [B, max_len] with pad = num_movies, mask [B, max_len]).
        """
        start = self.offsets[users]
        end = self.offsets[users + 1]
        steps = torch.arange(max_len, device=self.device)

        if split == "test":
            # last slot is the validation item, the rest come from train
            idx = end[:, None] - (max_len - 1) + steps[None, :-1]
            valid = idx >= start[:, None]
            hist = self.items[idx.clamp(min=0)]
            hist = torch.where(valid, hist, self.num_movies)
            val = self.val_item[users]
            has_val = val >= 0
            last = torch.where(has_val, val, self.num_movies)
            hist = torch.cat([hist, last[:, None]], dim=1)
            valid = torch.cat([valid, has_val[:, None]], dim=1)
            # users without a validation item: shift is unnecessary, the pad
            # in the final slot is simply masked out.
        else:
            idx = end[:, None] - max_len + steps[None, :]
            valid = idx >= start[:, None]
            hist = self.items[idx.clamp(min=0)]
            hist = torch.where(valid, hist, self.num_movies)
        return hist, valid

    def mask_seen(self, scores: torch.Tensor, users: torch.Tensor, split: str):
        """Set scores of already-seen movies to -inf, in place."""
        start = self.offsets[users]
        lengths = self.offsets[users + 1] - start
        rows = torch.repeat_interleave(
            torch.arange(len(users), device=self.device), lengths
        )
        within = torch.arange(int(lengths.sum()), device=self.device)
        within = within - torch.repeat_interleave(
            torch.cumsum(lengths, 0) - lengths, lengths
        )
        cols = self.items[torch.repeat_interleave(start, lengths) + within]
        scores[rows, cols] = float("-inf")
        if split == "test":
            val = self.val_item[users]
            has_val = val >= 0
            scores[torch.nonzero(has_val).squeeze(1), val[has_val]] = float("-inf")
        return scores


@torch.no_grad()
def evaluate(
    score_fn,
    data: EvalData,
    split: str = "validation",
    max_users: int | None = None,
    max_len: int = 50,
    batch_size: int = 1024,
    users: torch.Tensor | None = None,
) -> dict:
    """
    score_fn(users, hist, mask) -> [B, num_movies] scores.

    users: optionally restrict evaluation to these user indices.
    """
    users_all = data.eval_users(split, max_users) if users is None else users
    target_all = data.test_item if split == "test" else data.val_item

    ranks = []
    for i in range(0, len(users_all), batch_size):
        users = users_all[i : i + batch_size]
        hist, mask = data.history(users, split, max_len)
        scores = score_fn(users, hist, mask).float()
        target = target_all[users]
        target_score = scores.gather(1, target[:, None])
        scores = data.mask_seen(scores, users, split)
        # rank 0 = best. Ties are counted against the model. The target is
        # never in the seen set, so it matches itself exactly once.
        ranks.append((scores >= target_score).sum(dim=1) - 1)

    ranks = torch.cat(ranks).double()
    metrics = {"split": split, "users": int(len(ranks))}
    for k in K_VALUES:
        hit = ranks < k
        metrics[f"recall@{k}"] = float(hit.double().mean())
        metrics[f"ndcg@{k}"] = float(
            (hit.double() / torch.log2(ranks + 2)).mean()
        )
    metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
    return metrics


def popularity_score_fn(data: EvalData):
    counts = torch.from_numpy(data.seqs.item_counts()).float().to(data.device)
    # tiny index-based jitter breaks ties deterministically
    counts = counts + torch.linspace(0, 0.5, len(counts), device=data.device)

    def score_fn(users, hist, mask):
        return counts[None, :].expand(len(users), -1).clone()

    return score_fn


def model_score_fn(model):
    item_vectors = model.item_vectors()

    def score_fn(users, hist, mask):
        return model.encode_users(hist) @ item_vectors.T

    return score_fn


def load_model(checkpoint_path: Path, device: str):
    from src.model import TwoTowerModel

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    genres, numerical = load_movie_features()
    model = TwoTowerModel(
        genres=torch.from_numpy(genres),
        numerical=torch.from_numpy(numerical),
        **checkpoint["config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def format_metrics(name: str, m: dict) -> str:
    return (
        f"{name:<12} users={m['users']:>6}  "
        f"Recall@10={m['recall@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}  "
        f"Recall@20={m['recall@20']:.4f}  NDCG@20={m['ndcg@20']:.4f}  "
        f"Recall@50={m['recall@50']:.4f}  MRR={m['mrr']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--checkpoint", default="models/two_tower_seq_best.pt")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = EvalData(load_sequences(), device)

    results = {}
    results["popularity"] = evaluate(
        popularity_score_fn(data), data, args.split, args.max_users
    )
    print(format_metrics("popularity", results["popularity"]))

    model, checkpoint = load_model(Path(args.checkpoint), device)
    results["two_tower"] = evaluate(
        model_score_fn(model),
        data,
        args.split,
        args.max_users,
        max_len=checkpoint["config"]["max_len"],
    )
    print(format_metrics("two_tower", results["two_tower"]))

    for k in (10, 20):
        base = results["popularity"][f"ndcg@{k}"]
        ours = results["two_tower"][f"ndcg@{k}"]
        print(f"NDCG@{k} lift over popularity: {ours / base:.2f}x")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results["protocol"] = (
        "leave-one-out next-item, full-catalog ranking, seen items filtered"
    )
    results["checkpoint"] = str(args.checkpoint)
    output.write_text(json.dumps(results, indent=2))
    print("Saved", output)


if __name__ == "__main__":
    main()
