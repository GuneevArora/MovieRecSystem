"""
Train the sequential two-tower retrieval model.

Each training example is a window of a user's chronologically ordered
positive interactions. Every position in the window predicts the next movie
with a full softmax over the catalog, so no negative sampling is needed.

Model selection uses the validation split (full-catalog ranking, NDCG@10).
The test split is never touched here.

Usage
-----
    python -m src.train
    python -m src.train --epochs 60 --dim 256 --max-len 50
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_sequences, load_movie_features
from src.evaluate import EvalData, evaluate, model_score_fn, format_metrics
from src.model import TwoTowerModel

MODEL_DIR = Path("models")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="two_tower_seq")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-users", type=int, default=20000)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class WindowSampler:
    """
    Draws training windows directly on the GPU.

    One epoch visits every trainable user once. For each user a window end
    is chosen: half the time the most recent interactions (which is what
    evaluation and serving see), otherwise a random point in the history so
    that long histories are fully used across epochs.
    """

    def __init__(self, data: EvalData, max_len: int, batch_size: int, seed: int):
        self.data = data
        self.max_len = max_len
        self.batch_size = batch_size
        lengths = data.offsets[1:] - data.offsets[:-1]
        self.users = torch.nonzero(lengths >= 2).squeeze(1)
        self.generator = torch.Generator(device=data.device)
        self.generator.manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.users) / self.batch_size)

    def __iter__(self):
        d = self.data
        perm = torch.randperm(
            len(self.users), device=d.device, generator=self.generator
        )
        steps = torch.arange(self.max_len + 1, device=d.device)
        for i in range(0, len(perm), self.batch_size):
            users = self.users[perm[i : i + self.batch_size]]
            start = d.offsets[users]
            length = d.offsets[users + 1] - start

            rand = torch.rand(len(users), device=d.device, generator=self.generator)
            use_tail = torch.rand(
                len(users), device=d.device, generator=self.generator
            ) < 0.5
            # window end (exclusive), at least 2 so there is one input/target pair
            random_end = 2 + torch.floor(rand * (length - 1)).long()
            end = torch.where(use_tail, length, random_end.clamp(max=length))

            idx = start[:, None] + end[:, None] - (self.max_len + 1) + steps[None, :]
            valid = idx >= start[:, None]
            window = d.items[idx.clamp(min=0)]
            window = torch.where(valid, window, d.num_movies)

            inputs = window[:, :-1]
            targets = window[:, 1:]
            # a target only counts if its input position is a real movie
            target_mask = valid[:, :-1]
            yield inputs, targets, target_mask


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    seqs = load_sequences()
    data = EvalData(seqs, device)
    genres, numerical = load_movie_features()
    print(
        f"Users: {seqs.num_users:,}  Movies: {seqs.num_movies:,}  "
        f"Train interactions: {len(seqs.items):,}"
    )

    config = dict(
        dim=args.dim,
        max_len=args.max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = TwoTowerModel(
        genres=torch.from_numpy(genres),
        numerical=torch.from_numpy(numerical),
        **config,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    sampler = WindowSampler(data, args.max_len, args.batch_size, args.seed)
    total_steps = len(sampler) * args.epochs

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or "embedding" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / f"{args.name}_best.pt"
    history_path = MODEL_DIR / f"{args.name}_history.json"
    best_ndcg, best_epoch, history = -1.0, -1, []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, loss_count = 0.0, 0

        for inputs, targets, target_mask in sampler:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                item_vectors = model.item_vectors()
                hidden = model.encode_sequence(inputs, item_vectors)
                logits = hidden[target_mask] @ item_vectors.T
            loss = F.cross_entropy(logits.float(), targets[target_mask])

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item()
            loss_count += 1

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / loss_count,
            "lr": scheduler.get_last_lr()[0],
            "minutes": (time.time() - started) / 60,
        }
        line = (
            f"epoch {epoch:>3}/{args.epochs}  loss={record['train_loss']:.4f}  "
            f"lr={record['lr']:.2e}  {record['minutes']:.1f} min"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            metrics = evaluate(
                model_score_fn(model),
                data,
                split="validation",
                max_users=args.eval_users,
                max_len=args.max_len,
            )
            record["validation"] = metrics
            line += "\n  " + format_metrics("validation", metrics)

            if metrics["ndcg@10"] > best_ndcg:
                best_ndcg, best_epoch = metrics["ndcg@10"], epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "args": vars(args),
                        "epoch": epoch,
                        "validation": metrics,
                    },
                    best_path,
                )
                line += "\n  saved new best checkpoint"

        print(line, flush=True)
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))

        if (epoch - best_epoch) >= args.patience * args.eval_every and best_epoch > 0:
            print(f"Early stopping: no improvement since epoch {best_epoch}")
            break

    print(
        f"Done. Best validation NDCG@10 = {best_ndcg:.4f} at epoch {best_epoch}. "
        f"Checkpoint: {best_path}"
    )


if __name__ == "__main__":
    main()
