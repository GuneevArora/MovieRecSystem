import numpy as np
import torch

from src.data import Sequences
from src.evaluate import EvalData, evaluate
from src.model import TwoTowerModel


def tiny_model(num_movies=30, max_len=8):
    torch.manual_seed(0)
    return TwoTowerModel(
        genres=torch.rand(num_movies, 5),
        numerical=torch.rand(num_movies, 3),
        dim=16,
        max_len=max_len,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    ).eval()


def test_shapes():
    model = tiny_model()
    history = torch.randint(0, 30, (4, 8))
    assert model.item_vectors().shape == (30, 16)
    assert model.encode_users(history).shape == (4, 16)
    assert model(history).shape == (4, 30)


def test_encoder_is_causal():
    """Changing a later movie must not change earlier hidden states."""
    model = tiny_model()
    a = torch.randint(0, 30, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 30
    ha, hb = model.encode_sequence(a), model.encode_sequence(b)
    assert torch.allclose(ha[:, :-1], hb[:, :-1], atol=1e-5)
    assert not torch.allclose(ha[:, -1], hb[:, -1], atol=1e-5)


def test_left_padding_is_ignored():
    """A short history gives the same user vector however much it is padded."""
    model = tiny_model()
    pad = model.pad_idx
    short = torch.tensor([[3, 7, 11]])
    padded = torch.tensor([[pad, pad, pad, pad, pad, 3, 7, 11]])
    assert torch.allclose(
        model.encode_users(short), model.encode_users(padded), atol=1e-5
    )


def toy_data():
    # user 0: train [0, 1, 2], val 3, test 4
    # user 1: train [5, 6],    val 7, test 8
    seqs = Sequences(
        items=np.array([0, 1, 2, 5, 6]),
        offsets=np.array([0, 3, 5]),
        val_item=np.array([3, 7]),
        test_item=np.array([4, 8]),
        num_movies=10,
    )
    return EvalData(seqs, "cpu")


def test_history_includes_validation_item_for_test_split():
    data = toy_data()
    users = torch.tensor([0, 1])
    hist, mask = data.history(users, "test", max_len=4)
    assert hist.tolist() == [[0, 1, 2, 3], [10, 5, 6, 7]]
    assert mask.tolist() == [[True] * 4, [False, True, True, True]]

    hist, _ = data.history(users, "validation", max_len=4)
    assert hist.tolist() == [[10, 0, 1, 2], [10, 10, 5, 6]]


def test_perfect_and_worst_scorers():
    data = toy_data()

    def perfect(users, hist, mask):
        scores = torch.zeros(len(users), 10)
        scores[torch.arange(len(users)), data.test_item[users]] = 1.0
        return scores

    metrics = evaluate(perfect, data, split="test", max_len=4)
    assert metrics["recall@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0

    def seen_only(users, hist, mask):
        # ranks train movies on top; they must be filtered, not counted as misses
        scores = torch.zeros(len(users), 10)
        scores[0, [0, 1, 2, 3]] = 5.0
        scores[1, [5, 6, 7]] = 5.0
        scores[torch.arange(len(users)), data.test_item[users]] = 1.0
        return scores

    metrics = evaluate(seen_only, data, split="test", max_len=4)
    assert metrics["ndcg@10"] == 1.0
