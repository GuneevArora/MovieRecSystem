import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.recommender import Recommender
from src.model import TwoTowerModel

NUM_MOVIES = 40


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(0)
    model = TwoTowerModel(
        genres=torch.rand(NUM_MOVIES, 5),
        numerical=torch.rand(NUM_MOVIES, 3),
        dim=16,
        max_len=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    movies = pd.DataFrame(
        {
            "movie_idx": range(NUM_MOVIES),
            # ids deliberately differ from indices
            "movie_id": [100 + i for i in range(NUM_MOVIES)],
            "title": [f"Movie {i}" for i in range(NUM_MOVIES)],
            "year": pd.array([2000 + i if i else None for i in range(NUM_MOVIES)], dtype="Int64"),
            "genres": [["Drama"] if i % 2 else [] for i in range(NUM_MOVIES)],
            "popularity": [NUM_MOVIES - i for i in range(NUM_MOVIES)],
        }
    )
    movies.loc[3, "title"] = "The Matrix"
    with TestClient(create_app(Recommender(model, movies))) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["movies"] == NUM_MOVIES


def test_recommend_excludes_history_and_respects_k(client):
    history = [101, 105, 110]
    response = client.post("/recommend", json={"movie_ids": history, "k": 5})
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    body = response.json()
    assert body["strategy"] == "two_tower"
    ids = [item["movie_id"] for item in body["items"]]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert not set(ids) & set(history)
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_recommend_is_order_sensitive(client):
    a = client.post("/recommend", json={"movie_ids": [101, 105, 110], "k": 10}).json()
    b = client.post("/recommend", json={"movie_ids": [110, 105, 101], "k": 10}).json()
    assert [i["score"] for i in a["items"]] != [i["score"] for i in b["items"]]


def test_recommend_cold_start_falls_back_to_popularity(client):
    body = client.post("/recommend", json={"movie_ids": [], "k": 3}).json()
    assert body["strategy"] == "popularity"
    assert [item["movie_id"] for item in body["items"]] == [100, 101, 102]

    body = client.post("/recommend", json={"movie_ids": [999999], "k": 3}).json()
    assert body["strategy"] == "popularity"
    assert body["unknown_ids"] == [999999]


def test_recommend_reports_unknown_ids_but_still_recommends(client):
    body = client.post("/recommend", json={"movie_ids": [101, 999999], "k": 3}).json()
    assert body["strategy"] == "two_tower"
    assert body["unknown_ids"] == [999999]


def test_recommend_validates_k(client):
    assert client.post("/recommend", json={"movie_ids": [101], "k": 0}).status_code == 422
    assert client.post("/recommend", json={"movie_ids": [101], "k": 1000}).status_code == 422


def test_search(client):
    items = client.get("/movies/search", params={"q": "matrix"}).json()["items"]
    assert [item["movie_id"] for item in items] == [103]
    assert client.get("/movies/search", params={"q": "zzzz"}).json()["items"] == []
    # prefix matches come first, then popularity order
    items = client.get("/movies/search", params={"q": "movie 1", "limit": 3}).json()["items"]
    assert [item["title"] for item in items] == ["Movie 1", "Movie 10", "Movie 11"]


def test_get_movie_and_missing_values(client):
    movie = client.get("/movies/100").json()
    assert movie["title"] == "Movie 0"
    assert movie["year"] is None
    assert movie["genres"] == []
    assert client.get("/movies/999999").status_code == 404


def test_similar(client):
    items = client.get("/movies/103/similar", params={"k": 4}).json()["items"]
    assert len(items) == 4
    assert 103 not in [item["movie_id"] for item in items]
    assert all(-1.0001 <= item["score"] <= 1.0001 for item in items)
    assert client.get("/movies/999999/similar").status_code == 404


def test_popular(client):
    items = client.get("/movies/popular", params={"k": 2}).json()["items"]
    assert [item["movie_id"] for item in items] == [100, 101]


def test_batched_results_match_single_requests(client):
    """Padding inside a mixed-length batch must not change anyone's results."""
    recommender = client.app.state.recommender
    requests = [([1, 5, 10], 5), ([2], 3), (list(range(12, 30)), 7)]
    batched = recommender.recommend_batch(requests)
    for request, together in zip(requests, batched):
        alone = recommender.recommend_batch([request])[0]
        assert [m["movie_id"] for m in alone] == [m["movie_id"] for m in together]
        assert len(together) == request[1]


def test_concurrent_requests_are_batched_and_correct(client):
    from concurrent.futures import ThreadPoolExecutor

    histories = [[100 + i, 100 + (i * 7) % NUM_MOVIES] for i in range(1, 30)]

    def call(history):
        return client.post("/recommend", json={"movie_ids": history, "k": 4}).json()

    expected = [call(h) for h in histories]
    with ThreadPoolExecutor(16) as pool:
        got = list(pool.map(call, histories))
    for a, b in zip(expected, got):
        assert [m["movie_id"] for m in a["items"]] == [m["movie_id"] for m in b["items"]]
