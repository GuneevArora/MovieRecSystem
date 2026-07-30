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
            # the last film has no TMDB match
            "tmdb_id": pd.array(
                [5000 + i for i in range(NUM_MOVIES - 1)] + [None], dtype="Int64"
            ),
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


def test_recommend_by_tmdb_id_matches_movielens_ids(client):
    by_id = client.post("/recommend", json={"movie_ids": [101, 105, 110], "k": 6}).json()
    by_tmdb = client.post(
        "/recommend/tmdb", json={"tmdb_ids": [5001, 5005, 5010, 42], "k": 6}
    ).json()
    assert by_tmdb["strategy"] == "two_tower"
    assert by_tmdb["unknown_ids"] == [42]
    assert [m["movie_id"] for m in by_tmdb["items"]] == [m["movie_id"] for m in by_id["items"]]
    assert all(m["tmdb_id"] == m["movie_id"] + 4900 for m in by_tmdb["items"])


def test_recommend_by_tmdb_id_within_and_outside_a_library(client):
    history = [5001, 5005, 5010]
    library = [5001, 5002, 5003, 5020, 5021, 99999]

    inside = client.post(
        "/recommend/tmdb",
        json={"tmdb_ids": history, "k": 10, "candidate_tmdb_ids": library},
    ).json()["items"]
    # only library films, minus the watched one and the id the model lacks
    assert sorted(m["tmdb_id"] for m in inside) == [5002, 5003, 5020, 5021]
    scores = [m["score"] for m in inside]
    assert scores == sorted(scores, reverse=True)

    outside = client.post(
        "/recommend/tmdb",
        json={"tmdb_ids": history, "k": 10, "exclude_tmdb_ids": library},
    ).json()["items"]
    assert len(outside) == 10
    assert not {m["tmdb_id"] for m in outside} & set(library + history)

    # nothing watched yet: popularity, still respecting the library filter
    body = client.post(
        "/recommend/tmdb", json={"tmdb_ids": [], "k": 2, "candidate_tmdb_ids": library}
    ).json()
    assert body["strategy"] == "popularity"
    assert [m["tmdb_id"] for m in body["items"]] == [5001, 5002]


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


def test_cold_start_movies_are_searchable_pickable_and_recommendable():
    """Movies with no ID embedding work through appended vectors."""
    torch.manual_seed(0)
    base, extra = 20, 3
    model = TwoTowerModel(
        genres=torch.rand(base, 5),
        numerical=torch.rand(base, 3),
        dim=16,
        max_len=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    total = base + extra
    movies = pd.DataFrame(
        {
            "movie_id": [100 + i for i in range(base)] + [9000 + i for i in range(extra)],
            "title": [f"Old {i}" for i in range(base)] + [f"Fresh {i}" for i in range(extra)],
            "year": pd.array([2000] * base + [2025] * extra, dtype="Int64"),
            "genres": [["Drama"]] * total,
            "popularity": [base - i for i in range(base)] + [0] * extra,
            "cold_start": [False] * base + [True] * extra,
            "search_weight": [float(base - i) for i in range(base)] + [50.0, 2.0, 1.0],
        }
    )
    recommender = Recommender(model, movies, extra_item_vectors=torch.randn(extra, 16))
    assert recommender.item_vectors.shape[0] == total

    # point the first new film straight at this user so it has to rank first
    with torch.inference_mode():
        user = recommender.model.encode_users(
            torch.tensor([[1, 2]]), recommender.item_vectors
        )
        recommender.item_vectors[base] = user[0] * 100

    with TestClient(create_app(recommender)) as client:
        found = client.get("/movies/search", params={"q": "fresh"}).json()["items"]
        assert [m["movie_id"] for m in found] == [9000, 9001, 9002]
        assert all(m["cold_start"] for m in found)

        body = client.post("/recommend", json={"movie_ids": [101, 102], "k": 3}).json()
        assert body["items"][0]["movie_id"] == 9000
        assert body["items"][0]["cold_start"] is True

        # a new film as the only pick still yields model recommendations
        body = client.post("/recommend", json={"movie_ids": [9001], "k": 5}).json()
        assert body["strategy"] == "two_tower"
        assert 9001 not in [m["movie_id"] for m in body["items"]]
        assert len(body["items"]) == 5

    # padding must not be confused with the first cold-start row
    alone = recommender.recommend_batch([([3], 5)])[0]
    mixed = recommender.recommend_batch([([3], 5), (list(range(8)), 5)])[0]
    assert [m["movie_id"] for m in alone] == [m["movie_id"] for m in mixed]
