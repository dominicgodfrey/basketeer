import pytest
from pydantic import ValidationError

from app.primitives import FindSimilarRequest, find_similar
from app.primitives.find_similar import PlayerNotFoundError
from app.similarity import InMemoryVectorStore, VectorRecord


def _seed_store() -> InMemoryVectorStore:
    """A small fixture: 4 players in a 2D space with metadata.

    a — pure scorer (1, 0), SG
    b — pure passer (0, 1), PG
    c — almost-a (0.95, 0.1), SG
    d — far away (-1, 0), C
    """
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], metadata={"position": "SG", "season": 2024}),
            VectorRecord(id="b", vector=[0.0, 1.0], metadata={"position": "PG", "season": 2024}),
            VectorRecord(id="c", vector=[0.95, 0.1], metadata={"position": "SG", "season": 2022}),
            VectorRecord(id="d", vector=[-1.0, 0.0], metadata={"position": "C", "season": 2024}),
        ]
    )
    return store


def test_query_by_vector_returns_ranked_hits() -> None:
    store = _seed_store()
    response = find_similar(FindSimilarRequest(vector=[1.0, 0.0], top_k=3), store)
    assert response.query_source == "vector"
    assert [h.id for h in response.hits] == ["a", "c", "b"]
    assert response.hits[0].score == pytest.approx(1.0)


def test_query_by_player_id_resolves_vector_and_excludes_self() -> None:
    store = _seed_store()
    response = find_similar(FindSimilarRequest(player_id="a", top_k=2), store)
    assert response.query_source == "player_id"
    ids = [h.id for h in response.hits]
    assert "a" not in ids
    assert ids[0] == "c"  # closest neighbor to a is c


def test_query_by_player_id_returns_full_top_k_when_self_present() -> None:
    """If the self-match would otherwise occupy a slot, we fetch top_k+1 and trim."""
    store = _seed_store()
    response = find_similar(FindSimilarRequest(player_id="a", top_k=3), store)
    assert len(response.hits) == 3
    assert all(h.id != "a" for h in response.hits)


def test_metadata_filter_applied_pinecone_side() -> None:
    store = _seed_store()
    response = find_similar(
        FindSimilarRequest(vector=[1.0, 0.0], top_k=10, filter={"position": "SG"}),
        store,
    )
    assert {h.id for h in response.hits} == {"a", "c"}


def test_missing_player_id_raises() -> None:
    store = _seed_store()
    with pytest.raises(PlayerNotFoundError):
        find_similar(FindSimilarRequest(player_id="missing"), store)


def test_request_requires_exactly_one_query_source() -> None:
    with pytest.raises(ValidationError, match="Provide exactly one"):
        FindSimilarRequest()  # neither
    with pytest.raises(ValidationError, match="Provide exactly one"):
        FindSimilarRequest(player_id="a", vector=[1.0])  # both


def test_top_k_clamped_to_50() -> None:
    with pytest.raises(ValidationError):
        FindSimilarRequest(vector=[1.0], top_k=51)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        FindSimilarRequest(vector=[1.0], top_k=0)


def test_hits_carry_metadata_through() -> None:
    store = _seed_store()
    response = find_similar(FindSimilarRequest(vector=[1.0, 0.0], top_k=1), store)
    assert response.hits[0].metadata["position"] == "SG"
    assert response.hits[0].metadata["season"] == 2024
