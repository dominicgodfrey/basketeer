import pytest

from app.similarity import InMemoryVectorStore, VectorRecord, VectorStore


def test_upsert_and_query_returns_most_similar() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0]),
            VectorRecord(id="b", vector=[0.0, 1.0]),
            VectorRecord(id="c", vector=[0.9, 0.1]),
        ]
    )
    results = store.query([1.0, 0.0], top_k=2)
    assert [r.id for r in results] == ["a", "c"]
    assert results[0].score == pytest.approx(1.0)


def test_upsert_overwrites_existing_id() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0])])
    store.upsert([VectorRecord(id="a", vector=[0.0, 1.0])])
    assert len(store) == 1
    assert store.query([0.0, 1.0], top_k=1)[0].score == pytest.approx(1.0)


def test_delete_removes_records_and_ignores_unknown_ids() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0])])
    store.delete(["a", "nonexistent"])
    assert len(store) == 0


def test_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0])])
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.query([1.0, 0.0, 0.0])


def test_zero_vector_yields_zero_score_not_nan() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[0.0, 0.0])])
    results = store.query([1.0, 0.0], top_k=1)
    assert results[0].score == 0.0


def test_filter_equality() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], metadata={"position": "SG"}),
            VectorRecord(id="b", vector=[1.0, 0.0], metadata={"position": "PG"}),
        ]
    )
    results = store.query([1.0, 0.0], filter={"position": "SG"})
    assert [r.id for r in results] == ["a"]


def test_filter_gte_and_lte_combined() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0], metadata={"season": 2020}),
            VectorRecord(id="b", vector=[1.0], metadata={"season": 2023}),
            VectorRecord(id="c", vector=[1.0], metadata={"season": 2025}),
        ]
    )
    results = store.query([1.0], filter={"season": {"$gte": 2022, "$lte": 2024}})
    assert [r.id for r in results] == ["b"]


def test_filter_in_operator() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0], metadata={"league": "NBA"}),
            VectorRecord(id="b", vector=[1.0], metadata={"league": "NCAA"}),
            VectorRecord(id="c", vector=[1.0], metadata={"league": "EuroLeague"}),
        ]
    )
    results = store.query([1.0], filter={"league": {"$in": ["NBA", "EuroLeague"]}})
    assert {r.id for r in results} == {"a", "c"}


def test_unsupported_filter_operator_raises() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0], metadata={"season": 2020})])
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        store.query([1.0], filter={"season": {"$ne": 2020}})


def test_in_memory_store_satisfies_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)
