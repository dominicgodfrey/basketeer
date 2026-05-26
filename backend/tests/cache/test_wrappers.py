from typing import Any

from app.cache import FIND_SIMILAR_PREFIX, InMemoryCache, cached_find_similar
from app.primitives import FindSimilarRequest
from app.similarity import InMemoryVectorStore, QueryResult, VectorRecord, VectorStore


class CountingVectorStore:
    """Wraps a VectorStore and counts query/fetch invocations for cache-hit tests."""

    def __init__(self, inner: VectorStore) -> None:
        self._inner = inner
        self.query_count = 0
        self.fetch_count = 0

    def upsert(self, records: list[VectorRecord]) -> None:
        self._inner.upsert(records)

    def fetch(self, ids: list[str]) -> list[VectorRecord]:
        self.fetch_count += 1
        return self._inner.fetch(ids)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        self.query_count += 1
        return self._inner.query(vector=vector, top_k=top_k, filter=filter)

    def delete(self, ids: list[str]) -> None:
        self._inner.delete(ids)


def _store_with_seed() -> CountingVectorStore:
    inner = InMemoryVectorStore()
    inner.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], metadata={"position": "SG"}),
            VectorRecord(id="b", vector=[0.0, 1.0], metadata={"position": "PG"}),
        ]
    )
    return CountingVectorStore(inner)


def test_first_call_misses_then_caches() -> None:
    store = _store_with_seed()
    cache = InMemoryCache()
    request = FindSimilarRequest(vector=[1.0, 0.0], top_k=2)
    response = cached_find_similar(request, store, cache)
    assert len(response.hits) == 2
    assert store.query_count == 1


def test_second_identical_call_hits_cache() -> None:
    store = _store_with_seed()
    cache = InMemoryCache()
    request = FindSimilarRequest(vector=[1.0, 0.0], top_k=2)
    cached_find_similar(request, store, cache)
    cached_find_similar(request, store, cache)
    cached_find_similar(request, store, cache)
    assert store.query_count == 1  # only first call touched the store


def test_different_request_misses_again() -> None:
    store = _store_with_seed()
    cache = InMemoryCache()
    cached_find_similar(FindSimilarRequest(vector=[1.0, 0.0]), store, cache)
    cached_find_similar(FindSimilarRequest(vector=[0.0, 1.0]), store, cache)
    assert store.query_count == 2


def test_filter_dict_order_does_not_affect_cache_key() -> None:
    """{a:1, b:2} and {b:2, a:1} must produce the same key."""
    store = _store_with_seed()
    cache = InMemoryCache()
    cached_find_similar(
        FindSimilarRequest(vector=[1.0, 0.0], filter={"position": "SG", "season": 2024}),
        store,
        cache,
    )
    cached_find_similar(
        FindSimilarRequest(vector=[1.0, 0.0], filter={"season": 2024, "position": "SG"}),
        store,
        cache,
    )
    assert store.query_count == 1


def test_cache_key_starts_with_find_similar_prefix() -> None:
    store = _store_with_seed()
    cache = InMemoryCache()
    cached_find_similar(FindSimilarRequest(vector=[1.0, 0.0]), store, cache)
    keys = [k for k in cache._store]  # noqa: SLF001 — test inspection
    assert all(k.startswith(FIND_SIMILAR_PREFIX) for k in keys)


def test_delete_prefix_invalidates_cache() -> None:
    store = _store_with_seed()
    cache = InMemoryCache()
    request = FindSimilarRequest(vector=[1.0, 0.0])
    cached_find_similar(request, store, cache)
    cache.delete_prefix(FIND_SIMILAR_PREFIX)
    cached_find_similar(request, store, cache)
    assert store.query_count == 2  # cache was wiped, so second call hit the store


def test_cached_response_round_trips_correctly() -> None:
    store = _store_with_seed()
    cache = InMemoryCache()
    request = FindSimilarRequest(vector=[1.0, 0.0], top_k=2)
    first = cached_find_similar(request, store, cache)
    second = cached_find_similar(request, store, cache)
    assert [h.id for h in first.hits] == [h.id for h in second.hits]
    assert [h.score for h in first.hits] == [h.score for h in second.hits]
    assert first.query_source == second.query_source
