import pytest

from app.cache import Cache, InMemoryCache


class FakeClock:
    """Manually-advanced clock for deterministic TTL tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_set_and_get_round_trip() -> None:
    cache = InMemoryCache()
    cache.set("key", {"hello": "world"})
    assert cache.get("key") == {"hello": "world"}


def test_get_missing_key_returns_none() -> None:
    cache = InMemoryCache()
    assert cache.get("absent") is None


def test_overwrite_replaces_value_and_ttl() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    cache.set("key", "old", ttl_seconds=10)
    cache.set("key", "new", ttl_seconds=100)
    clock.advance(11)
    # Old TTL would have expired; new TTL keeps it alive
    assert cache.get("key") == "new"


def test_delete_removes_key() -> None:
    cache = InMemoryCache()
    cache.set("key", "value")
    cache.delete("key")
    assert cache.get("key") is None


def test_delete_missing_key_is_silent() -> None:
    cache = InMemoryCache()
    cache.delete("absent")  # should not raise


def test_ttl_expires_entries() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    cache.set("key", "value", ttl_seconds=10)
    assert cache.get("key") == "value"
    clock.advance(9.999)
    assert cache.get("key") == "value"
    clock.advance(0.001)
    assert cache.get("key") is None


def test_no_ttl_means_no_expiry() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    cache.set("key", "value")  # ttl=None
    clock.advance(10**9)
    assert cache.get("key") == "value"


def test_expired_entry_is_removed_from_store() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    cache.set("key", "value", ttl_seconds=1)
    clock.advance(2)
    cache.get("key")  # triggers eviction
    assert len(cache) == 0


def test_clear_wipes_everything() -> None:
    cache = InMemoryCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_non_positive_ttl_raises() -> None:
    cache = InMemoryCache()
    with pytest.raises(ValueError, match="must be positive"):
        cache.set("k", "v", ttl_seconds=0)
    with pytest.raises(ValueError, match="must be positive"):
        cache.set("k", "v", ttl_seconds=-5)


def test_in_memory_cache_satisfies_protocol() -> None:
    assert isinstance(InMemoryCache(), Cache)
