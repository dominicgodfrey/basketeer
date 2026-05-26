"""In-memory Cache implementation for tests and local development.

Single-process, not LRU-bounded, not thread-safe. For production we'll swap to
RedisCache (lands when REDIS_URL is configured). The `clock` parameter lets
tests inject a fake clock for deterministic TTL behavior.
"""

import time
from collections.abc import Callable
from typing import Any


class InMemoryCache:
    """Dict-backed cache with per-entry TTL."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}
        self._clock = clock

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and self._clock() >= expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is None:
            expiry: float | None = None
        elif ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds!r}")
        else:
            expiry = self._clock() + ttl_seconds
        self._store[key] = (value, expiry)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)

    def clear(self) -> None:
        """Wipe all entries. Convenience for tests and post-ingestion invalidation."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
