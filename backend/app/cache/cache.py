"""Protocol for result caches.

Two use sites once the agent is wired:
- `query_stats` results — hash the SQL, cache for 24h (most season-ended queries
  are deterministic).
- `find_similar` results — hash (query_vector, top_k, filter), cache for 24h.

Implementations should treat values as JSON-serializable. The in-memory cache
is permissive (stores anything); the planned Redis cache will JSON-encode on
set and decode on get, so callers should avoid stashing non-serializable
objects there.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Key-value cache with optional TTL.

    Implementations may evict entries before their TTL expires (e.g. LRU under
    memory pressure). Callers must tolerate cache misses on entries they
    previously set.
    """

    def get(self, key: str) -> Any | None:
        """Return the value for `key`, or None if missing or expired."""
        ...

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store `value` under `key`. `ttl_seconds=None` means no expiry."""
        ...

    def delete(self, key: str) -> None:
        """Remove `key`. No-op if absent."""
        ...
