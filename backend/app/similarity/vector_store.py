"""Protocol for vector similarity stores.

The `find_similar` primitive depends on this protocol. Pinecone is the default
backend, but any implementation that satisfies the protocol can plug in. Never
import the Pinecone SDK outside `app/similarity/pinecone_store.py`.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class VectorRecord:
    """A single vector to upsert, with its id and arbitrary metadata."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    """A single result from a similarity query."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector similarity backends.

    Implementations should be safe to construct without making network calls;
    do connection setup lazily on first use.
    """

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update records by id."""
        ...

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Return up to `top_k` most-similar records, optionally filtered by metadata.

        `filter` follows Pinecone filter shape: `{"position": "SG"}`,
        `{"season": {"$gte": 2020}}`, `{"league": {"$in": ["NBA", "EuroLeague"]}}`.
        The in-memory backend supports equality and the `$gte`, `$lte`, `$in`
        operators; Pinecone supports the full expression syntax.
        """
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete records by id. No-op for ids that don't exist."""
        ...
