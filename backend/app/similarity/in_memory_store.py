"""In-memory VectorStore for tests and local development.

Brute-force cosine similarity. Not for production — switch to PineconeVectorStore
there. Supports a subset of Pinecone filter expressions: equality and the
`$gte`, `$lte`, `$in` operators.
"""

import math
from typing import Any

from app.similarity.vector_store import QueryResult, VectorRecord


class InMemoryVectorStore:
    """Brute-force in-memory vector store. Not thread-safe."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.id] = record

    def fetch(self, ids: list[str]) -> list[VectorRecord]:
        return [self._records[i] for i in ids if i in self._records]

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        candidates = (r for r in self._records.values() if _matches_filter(r.metadata, filter))
        scored = [
            QueryResult(
                id=r.id,
                score=_cosine_similarity(vector, r.vector),
                metadata=dict(r.metadata),
            )
            for r in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self._records.pop(record_id, None)

    def __len__(self) -> int:
        return len(self._records)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filter(metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    if not filter:
        return True
    for key, expected in filter.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            for op, op_value in expected.items():
                if not _matches_op(actual, op, op_value):
                    return False
        elif actual != expected:
            return False
    return True


def _matches_op(actual: Any, op: str, op_value: Any) -> bool:
    if op == "$gte":
        return actual is not None and actual >= op_value
    if op == "$lte":
        return actual is not None and actual <= op_value
    if op == "$in":
        return actual in op_value
    raise ValueError(f"Unsupported filter operator: {op}")
