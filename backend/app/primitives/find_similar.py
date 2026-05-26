"""`find_similar` primitive: ranked similarity search over the vector store.

Agent contract:
- Provide exactly one of `player_id` (resolve to that player's stored vector) or
  `vector` (use directly). The delta-description path — "similar to peak Klay
  but with better playmaking" — will land in a follow-up commit once an LLM
  client is wired into the primitive.
- `filter` follows the Pinecone filter shape (see VectorStore.query docstring).
- `top_k` is capped at 50 to bound agent context usage.
- When querying by `player_id`, the player itself is excluded from the results.

Pydantic request/response models are the boundary types: this primitive is
LLM-callable, so its inputs must be schema-validated and its outputs must be
JSON-serializable.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.logging_setup import get_logger
from app.similarity import VectorStore

logger = get_logger(__name__)

MAX_TOP_K = 50


class FindSimilarRequest(BaseModel):
    """Input schema for the `find_similar` primitive."""

    player_id: str | None = Field(
        default=None,
        description="Look up this player's stored vector and use it as the query.",
    )
    vector: list[float] | None = Field(
        default=None,
        description="Use this vector directly as the query. Mutually exclusive with player_id.",
    )
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    filter: dict[str, Any] | None = Field(
        default=None,
        description="Pinecone-shape metadata filter; e.g. {'position': 'SG'}.",
    )

    @model_validator(mode="after")
    def _exactly_one_query_source(self) -> "FindSimilarRequest":
        if (self.player_id is None) == (self.vector is None):
            raise ValueError("Provide exactly one of player_id or vector")
        return self


class FindSimilarHit(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class FindSimilarResponse(BaseModel):
    hits: list[FindSimilarHit]
    query_source: str = Field(description="'vector' or 'player_id' — for logs")


class PlayerNotFoundError(LookupError):
    """Raised when a `player_id` query references an id that isn't in the store."""


def find_similar(request: FindSimilarRequest, store: VectorStore) -> FindSimilarResponse:
    """Run a similarity query against `store` per the request contract."""
    if request.vector is not None:
        query_vector = request.vector
        source = "vector"
        fetch_k = request.top_k
    else:
        assert request.player_id is not None  # guaranteed by validator
        fetched = store.fetch([request.player_id])
        if not fetched:
            raise PlayerNotFoundError(f"player_id {request.player_id!r} not found in store")
        query_vector = fetched[0].vector
        source = "player_id"
        # Fetch one extra so dropping the self-match still yields top_k results.
        fetch_k = request.top_k + 1

    raw_results = store.query(vector=query_vector, top_k=fetch_k, filter=request.filter)

    if request.player_id is not None:
        raw_results = [r for r in raw_results if r.id != request.player_id][: request.top_k]

    hits = [FindSimilarHit(id=r.id, score=r.score, metadata=r.metadata) for r in raw_results]
    logger.info(
        "find_similar source=%s top_k=%d filter=%s hits=%d",
        source,
        request.top_k,
        request.filter or {},
        len(hits),
    )
    return FindSimilarResponse(hits=hits, query_source=source)
