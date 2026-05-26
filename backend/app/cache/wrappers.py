"""Caching wrappers around deterministic primitives.

Keys are namespaced (`<primitive>:<sha256>`) so the collaborator's ingestion
code can invalidate a single primitive's cache after a refresh:

    cache.delete_prefix("find_similar:")
    cache.delete_prefix("query_stats:")

Values are JSON-serializable (Pydantic `model_dump()` output), which keeps the
contract identical for InMemoryCache today and RedisCache once that lands.
"""

import hashlib
import json

from app.cache.cache import Cache
from app.logging_setup import get_logger
from app.primitives.find_similar import (
    FindSimilarRequest,
    FindSimilarResponse,
    find_similar,
)
from app.similarity import VectorStore

logger = get_logger(__name__)

DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 hours

FIND_SIMILAR_PREFIX = "find_similar:"


def cached_find_similar(
    request: FindSimilarRequest,
    store: VectorStore,
    cache: Cache,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> FindSimilarResponse:
    """Return a cached `find_similar` result if present, else compute and cache.

    Cache keys are `find_similar:<sha256(canonical_request_json)>`. To
    invalidate after ingestion: `cache.delete_prefix("find_similar:")`.
    """
    key = _find_similar_key(request)
    cached = cache.get(key)
    if cached is not None:
        logger.info("cache_hit primitive=find_similar key=%s", key)
        return FindSimilarResponse.model_validate(cached)

    response = find_similar(request, store)
    cache.set(key, response.model_dump(), ttl_seconds=ttl_seconds)
    logger.info(
        "cache_miss primitive=find_similar key=%s hits=%d", key, len(response.hits)
    )
    return response


def _find_similar_key(request: FindSimilarRequest) -> str:
    """Stable cache key from a FindSimilarRequest.

    Uses `json.dumps(..., sort_keys=True)` so nested dicts (e.g. `filter`) hash
    deterministically regardless of construction order. Lists (e.g. `vector`)
    preserve order, which is correct: [1, 0] and [0, 1] are different queries.
    """
    canonical = json.dumps(request.model_dump(exclude_none=False), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{FIND_SIMILAR_PREFIX}{digest}"
