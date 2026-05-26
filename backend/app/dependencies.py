"""FastAPI dependency factories for the agent runtime.

Each factory returns the default impl for its abstraction. Defaults are safe
for local development (no external services touched) but produce canned
responses; swap each one for a real implementation as the corresponding
account is provisioned. See `SETUP.md` for the "Decisions still pending"
checklist.

Tests override these via `app.dependency_overrides[<factory>] = <stub>`.
"""

from functools import lru_cache

from app.cache import Cache, InMemoryCache
from app.llm.providers import FakeProvider, LLMProvider
from app.sandbox import RaisingSandbox, Sandbox
from app.similarity import InMemoryVectorStore, VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    """Default: `InMemoryVectorStore` (empty until populated).

    Swap to `PineconeVectorStore` once `PINECONE_API_KEY` /
    `PINECONE_INDEX_NAME` are set and the Pinecone implementation lands.
    """
    return InMemoryVectorStore()


@lru_cache
def get_cache() -> Cache:
    """Default: `InMemoryCache` (in-process). Swap to `RedisCache` when
    `REDIS_URL` is set and the Redis implementation lands."""
    return InMemoryCache()


@lru_cache
def get_sandbox() -> Sandbox:
    """Default: `RaisingSandbox` — fails loudly when `compute` is called.

    This is deliberate per CLAUDE.md: never run LLM-generated code in-process.
    Swap to `E2BSandbox` (preferred) or `DockerSandbox` (fallback) once the
    corresponding credentials / Docker socket are configured.
    """
    return RaisingSandbox()


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Default: `FakeProvider` returning a generic placeholder string.

    This means `/ask` runs end-to-end but produces canned text. Replace with
    a real provider implementation (Anthropic / Google / OpenAI-compatible)
    once you've picked one and added its API key to `.env`.

    Note: until per-task provider routing is plumbed end-to-end, the single
    provider returned here handles every task. Production wire-up will return
    a `dict[str, LLMProvider]` and the agent loop will look up the right
    provider based on `ModelSpec.provider`.
    """
    return FakeProvider(
        "Placeholder LLM response. Wire a real provider in app/dependencies.py "
        "(see SETUP.md)."
    )
