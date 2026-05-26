"""Per-task LLM model routing.

The agent and primitives never hardcode model IDs. They ask the router for the
model assigned to a given Task; the router returns a ModelSpec describing which
provider to use and the provider-specific model id. Changing the routing
(e.g. escalating narrative writes to Sonnet, swapping all tasks to DeepSeek)
is either a one-line edit in DEFAULT_ROUTING or — preferred — an environment
variable override read by `routing_from_env`.

The `provider` field is intentionally a free-form string so any new provider
plugs in by registering an LLMProvider implementation under a matching key.
Three known provider tokens are currently in use:

- `anthropic` — Claude family; uses the Anthropic SDK; supports explicit
  prompt-caching breakpoints.
- `google` — Gemini family; uses the Google AI SDK.
- `openai-compatible` — anything exposing the OpenAI Chat Completions API
  shape: DeepSeek, Together, Groq, Fireworks, Moonshot, most local llama.cpp
  servers. The base URL is configured per-provider-instance, not per-call.

See CLAUDE.md "Model routing" for the rationale behind each task assignment.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from app.llm.providers.base import LLMProvider


class Task(str, Enum):
    """All distinct LLM call-sites in the system. Add a new entry when you add a
    new place that calls an LLM; do not reuse an existing task for an unrelated
    purpose just because the assigned model happens to match."""

    INTENT_CLASSIFIER = "intent_classifier"
    AGENT_PLANNING = "agent_planning"
    TEXT_TO_SQL = "text_to_sql"
    CODE_GENERATION = "code_generation"
    NARRATIVE_WRITE = "narrative_write"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model assignment.

    `supports_prompt_caching` specifically means the provider's SDK requires
    explicit `cache_control` breakpoints to enable caching (currently only
    Anthropic). Providers that auto-cache (Gemini, OpenAI-compatible endpoints
    like DeepSeek) should leave this False; their caching happens server-side
    regardless of the flag.
    """

    provider: str
    model_id: str
    supports_prompt_caching: bool = False


# Defaults — PLACEHOLDERS until a provider decision is made. Override per task
# via `MODEL_<TASK_NAME>` env vars (see `.env.example`) so swapping providers
# never requires a code change.
DEFAULT_ROUTING: dict[Task, ModelSpec] = {
    Task.INTENT_CLASSIFIER: ModelSpec(
        provider="google",
        model_id="gemini-2.5-flash-lite",
    ),
    Task.AGENT_PLANNING: ModelSpec(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        supports_prompt_caching=True,
    ),
    Task.TEXT_TO_SQL: ModelSpec(
        provider="google",
        model_id="gemini-2.5-flash",
    ),
    Task.CODE_GENERATION: ModelSpec(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        supports_prompt_caching=True,
    ),
    Task.NARRATIVE_WRITE: ModelSpec(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        supports_prompt_caching=True,
    ),
}


def model_for(task: Task, routing: dict[Task, ModelSpec] | None = None) -> ModelSpec:
    """Return the ModelSpec assigned to `task` in the given routing table.

    Pass a custom `routing` for tests or experiments; defaults to DEFAULT_ROUTING.
    """
    table = routing if routing is not None else DEFAULT_ROUTING
    if task not in table:
        raise KeyError(f"No model assigned to task {task!r}")
    return table[task]


def get_provider(spec: ModelSpec, providers: dict[str, LLMProvider]) -> LLMProvider:
    """Look up the LLMProvider implementation for a given ModelSpec."""
    if spec.provider not in providers:
        raise KeyError(
            f"No provider registered for {spec.provider!r}; "
            f"known: {sorted(providers.keys())}"
        )
    return providers[spec.provider]


def routing_from_env(env: Mapping[str, str] | None = None) -> dict[Task, ModelSpec]:
    """Build a routing table from `MODEL_<TASK>` env vars, falling back to DEFAULT_ROUTING.

    Format: `MODEL_<TASK_NAME>=<provider>:<model_id>[:cache]`. Append `:cache`
    only for providers that require explicit cache breakpoints (Anthropic).
    Example:

        MODEL_AGENT_PLANNING=anthropic:claude-haiku-4-5-20251001:cache
        MODEL_INTENT_CLASSIFIER=openai-compatible:deepseek-chat
        MODEL_NARRATIVE_WRITE=google:gemini-2.5-flash

    Tasks without an env override use the DEFAULT_ROUTING entry. Pass `env`
    explicitly in tests; defaults to `os.environ` in production.
    """
    source = env if env is not None else os.environ
    routing = dict(DEFAULT_ROUTING)
    for task in Task:
        key = f"MODEL_{task.name}"
        value = source.get(key)
        if value:
            routing[task] = _parse_model_spec(value)
    return routing


def _parse_model_spec(value: str) -> ModelSpec:
    if ":" not in value:
        raise ValueError(
            f"Bad model spec {value!r}; expected 'provider:model_id[:cache]'"
        )
    provider, rest = value.split(":", 1)
    cache_suffix = ":cache"
    if rest.endswith(cache_suffix):
        model_id = rest[: -len(cache_suffix)]
        supports_caching = True
    else:
        model_id = rest
        supports_caching = False
    if not provider or not model_id:
        raise ValueError(
            f"Bad model spec {value!r}; provider and model_id must be non-empty"
        )
    return ModelSpec(
        provider=provider,
        model_id=model_id,
        supports_prompt_caching=supports_caching,
    )
