"""Per-task LLM model routing.

The agent and primitives never hardcode model IDs. They ask the router for the
model assigned to a given Task; the router returns a ModelSpec describing which
provider to use and the provider-specific model id. Changing the routing
(e.g. escalating narrative writes to Sonnet) is a one-line edit here.

See CLAUDE.md "Model routing" for the rationale behind each assignment.
"""

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
    provider: str
    model_id: str
    supports_prompt_caching: bool = False


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
