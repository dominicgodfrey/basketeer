"""Provider-agnostic LLM call abstractions.

Each provider implementation (AnthropicProvider, GoogleProvider, FakeProvider)
satisfies the LLMProvider protocol. The agent and primitives never import a
provider SDK directly — they receive an LLMProvider via dependency injection.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class Message:
    """A single chat message.

    `cache=True` marks the message as a prompt-caching breakpoint for providers
    that support it (Anthropic). On providers that don't, the flag is ignored.
    """

    role: str
    content: str
    cache: bool = False


@dataclass(slots=True)
class CompletionRequest:
    model: str
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompletionResponse:
    """The result of a completion call.

    Cache-token counters (Anthropic prompt caching) are populated when available
    and zero otherwise. Always log these on the agent loop to verify caching
    is actually firing — see CLAUDE.md "Caching" section.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    """Synchronous completion interface. Async variants can be added when needed."""

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run a completion. Should raise on transport / auth errors; should NOT
        retry internally — retries are the caller's policy decision."""
        ...
