"""Provider-agnostic LLM call abstractions.

Each provider implementation (AnthropicProvider, GoogleProvider, OpenAICompatible
Provider, FakeProvider) satisfies the LLMProvider protocol. The agent and
primitives never import a provider SDK directly — they receive an LLMProvider
via dependency injection.

Tool calls cross this boundary in a unified `ToolCall` shape: each provider
adapter translates its native tool-call format (Anthropic content blocks,
OpenAI `tool_calls`, Google `function_call` parts) to/from `ToolCall`.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class ToolCall:
    """One tool invocation requested by the LLM.

    `id` is the provider-assigned identifier used to correlate the tool result
    back into the conversation. `arguments` is already-parsed JSON (a dict) —
    the provider adapter handles JSON decoding so callers never deal with raw
    JSON strings.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    """A single chat message.

    Roles:
    - `system`: instructions; typically the first message.
    - `user`: the user's query.
    - `assistant`: model output. May carry `tool_calls` instead of (or alongside)
      `content` text.
    - `tool`: a tool result message. Must set `tool_call_id` to the id of the
      ToolCall it satisfies.

    `cache=True` marks the message as a prompt-caching breakpoint for providers
    that require explicit markers (Anthropic). Ignored elsewhere.
    """

    role: str
    content: str
    cache: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class CompletionRequest:
    """One LLM call.

    `tools` is the provider-native tools spec (already translated from
    ToolSpec via `to_anthropic_tools` / `to_openai_tools` / `to_google_tools`
    at the agent layer). `tool_choice` controls forcing:
    - None / "auto": model picks whether to call tools.
    - "required": model MUST call at least one tool.
    - "<name>": model MUST call the named tool.
    """

    model: str
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | None = None


@dataclass(slots=True)
class CompletionResponse:
    """The result of a completion call.

    `text` is the model's prose output (may be empty if the model only called
    tools). `tool_calls` is non-empty when the model requested tool execution.
    Both can be present in a single response — Anthropic in particular often
    interleaves reasoning text with tool calls.

    `stop_reason` is the model's reported stop condition (`end_turn`,
    `tool_use`, `max_tokens`, etc.). Use this rather than inferring from
    `tool_calls` emptiness when possible.

    Cache-token counters (Anthropic prompt caching) populate when available.
    Always log these on the agent loop to verify caching fires.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    """Synchronous completion interface. Async variants can be added when needed."""

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run a completion. Should raise on transport / auth errors; should NOT
        retry internally — retries are the caller's policy decision."""
        ...
