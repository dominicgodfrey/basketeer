"""Tool definitions for the agent.

A `ToolSpec` is provider-agnostic: name + description + Pydantic args schema +
an invoke function. Translators (`to_anthropic_tools`, `to_openai_tools`,
`to_google_tools`) convert a list of ToolSpecs to the JSON shape each provider
expects. The agent loop dispatches tool calls by name, validates the raw args
through the Pydantic schema, then calls `invoke` with the validated instance.

This is the pattern CLAUDE.md calls "tool decorator pattern, no higher-level
LangChain abstractions" — we implement it directly with stdlib + Pydantic so
the system stays portable across Anthropic / Google / openai-compatible
providers without dragging in LangChain's adapters.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool the agent can call.

    `name`: the identifier the LLM uses to invoke. Must match `[a-zA-Z0-9_]+`.
    `description`: the agent's API contract. Be specific about when to call,
        when *not* to call, and the meaning of each argument. The LLM reads
        this; precision here pays off across thousands of calls.
    `args_schema`: a Pydantic BaseModel describing the tool's arguments.
        The translator emits its JSON Schema; the dispatcher validates the
        LLM's raw args against it.
    `invoke`: callable taking a validated args instance and returning a
        JSON-serializable result (dict, list, scalar, str).
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    invoke: Callable[[BaseModel], Any]


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Translate to Anthropic Messages API tools format.

    Shape per https://docs.anthropic.com/en/api/messages tools:
        {"name": ..., "description": ..., "input_schema": <JSON Schema>}
    """
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.args_schema.model_json_schema(),
        }
        for t in tools
    ]


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Translate to OpenAI Chat Completions tools format.

    DeepSeek, Together, Groq, Fireworks, Moonshot, and most "openai-compatible"
    providers consume this exact shape.

    Shape per https://platform.openai.com/docs/api-reference/chat/create:
        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema(),
            },
        }
        for t in tools
    ]


def to_google_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Translate to Google GenerativeAI function-calling format.

    Google accepts a subset of full JSON Schema in `parameters`. We pass the
    Pydantic-generated schema through; the Google provider implementation
    will strip unsupported keys (`$defs`, `title`, `additionalProperties`)
    when it's built.

    Shape per https://ai.google.dev/gemini-api/docs/function-calling:
        {"function_declarations": [{"name": ..., "description": ..., "parameters": <schema>}]}
    """
    return [
        {
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema(),
                }
            ]
        }
        for t in tools
    ]


def find_tool(tools: list[ToolSpec], name: str) -> ToolSpec | None:
    """Look up a ToolSpec by name. Returns None if absent."""
    for t in tools:
        if t.name == name:
            return t
    return None
