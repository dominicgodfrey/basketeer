"""Hand-rolled ReAct agent loop.

Inputs: a user message, a list of ToolSpecs, an LLMProvider, a ModelSpec.
Outputs: the final narrative text plus diagnostic metadata (iteration count,
tokens used, wall-clock seconds, trace events).

Guardrails per PLAN.md:
- Max 6 iterations (configurable)
- Max 30 s wall-clock (configurable)
- Max 30K tokens total (configurable)

When any limit is hit the loop ends with `partial=True`. If `write` was reached
before the limit, its output is the final text; otherwise `text` is empty and
the caller decides how to render the partial state.

No LangChain dependency. Tool formats are translated per-provider via the
registry in `_TOOL_FORMATTERS`; add new providers there.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.agents.prompts import load_prompt
from app.agents.tools import (
    ToolSpec,
    find_tool,
    to_anthropic_tools,
    to_google_tools,
    to_openai_tools,
)
from app.llm.providers import (
    CompletionRequest,
    LLMProvider,
    Message,
    ToolCall,
)
from app.llm.router import ModelSpec
from app.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MAX_WALL_SECONDS = 30.0
DEFAULT_MAX_TOKENS_TOTAL = 30_000
DEFAULT_MAX_RESPONSE_TOKENS = 2048

_TOOL_FORMATTERS: dict[str, Callable[[list[ToolSpec]], list[dict[str, Any]]]] = {
    "anthropic": to_anthropic_tools,
    "google": to_google_tools,
    "openai-compatible": to_openai_tools,
}


@dataclass(slots=True)
class AgentResult:
    """One agent loop run.

    `text` is the final answer. When `partial` is True, the loop hit a guard
    rail before `write` produced its answer; `text` may be empty in that case.
    `trace` is an append-only event log useful for debugging / cost analysis.
    """

    text: str
    partial: bool
    iterations: int
    tokens_used: int
    wall_clock_seconds: float
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_agent(
    user_message: str,
    tools: list[ToolSpec],
    provider: LLMProvider,
    model_spec: ModelSpec,
    *,
    system_prompt: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
    max_tokens_total: int = DEFAULT_MAX_TOKENS_TOTAL,
    max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
    clock: Callable[[], float] = time.monotonic,
) -> AgentResult:
    """Run the ReAct loop until `write` completes or a guard rail trips."""
    system_text = system_prompt if system_prompt is not None else load_prompt("agent")
    provider_tools = _format_tools(model_spec, tools)

    messages: list[Message] = [Message(role="user", content=user_message)]
    trace: list[dict[str, Any]] = []
    tokens_used = 0
    final_text = ""
    partial = False
    start = clock()
    iteration = 0
    write_called = False

    while True:
        iteration += 1

        if iteration > max_iterations:
            trace.append({"event": "limit_iterations", "iteration": iteration - 1})
            partial = True
            break
        if clock() - start > max_wall_seconds:
            trace.append({"event": "limit_wall_clock", "seconds": clock() - start})
            partial = True
            break
        if tokens_used > max_tokens_total:
            trace.append({"event": "limit_tokens", "tokens": tokens_used})
            partial = True
            break

        request = CompletionRequest(
            model=model_spec.model_id,
            messages=[
                Message(
                    role="system",
                    content=system_text,
                    cache=model_spec.supports_prompt_caching,
                ),
                *messages,
            ],
            max_tokens=max_response_tokens,
            tools=provider_tools,
            tool_choice="auto" if provider_tools else None,
        )
        response = provider.complete(request)
        tokens_used += response.input_tokens + response.output_tokens

        logger.info(
            "agent.iter=%d model=%s in=%d out=%d cache_r=%d tool_calls=%d stop=%s",
            iteration,
            model_spec.model_id,
            response.input_tokens,
            response.output_tokens,
            response.cache_read_input_tokens,
            len(response.tool_calls),
            response.stop_reason,
        )

        if not response.tool_calls:
            # No more tool calls — the model is done. Use its prose as the answer
            # *unless* a previous `write` call set final_text.
            if not final_text:
                final_text = response.text
            trace.append({"event": "end_turn", "iteration": iteration})
            break

        # Persist the assistant turn (including the tool_calls) for context.
        messages.append(
            Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
            )
        )

        for tc in response.tool_calls:
            tool_result, ok = _execute_tool(tc, tools)
            trace.append(
                {
                    "event": "tool_call",
                    "iteration": iteration,
                    "name": tc.name,
                    "ok": ok,
                }
            )
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps(tool_result, default=str),
                    tool_call_id=tc.id,
                )
            )
            if tc.name == "write" and ok and isinstance(tool_result, dict) and "text" in tool_result:
                final_text = tool_result["text"]
                write_called = True

        if write_called:
            trace.append({"event": "write_complete", "iteration": iteration})
            break

    return AgentResult(
        text=final_text,
        partial=partial,
        iterations=iteration if not partial else iteration - 1,
        tokens_used=tokens_used,
        wall_clock_seconds=clock() - start,
        trace=trace,
    )


def _format_tools(spec: ModelSpec, tools: list[ToolSpec]) -> list[dict[str, Any]]:
    if not tools:
        return []
    formatter = _TOOL_FORMATTERS.get(spec.provider)
    if formatter is None:
        raise ValueError(
            f"No tool formatter registered for provider {spec.provider!r}. "
            f"Add one to app.agents.loop._TOOL_FORMATTERS."
        )
    return formatter(tools)


def _execute_tool(
    call: ToolCall, tools: list[ToolSpec]
) -> tuple[dict[str, Any], bool]:
    """Validate args and dispatch to the tool. Returns (result_dict, ok_flag).

    Failures are returned as structured error dicts the LLM can read and
    recover from — never raised. This is per CLAUDE.md: "structured errors
    the agent can read and recover from"."""
    spec = find_tool(tools, call.name)
    if spec is None:
        return {"error": "unknown_tool", "tool": call.name}, False
    try:
        validated = spec.args_schema.model_validate(call.arguments)
    except ValidationError as e:
        return {"error": "invalid_arguments", "tool": call.name, "detail": e.errors()}, False
    try:
        raw = spec.invoke(validated)
    except Exception as e:  # noqa: BLE001 — surface as structured error
        return {"error": "tool_exception", "tool": call.name, "type": type(e).__name__, "message": str(e)}, False
    if not isinstance(raw, dict):
        return {"result": raw}, True
    return raw, True
