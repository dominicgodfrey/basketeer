"""`write` primitive: narrative synthesis of the agent's scratchpad.

This is the only primitive whose job is human-readable prose. Everything else
returns structured data. It's a separate primitive (rather than just another
LLM call inside the agent loop) because the prompt and decoding settings differ:
planning calls want short structured tool calls; this wants longer, considered
prose.

The primitive takes a structured `WriteContext`, not free-form text. The agent
assembles the context from prior tool results before calling this primitive.
"""

import json

from pydantic import BaseModel, Field

from app.agents.prompts import load_prompt
from app.llm.providers import CompletionRequest, LLMProvider, Message
from app.llm.router import ModelSpec

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.5


class WriteContext(BaseModel):
    """Structured input to the write primitive.

    The agent fills these fields from accumulated tool results. Free-form prose
    is intentionally not accepted — keeping structure forces the agent to
    assemble specific findings rather than dumping the scratchpad.
    """

    question: str = Field(description="The user's original question, verbatim.")
    findings: list[str] = Field(
        default_factory=list,
        description="Discrete facts the agent established. One bullet per finding.",
    )
    data: dict | None = Field(
        default=None,
        description="Optional structured supporting data (records, summary stats).",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Caveats or framing the answer must respect (e.g. 'only post-2010', "
        "'small sample for Cooper Flagg').",
    )


class WriteResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def write(
    context: WriteContext,
    provider: LLMProvider,
    model_spec: ModelSpec,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> WriteResponse:
    """Generate the narrative answer for `context` and return it as prose."""
    system_prompt = load_prompt("write")
    user_message = _format_user_message(context)

    request = CompletionRequest(
        model=model_spec.model_id,
        messages=[
            Message(
                role="system",
                content=system_prompt,
                cache=model_spec.supports_prompt_caching,
            ),
            Message(role="user", content=user_message),
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = provider.complete(request)
    return WriteResponse(
        text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_read_input_tokens=response.cache_read_input_tokens,
        cache_creation_input_tokens=response.cache_creation_input_tokens,
    )


def _format_user_message(context: WriteContext) -> str:
    parts: list[str] = [f"Question: {context.question}"]

    if context.findings:
        findings_block = "\n".join(f"- {f}" for f in context.findings)
        parts.append(f"Findings:\n{findings_block}")

    if context.data:
        parts.append(f"Supporting data:\n{json.dumps(context.data, indent=2, default=str)}")

    if context.constraints:
        constraints_block = "\n".join(f"- {c}" for c in context.constraints)
        parts.append(f"Constraints:\n{constraints_block}")

    return "\n\n".join(parts)
