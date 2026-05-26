"""`POST /ask` — the unified entrypoint for the basketeer scouting agent.

Wires the classifier + agent loop together over the four primitives (find_similar,
write, compute; query_stats lands when the collaborator's DB foundation is in).
The classifier runs first for observability — its result is included in the
response — but every query currently goes through the agent loop until the
entity-resolution layer that turns "Klay Thompson" → vector-store ID is built.
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.agents import (
    classify,
    make_compute_tool,
    make_find_similar_tool,
    make_write_tool,
    run_agent,
)
from app.cache import Cache
from app.dependencies import (
    get_cache,
    get_llm_provider,
    get_sandbox,
    get_vector_store,
)
from app.llm.providers import LLMProvider
from app.llm.router import Task, model_for
from app.logging_setup import get_logger
from app.sandbox import Sandbox
from app.similarity import VectorStore

logger = get_logger(__name__)

router = APIRouter(tags=["ask"])

MAX_QUESTION_CHARS = 4000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    text: str
    partial: bool
    iterations: int
    tokens_used: int
    wall_clock_seconds: float
    classifier_path: str
    classifier_primitive: str | None
    classifier_confidence: float
    trace: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    store: VectorStore = Depends(get_vector_store),
    cache: Cache = Depends(get_cache),
    provider: LLMProvider = Depends(get_llm_provider),
    sandbox: Sandbox = Depends(get_sandbox),
) -> AskResponse:
    logger.info("ask.received chars=%d", len(request.question))

    classifier_spec = model_for(Task.INTENT_CLASSIFIER)
    classification = classify(request.question, provider, classifier_spec)

    write_spec = model_for(Task.NARRATIVE_WRITE)
    planning_spec = model_for(Task.AGENT_PLANNING)

    tools = [
        make_find_similar_tool(store, cache=cache),
        make_write_tool(provider, write_spec),
        make_compute_tool(sandbox),
    ]

    result = run_agent(
        user_message=request.question,
        tools=tools,
        provider=provider,
        model_spec=planning_spec,
    )

    return AskResponse(
        text=result.text,
        partial=result.partial,
        iterations=result.iterations,
        tokens_used=result.tokens_used,
        wall_clock_seconds=result.wall_clock_seconds,
        classifier_path=classification.path,
        classifier_primitive=classification.primitive,
        classifier_confidence=classification.confidence,
        trace=result.trace,
    )
