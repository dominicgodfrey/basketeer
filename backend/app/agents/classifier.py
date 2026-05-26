"""Intent classifier: trivial single-primitive call vs full agent loop.

This is the fast-path gate from PLAN.md Phase 5. A single classification call
(~10x cheaper than a Haiku planning iteration) routes obvious queries directly
to one primitive, skipping the full ReAct loop. Misroutes are benign — the
agent loop can always answer; we just lose the cost saving on those.

The classifier returns a structured `ClassifierResult`. Parsing failures or
schema violations fall back to `path="agent"` so the system never blocks on
classifier flakiness.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agents.prompts import load_prompt
from app.llm.providers import CompletionRequest, LLMProvider, Message
from app.llm.router import ModelSpec
from app.logging_setup import get_logger

logger = get_logger(__name__)

MAX_RESPONSE_TOKENS = 256
CLASSIFIER_TEMPERATURE = 0.0


class ClassifierResult(BaseModel):
    """Classifier output.

    `path="trivial"` requires a `primitive` and typically extracts `entities`.
    `path="agent"` ignores both.
    """

    path: Literal["trivial", "agent"]
    primitive: Literal["find_similar", "query_stats"] | None = None
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _trivial_requires_primitive(self) -> "ClassifierResult":
        if self.path == "trivial" and self.primitive is None:
            raise ValueError("trivial path requires a primitive")
        if self.path == "agent" and self.primitive is not None:
            # Be lenient: clear the field rather than reject.
            object.__setattr__(self, "primitive", None)
        return self


def classify(
    query: str,
    provider: LLMProvider,
    model_spec: ModelSpec,
) -> ClassifierResult:
    """Run the classifier on `query`. Falls back to `path="agent"` on any
    parsing or validation error."""
    request = CompletionRequest(
        model=model_spec.model_id,
        messages=[
            Message(
                role="system",
                content=load_prompt("classifier"),
                cache=model_spec.supports_prompt_caching,
            ),
            Message(role="user", content=query),
        ],
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=CLASSIFIER_TEMPERATURE,
    )
    response = provider.complete(request)

    extracted = _strip_markdown_fences(response.text)
    try:
        data = json.loads(extracted)
        result = ClassifierResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(
            "classifier.parse_failed err=%s response_head=%r",
            e.__class__.__name__,
            response.text[:120],
        )
        return ClassifierResult(path="agent")

    logger.info(
        "classifier.result path=%s primitive=%s confidence=%.2f",
        result.path,
        result.primitive,
        result.confidence,
    )
    return result


def _strip_markdown_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite the prompt telling them not to.
    Strip the outer fences so `json.loads` succeeds anyway."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    # Drop the first line (```json or just ```)
    newline = text.find("\n")
    if newline == -1:
        return text
    body = text[newline + 1 :].strip()
    if body.endswith("```"):
        body = body[:-3].strip()
    return body
