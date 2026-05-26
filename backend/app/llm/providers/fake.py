"""In-memory LLM provider for tests.

Two usage modes:

1. Plain string responses (existing pattern):
       FakeProvider("hi")            — always returns "hi"
       FakeProvider(["a", "b"])      — pops one per call

2. Scripted multi-turn responses (for testing the agent loop):
       FakeProvider([
           CompletionResponse(text="", tool_calls=[ToolCall(...)], stop_reason="tool_use"),
           CompletionResponse(text="here's the answer", stop_reason="end_turn"),
       ])

Mixing strings and CompletionResponses in the queue is allowed; strings get
wrapped in a minimal CompletionResponse.
"""

from collections import deque
from collections.abc import Iterable
from typing import Any

from app.llm.providers.base import CompletionRequest, CompletionResponse


class FakeProvider:
    """Test double satisfying the LLMProvider protocol."""

    def __init__(
        self,
        responses: str | Iterable[str | CompletionResponse] = "fake response",
    ) -> None:
        if isinstance(responses, str):
            self._queue: deque[str | CompletionResponse] | None = None
            self._fixed: str | None = responses
        else:
            self._queue = deque(responses)
            self._fixed = None
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self._fixed is not None:
            return _wrap(self._fixed, request)
        assert self._queue is not None
        if not self._queue:
            raise StopIteration("FakeProvider has no more queued responses")
        item = self._queue.popleft()
        if isinstance(item, CompletionResponse):
            return item
        return _wrap(item, request)


def _wrap(text: str, request: CompletionRequest) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        input_tokens=_estimate_tokens(request),
        output_tokens=_estimate_text_tokens(text),
        stop_reason="end_turn",
    )


def _estimate_tokens(request: CompletionRequest) -> int:
    return sum(_estimate_text_tokens(m.content) for m in request.messages)


def _estimate_text_tokens(text: Any) -> int:
    """Rough 4-chars-per-token approximation for fake accounting."""
    if not text:
        return 1
    return max(1, len(str(text)) // 4)
