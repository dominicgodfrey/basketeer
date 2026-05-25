"""In-memory LLM provider for tests.

Records every request so tests can assert on what the agent or a primitive
actually sent. Responses are configurable: either a fixed string, or a queue of
responses for multi-turn tests.
"""

from collections import deque
from collections.abc import Iterable

from app.llm.providers.base import CompletionRequest, CompletionResponse


class FakeProvider:
    """Test double satisfying the LLMProvider protocol.

    `responses` is an iterable of response texts; the provider pops one per call
    and raises `StopIteration` if exhausted. Pass a single string to return the
    same text on every call.
    """

    def __init__(self, responses: str | Iterable[str] = "fake response") -> None:
        if isinstance(responses, str):
            self._responses: deque[str] | None = None
            self._fixed_response: str | None = responses
        else:
            self._responses = deque(responses)
            self._fixed_response = None
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self._fixed_response is not None:
            text = self._fixed_response
        else:
            assert self._responses is not None
            if not self._responses:
                raise StopIteration("FakeProvider has no more queued responses")
            text = self._responses.popleft()
        return CompletionResponse(
            text=text,
            input_tokens=_estimate_tokens(request),
            output_tokens=_estimate_text_tokens(text),
        )


def _estimate_tokens(request: CompletionRequest) -> int:
    return sum(_estimate_text_tokens(m.content) for m in request.messages)


def _estimate_text_tokens(text: str) -> int:
    """Rough 4-chars-per-token approximation for fake accounting."""
    return max(1, len(text) // 4)
