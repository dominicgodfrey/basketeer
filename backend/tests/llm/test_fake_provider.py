import pytest

from app.llm.providers import CompletionRequest, FakeProvider, LLMProvider, Message


def test_fake_provider_satisfies_protocol() -> None:
    assert isinstance(FakeProvider(), LLMProvider)


def test_fixed_response_returned_repeatedly() -> None:
    provider = FakeProvider("hello")
    request = CompletionRequest(model="x", messages=[Message(role="user", content="hi")])
    for _ in range(3):
        assert provider.complete(request).text == "hello"
    assert len(provider.requests) == 3


def test_queued_responses_pop_in_order() -> None:
    provider = FakeProvider(["one", "two", "three"])
    request = CompletionRequest(model="x", messages=[Message(role="user", content="hi")])
    assert [provider.complete(request).text for _ in range(3)] == ["one", "two", "three"]


def test_queued_responses_exhausted_raises() -> None:
    provider = FakeProvider(["only-one"])
    request = CompletionRequest(model="x", messages=[Message(role="user", content="hi")])
    provider.complete(request)
    with pytest.raises(StopIteration):
        provider.complete(request)


def test_token_estimates_populated() -> None:
    provider = FakeProvider("a longer response with some content")
    request = CompletionRequest(
        model="x",
        messages=[Message(role="user", content="some prompt text here")],
    )
    response = provider.complete(request)
    assert response.input_tokens >= 1
    assert response.output_tokens >= 1


def test_cache_flag_round_trips_in_request_log() -> None:
    provider = FakeProvider()
    request = CompletionRequest(
        model="x",
        messages=[
            Message(role="system", content="big system prompt", cache=True),
            Message(role="user", content="question"),
        ],
    )
    provider.complete(request)
    recorded = provider.requests[0]
    assert recorded.messages[0].cache is True
    assert recorded.messages[1].cache is False


def test_scripted_completion_response_returned_as_is() -> None:
    """Queued CompletionResponse instances should be returned without re-wrapping
    so tests can specify tool_calls and stop_reason."""
    from app.llm.providers import CompletionResponse, ToolCall

    scripted = CompletionResponse(
        text="",
        input_tokens=42,
        output_tokens=7,
        stop_reason="tool_use",
        tool_calls=[ToolCall(id="t1", name="find_similar", arguments={"player_id": "x"})],
    )
    provider = FakeProvider([scripted])
    request = CompletionRequest(model="x", messages=[Message(role="user", content="hi")])
    response = provider.complete(request)
    assert response is scripted
    assert response.tool_calls[0].name == "find_similar"


def test_mixed_string_and_response_queue() -> None:
    from app.llm.providers import CompletionResponse, ToolCall

    provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                stop_reason="tool_use",
                tool_calls=[ToolCall(id="t1", name="write", arguments={"question": "?"})],
            ),
            "final-text",
        ]
    )
    request = CompletionRequest(model="x", messages=[Message(role="user", content="hi")])
    first = provider.complete(request)
    second = provider.complete(request)
    assert first.stop_reason == "tool_use"
    assert second.text == "final-text"
    assert second.stop_reason == "end_turn"
