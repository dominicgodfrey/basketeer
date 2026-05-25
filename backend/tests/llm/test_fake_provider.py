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
