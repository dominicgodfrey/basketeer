import json

from app.agents.prompts import load_prompt
from app.llm.providers import FakeProvider
from app.llm.router import ModelSpec, Task, model_for
from app.primitives import WriteContext, write


def _spec_with_caching() -> ModelSpec:
    return model_for(Task.NARRATIVE_WRITE)


def _spec_without_caching() -> ModelSpec:
    return ModelSpec(provider="google", model_id="gemini-2.5-flash", supports_prompt_caching=False)


def test_write_returns_provider_text() -> None:
    provider = FakeProvider("Klay's age-30 dropoff was steeper than most...")
    response = write(
        WriteContext(question="Who has the biggest age-30 dropoff?"),
        provider,
        _spec_with_caching(),
    )
    assert response.text.startswith("Klay's age-30 dropoff")
    assert response.input_tokens >= 1
    assert response.output_tokens >= 1


def test_write_uses_write_prompt_as_system_message() -> None:
    provider = FakeProvider()
    write(WriteContext(question="anything"), provider, _spec_with_caching())
    request = provider.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[0].content == load_prompt("write")


def test_cache_flag_set_when_model_supports_caching() -> None:
    provider = FakeProvider()
    write(WriteContext(question="anything"), provider, _spec_with_caching())
    assert provider.requests[0].messages[0].cache is True


def test_cache_flag_off_when_model_does_not_support_caching() -> None:
    provider = FakeProvider()
    write(WriteContext(question="anything"), provider, _spec_without_caching())
    assert provider.requests[0].messages[0].cache is False


def test_user_message_includes_findings_and_constraints() -> None:
    provider = FakeProvider()
    write(
        WriteContext(
            question="Most underpaid wings?",
            findings=["Player A on a min deal had 4.2 EPM", "Player B re-signed for $5M"],
            constraints=["wings only", "2024-25 season"],
        ),
        provider,
        _spec_with_caching(),
    )
    user_text = provider.requests[0].messages[1].content
    assert "Player A on a min deal" in user_text
    assert "wings only" in user_text


def test_user_message_includes_supporting_data_as_json() -> None:
    provider = FakeProvider()
    data = {"top_wing": {"name": "X", "epm": 4.2}}
    write(
        WriteContext(question="anything", data=data),
        provider,
        _spec_with_caching(),
    )
    user_text = provider.requests[0].messages[1].content
    assert "Supporting data" in user_text
    # The JSON block round-trips
    json_start = user_text.index("{")
    parsed = json.loads(user_text[json_start:])
    assert parsed == data


def test_user_message_omits_empty_sections() -> None:
    provider = FakeProvider()
    write(WriteContext(question="bare question"), provider, _spec_with_caching())
    user_text = provider.requests[0].messages[1].content
    assert user_text == "Question: bare question"


def test_token_counts_round_trip() -> None:
    provider = FakeProvider("a response")
    response = write(WriteContext(question="anything"), provider, _spec_with_caching())
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.cache_read_input_tokens == 0
    assert response.cache_creation_input_tokens == 0
