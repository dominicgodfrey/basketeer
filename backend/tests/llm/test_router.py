import pytest

from app.llm import DEFAULT_ROUTING, ModelSpec, Task, get_provider, model_for
from app.llm.providers import FakeProvider, LLMProvider


def test_default_routing_covers_every_task() -> None:
    for task in Task:
        assert task in DEFAULT_ROUTING, f"{task} has no default model"


def test_planning_uses_anthropic_with_prompt_caching() -> None:
    spec = model_for(Task.AGENT_PLANNING)
    assert spec.provider == "anthropic"
    assert spec.supports_prompt_caching is True


def test_classifier_routed_to_google_flash_lite() -> None:
    spec = model_for(Task.INTENT_CLASSIFIER)
    assert spec.provider == "google"
    assert "flash-lite" in spec.model_id


def test_text_to_sql_routed_to_google_flash() -> None:
    spec = model_for(Task.TEXT_TO_SQL)
    assert spec.provider == "google"
    assert spec.model_id == "gemini-2.5-flash"


def test_model_for_with_custom_routing() -> None:
    custom = {Task.NARRATIVE_WRITE: ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6")}
    spec = model_for(Task.NARRATIVE_WRITE, routing=custom)
    assert spec.model_id == "claude-sonnet-4-6"


def test_model_for_raises_on_missing_task() -> None:
    with pytest.raises(KeyError, match="No model assigned"):
        model_for(Task.AGENT_PLANNING, routing={})


def test_get_provider_returns_registered_implementation() -> None:
    fake = FakeProvider()
    providers: dict[str, LLMProvider] = {"anthropic": fake}
    spec = model_for(Task.AGENT_PLANNING)
    assert get_provider(spec, providers) is fake


def test_get_provider_raises_for_unregistered_provider() -> None:
    with pytest.raises(KeyError, match="No provider registered"):
        get_provider(ModelSpec(provider="cohere", model_id="x"), providers={})


def test_modelspec_is_hashable_and_frozen() -> None:
    a = ModelSpec(provider="anthropic", model_id="x")
    b = ModelSpec(provider="anthropic", model_id="x")
    assert hash(a) == hash(b)
    with pytest.raises(Exception):
        a.provider = "google"  # type: ignore[misc]
