import pytest

from app.llm import DEFAULT_ROUTING, ModelSpec, Task, get_provider, model_for, routing_from_env
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


def test_routing_from_env_empty_returns_defaults() -> None:
    routing = routing_from_env(env={})
    assert routing == DEFAULT_ROUTING


def test_routing_from_env_overrides_single_task() -> None:
    routing = routing_from_env(env={"MODEL_NARRATIVE_WRITE": "openai-compatible:deepseek-chat"})
    spec = routing[Task.NARRATIVE_WRITE]
    assert spec.provider == "openai-compatible"
    assert spec.model_id == "deepseek-chat"
    assert spec.supports_prompt_caching is False
    # Other tasks unchanged
    assert routing[Task.AGENT_PLANNING] == DEFAULT_ROUTING[Task.AGENT_PLANNING]


def test_routing_from_env_cache_suffix_sets_flag() -> None:
    routing = routing_from_env(
        env={"MODEL_AGENT_PLANNING": "anthropic:claude-haiku-4-5-20251001:cache"}
    )
    spec = routing[Task.AGENT_PLANNING]
    assert spec.supports_prompt_caching is True
    assert spec.model_id == "claude-haiku-4-5-20251001"


def test_routing_from_env_supports_model_id_with_colon() -> None:
    """Some bedrock-style ids contain colons (e.g. 'foo.bar:0'). Only the
    leading 'provider:' and trailing ':cache' are stripped."""
    routing = routing_from_env(env={"MODEL_TEXT_TO_SQL": "anthropic:claude.haiku-4-5:0"})
    spec = routing[Task.TEXT_TO_SQL]
    assert spec.model_id == "claude.haiku-4-5:0"
    assert spec.supports_prompt_caching is False


def test_routing_from_env_supports_complex_model_id_with_cache() -> None:
    routing = routing_from_env(env={"MODEL_TEXT_TO_SQL": "anthropic:claude.haiku-4-5:0:cache"})
    spec = routing[Task.TEXT_TO_SQL]
    assert spec.model_id == "claude.haiku-4-5:0"
    assert spec.supports_prompt_caching is True


def test_routing_from_env_ignores_unknown_keys() -> None:
    routing = routing_from_env(
        env={
            "MODEL_DOES_NOT_EXIST": "anthropic:whatever",
            "UNRELATED_VAR": "skipped",
        }
    )
    assert routing == DEFAULT_ROUTING


def test_routing_from_env_bad_value_raises() -> None:
    with pytest.raises(ValueError, match="provider:model_id"):
        routing_from_env(env={"MODEL_AGENT_PLANNING": "no-colon-here"})


def test_routing_from_env_empty_components_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        routing_from_env(env={"MODEL_AGENT_PLANNING": ":model-only"})
    with pytest.raises(ValueError, match="non-empty"):
        routing_from_env(env={"MODEL_AGENT_PLANNING": "provider-only:"})
