import json

from app.agents import classify
from app.agents.classifier import ClassifierResult
from app.llm.providers import FakeProvider
from app.llm.router import Task, model_for


def _spec():
    return model_for(Task.INTENT_CLASSIFIER)


def test_parses_clean_trivial_response() -> None:
    response = json.dumps(
        {
            "path": "trivial",
            "primitive": "find_similar",
            "entities": {"player_name": "Klay Thompson"},
            "confidence": 0.9,
        }
    )
    result = classify("comps for klay", FakeProvider(response), _spec())
    assert result.path == "trivial"
    assert result.primitive == "find_similar"
    assert result.entities["player_name"] == "Klay Thompson"
    assert result.confidence == 0.9


def test_parses_clean_agent_response() -> None:
    response = json.dumps(
        {
            "path": "agent",
            "primitive": None,
            "entities": {},
            "confidence": 0.85,
        }
    )
    result = classify("complex query", FakeProvider(response), _spec())
    assert result.path == "agent"
    assert result.primitive is None


def test_strips_markdown_fences() -> None:
    wrapped = "```json\n" + json.dumps(
        {"path": "trivial", "primitive": "find_similar", "entities": {}, "confidence": 1.0}
    ) + "\n```"
    result = classify("comps", FakeProvider(wrapped), _spec())
    assert result.path == "trivial"


def test_strips_plain_fences() -> None:
    wrapped = "```\n" + json.dumps(
        {"path": "agent", "primitive": None, "entities": {}, "confidence": 0.5}
    ) + "\n```"
    result = classify("complex", FakeProvider(wrapped), _spec())
    assert result.path == "agent"


def test_falls_back_to_agent_on_invalid_json() -> None:
    result = classify("anything", FakeProvider("this is not json at all"), _spec())
    assert result.path == "agent"
    assert result.primitive is None


def test_falls_back_to_agent_on_schema_violation() -> None:
    """trivial path without primitive should fail validation → fallback to agent."""
    response = json.dumps({"path": "trivial", "primitive": None, "entities": {}, "confidence": 1.0})
    result = classify("ambiguous", FakeProvider(response), _spec())
    assert result.path == "agent"


def test_lenient_about_agent_with_lingering_primitive() -> None:
    """If LLM sets primitive even on agent path, we accept and clear it."""
    response = json.dumps(
        {
            "path": "agent",
            "primitive": "find_similar",  # contradictory
            "entities": {},
            "confidence": 0.5,
        }
    )
    result = classify("anything", FakeProvider(response), _spec())
    assert result.path == "agent"
    assert result.primitive is None


def test_confidence_clamped_in_validation() -> None:
    """Out-of-bounds confidence falls back to agent path."""
    response = json.dumps(
        {"path": "trivial", "primitive": "find_similar", "entities": {}, "confidence": 1.5}
    )
    result = classify("?", FakeProvider(response), _spec())
    assert result.path == "agent"  # validation failed → fallback


def test_classifier_result_validator_directly() -> None:
    """Direct ClassifierResult validation path (no LLM)."""
    r = ClassifierResult(
        path="trivial",
        primitive="query_stats",
        entities={"stat": "PER"},
        confidence=0.8,
    )
    assert r.primitive == "query_stats"
