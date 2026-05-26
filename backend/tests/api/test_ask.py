"""End-to-end test of POST /ask using FastAPI's dependency_overrides.

We override the LLM provider with a scripted FakeProvider that:
1. Returns a classifier JSON response
2. Triggers a `write` tool call
3. Returns a final narrative from the write provider call

This proves the wiring works: api → dependencies → classifier → agent → tools → response.
"""

import json

from fastapi.testclient import TestClient

from app.dependencies import get_llm_provider, get_vector_store
from app.llm.providers import CompletionResponse, FakeProvider, ToolCall
from app.main import app
from app.similarity import InMemoryVectorStore, VectorRecord


def _scripted_provider() -> FakeProvider:
    """Returns: (1) classifier JSON, (2) tool_use → write, (3) write text."""
    return FakeProvider(
        [
            # 1) classifier call
            json.dumps(
                {
                    "path": "trivial",
                    "primitive": "find_similar",
                    "entities": {"player_name": "Klay Thompson"},
                    "confidence": 0.9,
                }
            ),
            # 2) agent first iteration: call write directly
            CompletionResponse(
                text="",
                input_tokens=10,
                output_tokens=5,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="write",
                        arguments={"question": "comps for klay"},
                    )
                ],
            ),
            # 3) write primitive's internal LLM call
            "Klay's best comps are Booker and Middleton.",
        ]
    )


def _populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="klay", vector=[1.0, 0.0], metadata={"position": "SG"}),
            VectorRecord(id="booker", vector=[0.95, 0.05], metadata={"position": "SG"}),
        ]
    )
    return store


def test_ask_runs_end_to_end_with_scripted_provider() -> None:
    provider = _scripted_provider()
    store = _populated_store()
    app.dependency_overrides[get_llm_provider] = lambda: provider
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        client = TestClient(app)
        response = client.post("/ask", json={"question": "comps for klay"})
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "Klay's best comps are Booker and Middleton."
        assert body["partial"] is False
        assert body["classifier_path"] == "trivial"
        assert body["classifier_primitive"] == "find_similar"
        assert body["classifier_confidence"] == 0.9
        assert body["iterations"] >= 1
        assert body["tokens_used"] > 0
        assert any(t.get("event") == "tool_call" for t in body["trace"])
    finally:
        app.dependency_overrides.clear()


def test_ask_rejects_empty_question() -> None:
    client = TestClient(app)
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_oversized_question() -> None:
    client = TestClient(app)
    response = client.post("/ask", json={"question": "x" * 5000})
    assert response.status_code == 422


def test_ask_with_classifier_agent_path() -> None:
    """When classifier picks 'agent', response reflects that."""
    provider = FakeProvider(
        [
            json.dumps(
                {"path": "agent", "primitive": None, "entities": {}, "confidence": 0.7}
            ),
            CompletionResponse(
                text="",
                input_tokens=5,
                output_tokens=5,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="write", arguments={"question": "?"})
                ],
            ),
            "Complex analysis follows...",
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: provider
    try:
        client = TestClient(app)
        body = client.post("/ask", json={"question": "complex query"}).json()
        assert body["classifier_path"] == "agent"
        assert body["classifier_primitive"] is None
        assert body["text"] == "Complex analysis follows..."
    finally:
        app.dependency_overrides.clear()
