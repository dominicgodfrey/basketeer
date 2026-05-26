"""Agent loop integration tests, all driven by FakeProvider + FakeSandbox."""

from typing import Any

from app.agents import (
    make_compute_tool,
    make_find_similar_tool,
    make_write_tool,
    run_agent,
)
from app.llm.providers import CompletionResponse, FakeProvider, ToolCall
from app.llm.router import ModelSpec, Task, model_for
from app.sandbox import FakeSandbox
from app.similarity import InMemoryVectorStore, VectorRecord


def _seed_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="klay", vector=[1.0, 0.0], metadata={"position": "SG"}),
            VectorRecord(id="booker", vector=[0.95, 0.05], metadata={"position": "SG"}),
            VectorRecord(id="curry", vector=[0.7, 0.3], metadata={"position": "PG"}),
        ]
    )
    return store


def _anthropic_spec() -> ModelSpec:
    """A ModelSpec that selects Anthropic tool formatting in the loop."""
    return model_for(Task.AGENT_PLANNING)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tools(store, provider, spec=None):
    spec = spec or model_for(Task.NARRATIVE_WRITE)
    return [
        make_find_similar_tool(store),
        make_write_tool(provider, spec),
        make_compute_tool(FakeSandbox()),
    ]


def test_single_turn_write_only() -> None:
    """LLM calls write on iteration 1; loop ends immediately."""
    write_provider = FakeProvider("Klay is the answer.")
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=10,
                output_tokens=5,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="write", arguments={"question": "who?"})
                ],
            ),
        ]
    )
    store = _seed_store()
    tools = _tools(store, write_provider)
    result = run_agent(
        "who is the answer?",
        tools,
        planning_provider,
        _anthropic_spec(),
    )
    assert result.text == "Klay is the answer."
    assert result.partial is False
    assert result.iterations == 1


def test_two_turn_find_similar_then_write() -> None:
    """LLM calls find_similar, then write. Final answer is write's output."""
    write_provider = FakeProvider("Closest comp: Booker.")
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=10,
                output_tokens=5,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="find_similar",
                        arguments={"player_id": "klay", "top_k": 2},
                    )
                ],
            ),
            CompletionResponse(
                text="",
                input_tokens=20,
                output_tokens=10,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="t2",
                        name="write",
                        arguments={
                            "question": "comp for klay?",
                            "findings": ["Booker is closest"],
                        },
                    )
                ],
            ),
        ]
    )
    store = _seed_store()
    tools = _tools(store, write_provider)
    result = run_agent("comp for klay?", tools, planning_provider, _anthropic_spec())
    assert result.text == "Closest comp: Booker."
    assert result.iterations == 2
    # Trace shows tool calls in order
    tool_events = [t for t in result.trace if t.get("event") == "tool_call"]
    assert [e["name"] for e in tool_events] == ["find_similar", "write"]


def test_iteration_cap_marks_partial() -> None:
    """If the LLM keeps calling tools without ever calling write, the loop
    should cap at max_iterations and return partial=True."""
    write_provider = FakeProvider()
    # Always call find_similar, never write
    looping_responses = [
        CompletionResponse(
            text="",
            input_tokens=1,
            output_tokens=1,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(id=f"t{i}", name="find_similar", arguments={"player_id": "klay"})
            ],
        )
        for i in range(10)
    ]
    planning_provider = FakeProvider(looping_responses)
    store = _seed_store()
    tools = _tools(store, write_provider)
    result = run_agent(
        "anything?",
        tools,
        planning_provider,
        _anthropic_spec(),
        max_iterations=3,
    )
    assert result.partial is True
    assert result.text == ""
    assert result.iterations == 3
    assert any(t.get("event") == "limit_iterations" for t in result.trace)


def test_wall_clock_cap_marks_partial() -> None:
    clock = FakeClock()
    write_provider = FakeProvider()
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="find_similar", arguments={"player_id": "klay"})
                ],
            ),
            CompletionResponse(
                text="",
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t2", name="find_similar", arguments={"player_id": "klay"})
                ],
            ),
        ]
    )
    store = _seed_store()
    tools = _tools(store, write_provider)

    # Advance the clock past the wall budget *between* the first and second iter.
    # We achieve this by hooking the clock to also tick on each call.
    original = clock.__call__

    def ticking_clock() -> float:
        # Advance 20 s per tick — three ticks puts us past a 30 s budget.
        val = original()
        clock.advance(20.0)
        return val

    result = run_agent(
        "anything?",
        tools,
        planning_provider,
        _anthropic_spec(),
        max_wall_seconds=30.0,
        clock=ticking_clock,
    )
    assert result.partial is True
    assert any(t.get("event") == "limit_wall_clock" for t in result.trace)


def test_unknown_tool_returned_as_structured_error() -> None:
    write_provider = FakeProvider("final")
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="this_tool_does_not_exist", arguments={})
                ],
            ),
            CompletionResponse(
                text="",
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t2", name="write", arguments={"question": "?"})
                ],
            ),
        ]
    )
    store = _seed_store()
    tools = _tools(store, write_provider)
    result = run_agent("?", tools, planning_provider, _anthropic_spec())
    # The agent recovered: hallucinated tool error → wrote anyway → final text
    assert result.text == "final"
    tool_events = [t for t in result.trace if t.get("event") == "tool_call"]
    assert tool_events[0]["ok"] is False
    assert tool_events[1]["ok"] is True


def test_invalid_tool_arguments_returned_as_structured_error() -> None:
    write_provider = FakeProvider("final")
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
                tool_calls=[
                    # find_similar requires exactly one of player_id / vector;
                    # passing both should fail Pydantic validation.
                    ToolCall(
                        id="t1",
                        name="find_similar",
                        arguments={"player_id": "klay", "vector": [1.0, 0.0]},
                    )
                ],
            ),
            CompletionResponse(
                text="",
                input_tokens=1,
                output_tokens=1,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t2", name="write", arguments={"question": "?"})
                ],
            ),
        ]
    )
    store = _seed_store()
    tools = _tools(store, write_provider)
    result = run_agent("?", tools, planning_provider, _anthropic_spec())
    tool_events = [t for t in result.trace if t.get("event") == "tool_call"]
    assert tool_events[0]["ok"] is False  # validation failure recorded
    assert tool_events[1]["ok"] is True


def test_no_tool_calls_falls_back_to_text() -> None:
    """If the LLM responds with prose and no tool calls, that prose IS the answer."""
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="The answer is plain text.",
                input_tokens=10,
                output_tokens=10,
                stop_reason="end_turn",
            )
        ]
    )
    store = _seed_store()
    tools = _tools(store, FakeProvider())
    result = run_agent("?", tools, planning_provider, _anthropic_spec())
    assert result.text == "The answer is plain text."
    assert result.partial is False


def test_tokens_used_accumulates() -> None:
    write_provider = FakeProvider("final")
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="",
                input_tokens=100,
                output_tokens=50,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(id="t1", name="write", arguments={"question": "?"})
                ],
            )
        ]
    )
    store = _seed_store()
    tools = _tools(store, write_provider)
    result = run_agent("?", tools, planning_provider, _anthropic_spec())
    assert result.tokens_used >= 150


def test_unknown_provider_raises_at_tool_format() -> None:
    import pytest

    bad_spec = ModelSpec(provider="cohere", model_id="something")
    store = _seed_store()
    tools = _tools(store, FakeProvider())
    with pytest.raises(ValueError, match="No tool formatter"):
        run_agent("?", tools, FakeProvider(), bad_spec)


def test_empty_tools_list_still_runs() -> None:
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="just prose",
                input_tokens=5,
                output_tokens=5,
                stop_reason="end_turn",
            )
        ]
    )
    result = run_agent("?", [], planning_provider, _anthropic_spec())
    assert result.text == "just prose"


def test_tool_choice_auto_sent_when_tools_provided() -> None:
    planning_provider = FakeProvider(
        [
            CompletionResponse(
                text="ok",
                input_tokens=1,
                output_tokens=1,
                stop_reason="end_turn",
            )
        ]
    )
    store = _seed_store()
    tools = _tools(store, FakeProvider())
    run_agent("?", tools, planning_provider, _anthropic_spec())
    sent = planning_provider.requests[0]
    assert sent.tool_choice == "auto"
    assert len(sent.tools) == 3
