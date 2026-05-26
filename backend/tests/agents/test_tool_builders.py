from app.agents import (
    make_compute_tool,
    make_find_similar_tool,
    make_write_tool,
)
from app.cache import InMemoryCache
from app.llm.providers import FakeProvider
from app.llm.router import Task, model_for
from app.primitives import (
    ComputeRequest,
    FindSimilarRequest,
    WriteContext,
)
from app.sandbox import FakeSandbox
from app.similarity import InMemoryVectorStore, VectorRecord


def test_find_similar_tool_invokes_underlying_primitive() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], metadata={"position": "SG"}),
            VectorRecord(id="b", vector=[0.0, 1.0], metadata={"position": "PG"}),
        ]
    )
    tool = make_find_similar_tool(store)
    result = tool.invoke(FindSimilarRequest(vector=[1.0, 0.0], top_k=1))
    assert "hits" in result
    assert result["hits"][0]["id"] == "a"


def test_find_similar_tool_uses_cache_when_provided() -> None:
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0])])
    cache = InMemoryCache()
    tool = make_find_similar_tool(store, cache=cache)
    request = FindSimilarRequest(vector=[1.0])
    tool.invoke(request)
    # Anything was cached
    assert len(cache) > 0


def test_write_tool_invokes_underlying_primitive() -> None:
    provider = FakeProvider("answer text")
    spec = model_for(Task.NARRATIVE_WRITE)
    tool = make_write_tool(provider, spec)
    result = tool.invoke(WriteContext(question="anything"))
    assert result["text"] == "answer text"
    assert "input_tokens" in result


def test_compute_tool_invokes_sandbox() -> None:
    sandbox = FakeSandbox(result=[1, 2, 3])
    tool = make_compute_tool(sandbox)
    result = tool.invoke(ComputeRequest(code="result = [1,2,3]"))
    assert result["result"] == [1, 2, 3]


def test_builders_use_primitive_pydantic_schemas() -> None:
    """The ToolSpec.args_schema must be the primitive's request model."""
    store = InMemoryVectorStore()
    sandbox = FakeSandbox()
    provider = FakeProvider()
    spec = model_for(Task.NARRATIVE_WRITE)

    assert make_find_similar_tool(store).args_schema is FindSimilarRequest
    assert make_write_tool(provider, spec).args_schema is WriteContext
    assert make_compute_tool(sandbox).args_schema is ComputeRequest


def test_tool_descriptions_are_nontrivial() -> None:
    """Sanity check: descriptions exist and have enough content to be useful."""
    store = InMemoryVectorStore()
    sandbox = FakeSandbox()
    provider = FakeProvider()
    spec = model_for(Task.NARRATIVE_WRITE)
    for tool in [
        make_find_similar_tool(store),
        make_write_tool(provider, spec),
        make_compute_tool(sandbox),
    ]:
        assert len(tool.description) > 100, f"{tool.name} description too short"
