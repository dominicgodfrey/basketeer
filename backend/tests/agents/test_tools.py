from pydantic import BaseModel

from app.agents import (
    ToolSpec,
    find_tool,
    to_anthropic_tools,
    to_google_tools,
    to_openai_tools,
)


class _Args(BaseModel):
    x: int
    y: str = "default"


def _spec(name: str = "demo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="A demo tool.",
        args_schema=_Args,
        invoke=lambda args: {"got": args.model_dump()},
    )


def test_invoke_round_trips() -> None:
    spec = _spec()
    result = spec.invoke(_Args(x=1, y="hi"))
    assert result == {"got": {"x": 1, "y": "hi"}}


def test_anthropic_format_uses_input_schema_key() -> None:
    out = to_anthropic_tools([_spec()])
    assert len(out) == 1
    t = out[0]
    assert t["name"] == "demo"
    assert t["description"] == "A demo tool."
    assert "input_schema" in t
    assert t["input_schema"]["type"] == "object"
    assert "x" in t["input_schema"]["properties"]


def test_openai_format_wraps_in_function() -> None:
    out = to_openai_tools([_spec()])
    t = out[0]
    assert t["type"] == "function"
    fn = t["function"]
    assert fn["name"] == "demo"
    assert fn["description"] == "A demo tool."
    assert "parameters" in fn
    assert "x" in fn["parameters"]["properties"]


def test_google_format_uses_function_declarations() -> None:
    out = to_google_tools([_spec()])
    t = out[0]
    assert "function_declarations" in t
    decl = t["function_declarations"][0]
    assert decl["name"] == "demo"
    assert "parameters" in decl


def test_find_tool_returns_match_or_none() -> None:
    tools = [_spec("a"), _spec("b")]
    assert find_tool(tools, "a") is tools[0]
    assert find_tool(tools, "missing") is None


def test_translators_handle_empty_list() -> None:
    assert to_anthropic_tools([]) == []
    assert to_openai_tools([]) == []
    assert to_google_tools([]) == []
