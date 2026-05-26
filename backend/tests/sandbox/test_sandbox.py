import pytest

from app.sandbox import (
    FakeSandbox,
    RaisingSandbox,
    Sandbox,
    SandboxNotConfiguredError,
)


def test_raising_sandbox_refuses_to_run() -> None:
    sandbox = RaisingSandbox()
    with pytest.raises(SandboxNotConfiguredError, match="No sandbox configured"):
        sandbox.run(code="result = 1", data={})


def test_raising_sandbox_satisfies_protocol() -> None:
    assert isinstance(RaisingSandbox(), Sandbox)


def test_fake_sandbox_returns_configured_result() -> None:
    sandbox = FakeSandbox(result={"answer": 42}, stdout="hi", stderr="warn")
    out = sandbox.run(code="anything", data={"df": [{"a": 1}]})
    assert out.result == {"answer": 42}
    assert out.stdout == "hi"
    assert out.stderr == "warn"


def test_fake_sandbox_records_calls() -> None:
    sandbox = FakeSandbox(result=7)
    sandbox.run(code="result = 7", data={"d": [{"x": 1}]}, timeout_seconds=3, memory_mb=128)
    assert len(sandbox.calls) == 1
    call = sandbox.calls[0]
    assert call["code"] == "result = 7"
    assert call["data"] == {"d": [{"x": 1}]}
    assert call["timeout_seconds"] == 3
    assert call["memory_mb"] == 128


def test_fake_sandbox_satisfies_protocol() -> None:
    assert isinstance(FakeSandbox(), Sandbox)
