import pytest
from pydantic import ValidationError

from app.primitives import ComputeRequest, compute
from app.sandbox import FakeSandbox, SandboxNotConfiguredError


def test_default_sandbox_raises_not_configured() -> None:
    """Without an injected sandbox, compute must refuse to run."""
    with pytest.raises(SandboxNotConfiguredError):
        compute(ComputeRequest(code="result = 1"))


def test_with_injected_sandbox_returns_result() -> None:
    sandbox = FakeSandbox(result=[1, 2, 3])
    response = compute(ComputeRequest(code="result = [1,2,3]"), sandbox=sandbox)
    assert response.result == [1, 2, 3]


def test_request_passes_data_and_limits_to_sandbox() -> None:
    sandbox = FakeSandbox(result=0)
    request = ComputeRequest(
        code="result = sum(r['x'] for r in df)",
        data={"df": [{"x": 1}, {"x": 2}]},
        timeout_seconds=2.5,
        memory_mb=128,
    )
    compute(request, sandbox=sandbox)
    call = sandbox.calls[0]
    assert call["code"] == request.code
    assert call["data"] == request.data
    assert call["timeout_seconds"] == 2.5
    assert call["memory_mb"] == 128


def test_stdout_and_stderr_round_trip() -> None:
    sandbox = FakeSandbox(result=None, stdout="printed", stderr="warning")
    response = compute(ComputeRequest(code="print('hi')"), sandbox=sandbox)
    assert response.stdout == "printed"
    assert response.stderr == "warning"


def test_code_length_limit_enforced() -> None:
    too_long = "a = 1\n" * 5000  # > 8000 chars
    with pytest.raises(ValidationError):
        ComputeRequest(code=too_long)


def test_timeout_bounds() -> None:
    with pytest.raises(ValidationError):
        ComputeRequest(code="result=1", timeout_seconds=0.0)
    with pytest.raises(ValidationError):
        ComputeRequest(code="result=1", timeout_seconds=31.0)


def test_memory_bounds() -> None:
    with pytest.raises(ValidationError):
        ComputeRequest(code="result=1", memory_mb=32)
    with pytest.raises(ValidationError):
        ComputeRequest(code="result=1", memory_mb=2048)
