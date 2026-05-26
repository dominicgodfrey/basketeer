"""`compute` primitive: sandboxed Python execution against already-fetched data.

This primitive is a thin contract layer over a `Sandbox`. The default sandbox
is `RaisingSandbox`, which fails loudly — that's deliberate. There is no
in-process fallback because CLAUDE.md is explicit: LLM-generated code must
never run in the API process. To use this primitive in anger, inject a real
Sandbox (E2B-backed or Docker-backed) at the agent's wire-up layer.

Contract for the agent (also stated in the docstring of `compute()`, since the
agent reads that):
- `data` is `dict[str, list[dict]]` — each value becomes a pandas DataFrame
  inside the sandbox under its dict key.
- `code` is Python. It must assign its final value to a variable named `result`.
- Allowed imports inside the sandbox: pandas, numpy, scipy, statistics, math.
- Wall-clock 5 s, memory 256 MB, no network.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.logging_setup import get_logger
from app.sandbox import RaisingSandbox, Sandbox

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MEMORY_MB = 256
MAX_CODE_CHARS = 8000


class ComputeRequest(BaseModel):
    """Input to the `compute` primitive.

    `code` references `data` keys by name. Assign the final value to `result`.
    Example:
        code = "result = sum(r['pts'] for r in df_players)"
        data = {"df_players": [{"pts": 22.1}, {"pts": 18.4}]}
    """

    code: str = Field(max_length=MAX_CODE_CHARS)
    data: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=0.1, le=30.0)
    memory_mb: int = Field(default=DEFAULT_MEMORY_MB, ge=64, le=1024)


class ComputeResponse(BaseModel):
    result: Any
    stdout: str = ""
    stderr: str = ""


def compute(request: ComputeRequest, sandbox: Sandbox | None = None) -> ComputeResponse:
    """Execute `request.code` against `request.data` inside `sandbox` and return the result.

    Pass `sandbox=None` (the default) and you'll get `RaisingSandbox`, which
    raises `SandboxNotConfiguredError`. Inject a real Sandbox in production.
    """
    runner: Sandbox = sandbox if sandbox is not None else RaisingSandbox()
    logger.info(
        "compute.invoke code_chars=%d inputs=%s timeout=%.1fs mem=%dMB",
        len(request.code),
        sorted(request.data.keys()),
        request.timeout_seconds,
        request.memory_mb,
    )
    sandbox_result = runner.run(
        code=request.code,
        data=request.data,
        timeout_seconds=request.timeout_seconds,
        memory_mb=request.memory_mb,
    )
    logger.info(
        "compute.complete result_type=%s stdout=%d stderr=%d",
        type(sandbox_result.result).__name__,
        len(sandbox_result.stdout),
        len(sandbox_result.stderr),
    )
    return ComputeResponse(
        result=sandbox_result.result,
        stdout=sandbox_result.stdout,
        stderr=sandbox_result.stderr,
    )
