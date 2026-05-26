"""Sandbox protocol — the boundary between the `compute` primitive and the
secure execution environment.

CLAUDE.md is explicit: never `exec()`, never `subprocess` from the agent path.
LLM-generated code goes through an isolated sandbox (E2B in production, or
Docker-per-call as a fallback) with no network, capped resources, and a strict
allowed-imports list. This module defines the abstraction; concrete impls live
in their own files (`e2b_sandbox.py`, `docker_sandbox.py`) and are not built
yet — `RaisingSandbox` is the deliberate default so calls fail loudly until
a real sandbox is wired.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class SandboxResult:
    """The output of one sandboxed execution."""

    result: Any
    stdout: str = ""
    stderr: str = ""


class SandboxNotConfiguredError(RuntimeError):
    """Raised when `compute` is invoked but no real Sandbox is wired in.

    Catching this in higher layers is a code smell — fail loudly instead so the
    operator knows to set up E2B or Docker.
    """


@runtime_checkable
class Sandbox(Protocol):
    """Synchronous execution of code against named DataFrame inputs.

    Implementations must:
    - Enforce a wall-clock timeout (kill the worker, raise from `run`).
    - Enforce a memory cap.
    - Disable all network access inside the sandbox.
    - Restrict imports to `pandas`, `numpy`, `scipy`, `statistics`, `math`.
    - Truncate stdout/stderr to 100 KB each.
    - Convert each value in `data` to a pandas DataFrame and expose it under
      its dict key (e.g. `data["df_players"]` becomes a global named `df_players`
      inside the sandbox).
    - Read the variable named `result` after the code runs; raise if absent.
    """

    def run(
        self,
        code: str,
        data: dict[str, list[dict[str, Any]]],
        *,
        timeout_seconds: float = 5.0,
        memory_mb: int = 256,
    ) -> SandboxResult: ...


class RaisingSandbox:
    """Default Sandbox that refuses to execute. Use this until E2B or Docker
    is configured. It exists so the rest of the system can reference a Sandbox
    by type without anyone accidentally falling back to in-process `exec()`."""

    def run(
        self,
        code: str,
        data: dict[str, list[dict[str, Any]]],
        *,
        timeout_seconds: float = 5.0,
        memory_mb: int = 256,
    ) -> SandboxResult:
        raise SandboxNotConfiguredError(
            "No sandbox configured. Set E2B_API_KEY (and wire E2BSandbox) or "
            "configure DockerSandbox before calling the compute primitive. "
            "Running LLM-generated Python in-process is never acceptable."
        )


class FakeSandbox:
    """Test double satisfying the Sandbox protocol. Returns a configured result
    without running any code. Records every `run` call for assertions."""

    def __init__(
        self,
        result: Any = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self._result = result
        self._stdout = stdout
        self._stderr = stderr
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        code: str,
        data: dict[str, list[dict[str, Any]]],
        *,
        timeout_seconds: float = 5.0,
        memory_mb: int = 256,
    ) -> SandboxResult:
        self.calls.append(
            {
                "code": code,
                "data": data,
                "timeout_seconds": timeout_seconds,
                "memory_mb": memory_mb,
            }
        )
        return SandboxResult(
            result=self._result,
            stdout=self._stdout,
            stderr=self._stderr,
        )
