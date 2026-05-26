"""Factories that turn the four primitives into ToolSpecs the agent can call.

Each builder closes over the runtime dependencies the primitive needs (vector
store, cache, sandbox, LLM provider). The agent loop receives a list of
ToolSpecs assembled by these builders at wire-up time. Adding query_stats
will follow the same pattern once the read-only DB role lands on the
collaborator's track.

Tool descriptions are written deliberately — the agent reads them to decide
when to call. Keep them precise about when to call, when *not* to call, and
the meaning of each argument.
"""

from typing import Any

from pydantic import BaseModel

from app.agents.tools import ToolSpec
from app.cache.cache import Cache
from app.cache.wrappers import cached_find_similar
from app.llm.providers import LLMProvider
from app.llm.router import ModelSpec
from app.primitives import (
    ComputeRequest,
    FindSimilarRequest,
    WriteContext,
    compute,
    find_similar,
    write,
)
from app.sandbox import Sandbox
from app.similarity import VectorStore

FIND_SIMILAR_DESCRIPTION = """\
Find players whose 50-dim stat profile vectors are most similar to a query.

Use this when the user asks for player comps, "find a player like X", "who plays \
similarly to Y", or wants to search by play style. Do NOT use for raw stat lookups \
(use query_stats) or for narrative synthesis (use write).

Provide EXACTLY ONE of:
- `player_id`: look up that player's stored vector and use it as the query.
- `vector`: a 50-d stat profile you've constructed.

`filter` is a Pinecone-style metadata filter applied before scoring. Supported \
keys depend on what's been indexed: `position` ("SG"), `season` ({"$gte": 2020}), \
`league` ("NBA"), `team_id`, `is_current_fa` (true). Use filters to cut the \
candidate set; do NOT post-filter in Python.

`top_k` is capped at 50. Returns hits ranked by cosine similarity, each with \
score and metadata. When querying by `player_id`, the player itself is excluded \
from results."""


WRITE_DESCRIPTION = """\
Synthesize a human-readable narrative answer from your accumulated findings.

This is the FINAL step. Call it once and only once, after you have enough \
information from the other tools to fully answer the user's question. Do not \
call any other tool after this.

- `question`: the user's original question, verbatim.
- `findings`: discrete facts you established during the loop (one per item, \
short and specific).
- `data`: optional structured supporting data (records, summary stats) the \
narrative should reference.
- `constraints`: caveats the answer must respect (small sample, league \
translation uncertainty, season range).

Returns the prose plus token counts."""


COMPUTE_DESCRIPTION = """\
Run sandboxed Python against data you've already fetched.

Use this for derived metrics, custom rankings, statistical tests, or anything \
that doesn't fit cleanly in SQL or vector search. Examples: value-per-dollar \
rankings, rolling averages, z-scores within a position group.

Contract:
- `code`: Python that references items in `data` by name. Each value of `data` \
becomes a pandas DataFrame in the sandbox under its dict key.
- `code` MUST assign its final output to a variable named `result`.
- Allowed imports inside the sandbox: pandas, numpy, scipy, statistics, math. \
NOTHING else. No network. No filesystem.
- 5-second wall-clock, 256 MB memory.

Example:
  code = "result = df_players.nlargest(5, 'ts_pct')[['name','ts_pct']].to_dict('records')"
  data = {"df_players": [...]}"""


def make_find_similar_tool(
    store: VectorStore,
    cache: Cache | None = None,
) -> ToolSpec:
    """Build the `find_similar` tool. Cache is optional; pass one in production."""

    def invoke(args: BaseModel) -> dict[str, Any]:
        request = _as(args, FindSimilarRequest)
        if cache is not None:
            response = cached_find_similar(request, store, cache)
        else:
            response = find_similar(request, store)
        return response.model_dump()

    return ToolSpec(
        name="find_similar",
        description=FIND_SIMILAR_DESCRIPTION,
        args_schema=FindSimilarRequest,
        invoke=invoke,
    )


def make_write_tool(provider: LLMProvider, model_spec: ModelSpec) -> ToolSpec:
    """Build the `write` tool. Closes over the LLM provider used for narrative
    synthesis (typically Haiku 4.5 or its env-overridden replacement)."""

    def invoke(args: BaseModel) -> dict[str, Any]:
        context = _as(args, WriteContext)
        response = write(context, provider, model_spec)
        return response.model_dump()

    return ToolSpec(
        name="write",
        description=WRITE_DESCRIPTION,
        args_schema=WriteContext,
        invoke=invoke,
    )


def make_compute_tool(sandbox: Sandbox) -> ToolSpec:
    """Build the `compute` tool. `sandbox` must be a real Sandbox (E2B-backed
    or Docker-backed). Passing `RaisingSandbox` is allowed but means every call
    will fail loudly — useful for catching agent regressions early."""

    def invoke(args: BaseModel) -> dict[str, Any]:
        request = _as(args, ComputeRequest)
        response = compute(request, sandbox=sandbox)
        return response.model_dump()

    return ToolSpec(
        name="compute",
        description=COMPUTE_DESCRIPTION,
        args_schema=ComputeRequest,
        invoke=invoke,
    )


def _as(args: BaseModel, expected: type[BaseModel]) -> BaseModel:
    """Defensive narrowing: the dispatcher should have validated args already,
    but verify the type here too — a wrong-typed invoke is a programming bug,
    not user error."""
    if not isinstance(args, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(args).__name__}")
    return args
