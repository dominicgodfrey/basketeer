# CLAUDE.md

Guidance for Claude Code when working on this repo.

## What this project is

NBA scouting tool with a dynamic agent. The agent answers open-ended basketball questions by composing four primitives (`query_stats`, `find_similar`, `compute`, `write`) in a ReAct loop. It can handle player comp queries, team-fit searches with cap constraints, and any analytical question answerable from the stored data — without per-question hardcoded endpoints.

Read `PLAN.md` for the full architecture and phased build order. This file is shorter and focused on day-to-day development rules.

## Tech stack

- Python 3.12 backend, FastAPI, SQLAlchemy, Pydantic v2
- Neon (serverless Postgres, free tier) for relational data
- Pinecone (serverless, free tier) for vector similarity, abstracted behind a `VectorStore` protocol
- E2B for sandboxed Python execution in the `compute` primitive
- LangChain ReAct agent
- Multi-provider LLM routing: Claude Haiku 4.5 for reasoning, Gemini 2.5 Flash / Flash-Lite for cheap structured steps
- `sqlglot` for SQL parsing and validation
- React + Vite + Tailwind + shadcn/ui for frontend
- ruff + black for Python lint/format
- pytest for tests

## Repo layout

```
backend/
  app/
    api/            # FastAPI routers
    agents/
      loop.py             # ReAct loop orchestration
      classifier.py       # Intent classifier (fast path vs agent)
      prompts/            # System prompts as .md files
    primitives/
      query_stats.py      # Text-to-SQL primitive
      find_similar.py     # Pinecone primitive
      compute.py          # Sandboxed Python primitive (E2B)
      write.py            # Narrative synthesis primitive
    data/
      ingest/             # Source-specific ingestion (nba_api, bbref, cbbdata, spotrac)
      translate/          # League translation coefficients + application
      embed/              # Stat profile vector construction
    db/
      models.py           # SQLAlchemy models
      session.py
      readonly.py         # Read-only role for agent SQL
    similarity/
      vector_store.py     # VectorStore protocol
      pinecone_store.py   # Pinecone implementation
    sandbox/
      e2b_client.py       # E2B integration
    llm/
      router.py           # Per-task model routing
      caching.py          # Prompt caching helpers
      providers/          # Anthropic, Google client wrappers
    schemas/              # Pydantic request/response models
    cache/                # SQL + Pinecone result caching
    config.py
    main.py
  tests/
  alembic/                # Migrations
  pyproject.toml
frontend/
  src/
    components/
      results/            # Per-result-type render components
    pages/
    api/                  # Typed client for backend endpoints
  package.json
scripts/                  # One-off ingestion / refresh scripts
PLAN.md
CLAUDE.md
README.md
```

## Development principles

**The agent should be small and the primitives should be smart.** Don't add tools for every query type. Don't put logic in the agent that belongs in deterministic Python. The agent picks primitives and writes their arguments; everything else is code.

**Determinism where possible, LLM where necessary.** Filtering by cap is SQL. Computing a custom value-per-dollar ranking is the `compute` primitive. Decomposing "3-and-D wing" into stat dimensions is LLM reasoning. Don't use the LLM for things SQL or Python can do.

**Translation is the load-bearing abstraction.** All non-NBA stats must pass through the translation layer before being embedded. Never compare a raw college stat to a raw NBA stat anywhere in the codebase. If you find yourself doing this, you've broken the contract.

**Cost discipline.** Default to the cheapest model that meets the reliability bar for each task — see the routing table below. Always enable prompt caching on the Anthropic agent loop. Cache deterministic tool results aggressively.

**Sandboxing is non-negotiable.** Anything the LLM generates as code goes through E2B (or the Docker fallback). Never run LLM-generated Python in-process, even for "simple" cases.

## Model routing

Routing is centralized in `app/llm/router.py`. Don't hardcode model names elsewhere.

| Task | Model | Notes |
|---|---|---|
| Intent classifier | Gemini 2.5 Flash-Lite | Single classification, cheap, failure is benign |
| Agent planning loop | Claude Haiku 4.5 | Tool-call reliability matters; prompt caching enabled |
| Text-to-SQL (`query_stats`) | Gemini 2.5 Flash | Validated by sqlglot, cheaper for structured output |
| Code generation (`compute`) | Claude Haiku 4.5 | Sandbox retry loops cost; reliability beats cheap |
| Narrative synthesis (`write`) | Claude Haiku 4.5 | User-facing prose quality |

Escalation: if narrative prose quality is consistently flat, swap the `write` task to Sonnet via the router config. Don't change other tasks without measuring first.

## Primitives

The agent only has these four. Resist adding more.

### `query_stats(sql: str)`
- SQL must pass `sqlglot.parse_one(sql, dialect='postgres')` and be a single `SELECT`.
- Reject `;` chaining, CTEs that mutate, anything that isn't read-only.
- Use the read-only Postgres role from `app/db/readonly.py`. Never the default role.
- 5-second statement timeout enforced at the connection level.
- Cap results at 1000 rows. If more, return summary + `truncated: true`.
- Return enriched results when cheap (join names, contract status, team). Saves agent iterations.

### `find_similar(...)`
- Input can be a player_id, a constructed vector, or a player_id with a delta description.
- Metadata filters happen Pinecone-side, not Python-side.
- Top-K capped at 50.
- Returns include similarity score and the metadata fields the agent typically needs next.

### `compute(code: str, data: dict)`
- Goes through `app/sandbox/e2b_client.py` exclusively. Never `exec()`, never `subprocess` from the agent path.
- Strict contract: data is `dict[str, list[dict]]` (DataFrame records), code references inputs by name, result assigned to `result` variable.
- Allowed imports inside sandbox: pandas, numpy, scipy, statistics, math. Nothing else.
- 5s timeout, 256MB memory, no network.
- Log every execution: code, input keys, output summary, success/failure.

### `write(context: dict)`
- Prompt template lives in `agents/prompts/write.md`.
- Context is a structured dict the agent assembles; not free-form.
- Output is prose, no JSON wrapping.

## ReAct loop guardrails

In `app/agents/loop.py`:

- Max 6 iterations per query (development). Tune down once measured.
- Max 30 seconds wall-clock.
- Max 30K tokens across the full loop.
- On any limit hit: return the partial result with a `partial: true` flag and a brief note. Don't fail silently.
- Every iteration logs the agent's planned tool call, the result summary, and the agent's reasoning. This is your debugging gold mine.

## Caching

Two layers, both essential:

**Prompt caching (Anthropic):** Mark the system prompt + tool descriptions + schema docs as cached via `cache_control`. This is the single biggest cost optimization. Verify it's firing by checking the `cache_read_input_tokens` field in API responses.

**Result caching (Redis or in-memory):** Hash each `query_stats` SQL and each `find_similar` query+filters. Cache the result with a 24h TTL. Most queries on completed seasons are deterministic. Invalidate caches when ingestion runs.

Don't cache LLM outputs directly — too non-deterministic to be useful. Cache the deterministic primitive results that feed into them.

## Code style

- Type hints on every function signature. Pydantic for anything crossing a boundary (API, DB, LLM tool input/output, sandbox).
- Functions should have one job. If a function name has "and" in it, split it.
- No bare `except:`. Catch specific exceptions.
- Log at INFO for major operations (ingestion start/end, query received, agent iteration), DEBUG for everything else.
- Docstrings on public functions and on every primitive (the agent reads them).
- Tests for translation logic, SQL validation, sandbox contract, and result caching are mandatory.

## Database

- SQLAlchemy ORM, Alembic for migrations.
- Postgres holds relational data only: players, seasons, contracts, teams, ingestion provenance.
- Two roles: default (used by ingestion + APIs) and read-only (used by the agent's `query_stats`). The read-only role has SELECT only, no access to system tables, no temp table creation.
- Player ID is internal; map external IDs (`nba_api_id`, `bbref_id`, `cbb_id`) as separate columns.
- Stat profile vectors live in Pinecone, not Postgres. Postgres holds raw and translated stats.

## Vector store

- All vector operations go through the `VectorStore` protocol in `similarity/vector_store.py`.
- The Pinecone implementation is the default. Don't import the Pinecone SDK anywhere outside `similarity/pinecone_store.py`.
- Pinecone metadata fields are first-class for filtering: `player_id`, `season`, `league`, `position`, `team_id`, `is_current_fa`. Filter in metadata, not in Python.

## LangChain conventions

- Use LangChain's tool decorator pattern. Tools take Pydantic input models and return Pydantic output models.
- Each tool's docstring is the agent's API contract. Write them carefully.
- System prompts live in `agents/prompts/` as `.md` files, not inline in code.
- Don't use LangChain's higher-level abstractions where they obscure cost (some agents make hidden LLM calls). Stay close to the metal so you can see and control every call.

## Frontend conventions

- Typed API client generated from FastAPI's OpenAPI schema (or hand-written if simpler).
- Response schema has a `result_type` discriminator: `comp_list`, `ranked_table`, `narrative`, `chart`, plus a fallback.
- One render component per result type in `components/results/`. The page dispatches on `result_type`.
- Use shadcn/ui primitives. No custom design system from scratch.
- No client-side LLM calls. All LLM calls go through the backend.

## Testing

- `pytest` for backend.
- Integration tests can hit a test Postgres (docker compose).
- Don't test against the live LLM in CI — mock the LangChain client. Run live LLM evals manually before merging significant agent changes.
- Always test SQL validation against attack inputs (injection attempts, multi-statement, DDL).
- Always test sandbox contract violations (network access attempts, fs access, infinite loops, OOM).

## Git workflow

Commit after every major milestone. The phases in `PLAN.md` are the milestone boundaries; each phase should produce one or more commits with a clear message.

**Commit message format:**
```
<type>(<scope>): <subject>

<body, optional, wrapped at 72 chars>
```

Types: `feat`, `fix`, `refactor`, `data`, `infra`, `docs`, `test`, `chore`.
Scopes: `ingest`, `translate`, `embed`, `similarity`, `agent`, `primitive`, `sandbox`, `llm`, `api`, `ui`, `db`, `cache`, `ci`.

**Examples:**
- `feat(primitive): add query_stats with sqlglot validation`
- `feat(sandbox): wire E2B into compute primitive`
- `feat(agent): ReAct loop with iteration cap and timeout`
- `feat(llm): enable prompt caching on agent system prompt`
- `feat(llm): route text-to-SQL to Gemini 2.5 Flash`
- `feat(ui): result-type dispatch in scout page`

**Commit at these checkpoints (from PLAN.md):**
1. After Phase 0 — `infra: scaffold backend, frontend, db schema, third-party services`
2. After Phase 1 — one commit per source (NBA, college, etc.) so we can roll back individually
3. After Phase 2 — `feat(embed): translation + 50d stat profiles + Pinecone index`
4. After Phase 3 — `feat(ui): comp finder end-to-end`
5. After Phase 4 — `feat(agent): four primitives + ReAct loop`
6. After Phase 5 — `feat(agent): intent classifier + fast path`
7. After Phase 6 — `feat(ui): result-type rendering`

Inside a phase, commit at logical sub-units (one primitive built, one source ingested, one component shipped). Don't let unstaged work pile up for days.

**Branching:** `main` is always deployable. Work on `feat/<short-description>` branches and squash-merge.

## Things to ask before doing

Surface these before making decisions:

- Adding a new primitive to the agent (does it need to be LLM-callable, or could it be a private helper inside an existing primitive?)
- Adding a new data source (cost, ToS, maintenance burden)
- Schema changes after Phase 1 (need a migration plan)
- Anything that could spike LLM costs (longer prompts, higher-tier models, more iteration headroom, bypassing the cache)
- Anything that could spike Pinecone or E2B usage
- Swapping a sandbox provider (E2B → Docker → other)
- Swapping a vector store provider (possible by design, but should be deliberate)
- Changes to the read-only Postgres role's permissions
- Adding a heavy dependency (GPU, large native binary)

## Common pitfalls to avoid

- **Don't trust raw stat comparisons across leagues.** Always go through the translation layer.
- **Don't fuzzy-match player names without a canonical ID layer.** Use name + birthdate as primary key for cross-source matching.
- **Don't put long prompts in code.** They belong in `prompts/` files.
- **Don't call the LLM for things SQL, Pinecone metadata filtering, or Python can do.**
- **Don't import Pinecone SDK outside `similarity/pinecone_store.py`.** Use the protocol.
- **Don't import E2B SDK outside `sandbox/e2b_client.py`.** Use the primitive.
- **Don't run LLM-generated code in-process, ever.** Even for "trivial" cases. The attack surface is real.
- **Don't connect the agent to the default Postgres role.** Always the read-only role.
- **Don't dump full tool results into the agent's context.** Summarize for the agent, keep full data in the scratchpad for `compute`.
- **Don't hardcode model names.** Use the router in `app/llm/router.py`.
- **Don't store secrets in code.** All keys come from env vars via Pydantic settings.
- **Don't fetch player headshots or video.** v2 concern, licensing complexity.
- **Don't panic about Neon cold-starts.** ~1s first query after idle is fine.

## When in doubt

Refer to `PLAN.md`. If the answer isn't there, ask before proceeding rather than guessing.
