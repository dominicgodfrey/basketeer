# NBA Scout Agent — Project Plan

A scouting tool that answers open-ended basketball questions using a dynamic agent that composes a small set of primitives over a unified player/season/contract dataset. Built as a side project, optimized for low cost while demonstrating production agentic-systems patterns.

## Goals

1. **Player comp finder** — given a player (NBA, college, or a custom stat line), return the most statistically similar players across history.
2. **Team-fit search** — given a natural language query like "find a 3-and-D wing who plays like Klay and fits the Warriors' cap," return ranked free agents or trade targets that match.
3. **Open-ended analytical queries** — anything answerable from the stored data: "most underpaid wings in the league," "biggest TS% dropoffs after age 30," "which rookies have the most Steph in them," etc. The agent composes primitives dynamically rather than hitting hardcoded endpoints.

## Non-goals (v1)

- Live game tracking or in-game analytics
- Video/film analysis
- High school stat coverage beyond recruit rankings (data quality too uneven)
- Trade machine simulation (cap-compliant multi-team trades) — possibly v2
- Mobile app — web-only for now

## Architecture overview

```
User query
   ↓
[Frontend: React + Tailwind]
   ↓
[FastAPI backend]
   ↓
[Intent classifier]                ← Gemini 2.5 Flash-Lite
   ↓
   ├── Fast path (trivial queries) → primitive call → response
   └── Agent loop (complex queries) → ReAct loop
        ↓
        [Hand-rolled ReAct loop: Haiku 4.5 (or equivalent) w/ prompt caching]
        ↓ (one primitive per iteration, max 6)
        ├── query_stats(sql)        → Neon Postgres (read-only role)
        ├── find_similar(...)       → Pinecone serverless
        ├── compute(code, data)     → E2B sandbox
        └── write(context)          → Haiku 4.5 narrative
        ↓
        [Scratchpad accumulates results, agent loops until done]
   ↓
Response (typed JSON: result_type + payload + provenance)
   ↓
[React renders by result_type: comp cards / table / chart / prose]
```

The agent operates as a **planning loop**, not a single forward pass. It calls a primitive, observes the result, decides what to do next, and loops until it has enough to answer. This is the ReAct pattern and it's what allows the system to answer questions we never explicitly designed for.

## The four primitives

These are the only tools the agent has. Everything else is composition.

### `query_stats(sql: str) -> rows`
Text-to-SQL against Neon Postgres. The agent generates SQL given the schema documentation in its system prompt. Hard constraints:
- Read-only Postgres role with `SELECT` only, no system tables.
- SQL parsed and validated with `sqlglot` before execution. Single `SELECT` statement only; reject anything else.
- 5-second statement timeout.
- Result row cap of 1000; if more, return summary + truncation flag.
- Returns enriched results where possible (joining commonly-needed metadata) so the agent doesn't have to chain calls just to look up names.

### `find_similar(player_id_or_vector, filters) -> ranked_list`
Pinecone semantic search over hand-engineered 50d stat profile vectors. The agent can pass a player ID, a constructed stat profile, or a query like "stats similar to peak Klay but with better playmaking" (which gets parsed into a vector adjustment). Metadata filters (position, current_fa, league, season range) narrow before scoring.

### `compute(code: str, data: dict) -> result`
Sandboxed Python execution against already-fetched data. Used for derived metrics, statistical tests, custom rankings, anything that doesn't fit cleanly in SQL. The agent writes pandas/numpy one-liners or small functions. Sandboxing is non-negotiable here — see the dedicated section below.

### `write(context: dict) -> prose`
Final narrative synthesis. Takes the accumulated scratchpad and produces a human-readable answer. Separate primitive from the agent's planning calls because the prompts differ (this one wants longer, more thoughtful prose; planning calls want short, structured outputs).

## Data layer

The hardest engineering problem in this project is making stats from different leagues comparable. The plan: **everything gets translated to NBA-equivalent stat space** before being stored.

### Sources

| Source | Coverage | Cost | Notes |
|---|---|---|---|
| `nba_api` (swar) | NBA stats 1996–present, including advanced + tracking | Free | Rate limited; respect terms of use. Primary NBA source. |
| Basketball-Reference | NBA pre-1996, college historical | Free (scrape) | Polite scraping with caching. One-time historical pull. |
| collegebasketballdata.com | NCAA D1 men's, advanced metrics | Free tier (Patreon for tier 2+) | Use for current + recent seasons. |
| Spotrac (scrape) | Salary cap, contracts | Free (scrape) | Update weekly during season. |
| 247Sports composite | HS recruit rankings | Free (scrape) | Single feature, not full stat profile. |

### League translation layer

Every player-season gets converted into an NBA-equivalent vector via translation coefficients per stat (published values from Vashro/Pelton, refined later from our own data). All vectors live in one space so Cooper Flagg's Duke season is comparable to Klay Thompson's 2015 NBA season.

### Stat profile (vector schema)

Per player-season, ~50 dimensions in NBA-equivalent space:

- **Scoring**: TS%, eFG%, 3PAr, FTr, points per shot attempt
- **Shooting splits**: 3PT%, mid-range %, rim %
- **Playmaking**: AST%, AST/USG, TOV%, AST/TO
- **Rebounding**: ORB%, DRB% (positional-adjusted)
- **Defense**: STL%, BLK%, DRTG, deflections, contested shots (where available)
- **Usage / role**: USG%, MIN%, on/off net rating
- **Athletic / physical**: height, wingspan (where measured), age, draft position
- **Context**: position, team pace, teammate quality

### Storage

- **Neon (serverless Postgres)** for player metadata, season records, contracts, team data. Free tier: 0.5GB + 100 compute-hours/month, auto-suspends after 5 min idle, ~1s cold start.
- **Pinecone serverless** for vector similarity. Free tier 2GB + 1M reads/month, far more than we need (~100K vectors total).
- `VectorStore` protocol abstracts Pinecone so the underlying provider is swappable.

**Why Neon over Supabase or Railway:** Supabase pauses projects after 7 days idle with manual reactivation. Railway requires $5/month minimum after trial. Neon's auto-suspend + auto-wake gives us free hosting with ~1s cold start (negligible next to the LLM call), plus database branching as a portfolio talking point.

## The compute primitive: sandboxing

The compute primitive lets the agent execute Python on fetched data. **An LLM writing code that runs on your server is a real attack surface** — prompt injection can absolutely cause the LLM to write malicious code. In-process sandboxes (RestrictedPython, AST walkers) are not secure; Python's dynamism makes them defeatable.

**Approach: E2B for development, Docker-per-call as a fallback if we self-host later.**

### E2B (primary)
- Managed sandbox service designed for LLM-generated code execution. Free tier covers our usage easily.
- Spawns isolated VM-grade environments per call with no network access and capped resources.
- Zero infra burden. Trade ~500ms latency per call for not having to operate sandboxes ourselves.

### Contract for the compute primitive
The primitive does not accept arbitrary scripts. The contract is strict:
- Input: a dict of named pandas DataFrames (already fetched by earlier tool calls).
- Code: a string that references inputs by name (`df_players`, `df_contracts`) and assigns its result to a variable called `result`.
- Output: JSON-serializable (DataFrame → records, dict, scalar, or list).
- Available imports inside the sandbox: `pandas`, `numpy`, `scipy`, `statistics`, `math`. Nothing else installed.
- Stdout/stderr truncated to 100KB.
- Hard timeout: 5 seconds. Memory cap: 256MB.

### Fallback path (Docker-per-call)
If we ever need to self-host or E2B costs become annoying:
- One-shot Docker container per call: `--network=none --read-only --memory=256m --cpus=0.5 --pids-limit=64`, tmpfs `/tmp`.
- Container image preinstalled with the allowed libraries and nothing else.
- ~500ms-1s spinup overhead, but full control.

Logging: every code execution is logged with the generated code, inputs, and outputs. Invaluable for debugging and for noticing when the agent is being weird.

## LLM layer

Hand-rolled ReAct loop orchestrating the four primitives (no LangChain dependency — see "Agent conventions" in CLAUDE.md for the rationale). **Per-task model routing** — different tasks have different reliability needs, so we route deliberately rather than picking one model for everything. The provider abstraction supports Anthropic, Google, and any OpenAI-compatible endpoint (DeepSeek, Together, Groq, Moonshot, etc.) interchangeably.

| Step | Model | Why |
|---|---|---|
| Intent classifier (fast-path vs agent loop) | Gemini 2.5 Flash-Lite | Single classification, ~10x cheaper than Haiku, failure is benign (just misroutes) |
| Agent planning loop | Claude Haiku 4.5 | Tool-call reliability matters most here; cost compounds across iterations |
| Text-to-SQL inside `query_stats` | Gemini 2.5 Flash | Validated downstream by sqlglot; cheaper, good at structured output |
| Code generation inside `compute` | Claude Haiku 4.5 | Sandbox error loops cost; reliability beats cheap |
| Final narrative `write` | Claude Haiku 4.5 | User-facing prose quality |

### Prompt caching is the single biggest cost lever
The system prompt for the agent loop is large — schema docs, primitive descriptions, example queries, agent instructions — probably 3-5K tokens of identical-across-every-query content. Anthropic supports prompt caching via `cache_control` markers: cached prefixes are billed at roughly 10% of normal input cost. Wire this up before any other optimization. Expected saving: 60-80% on input costs.

### Other cost levers
- **SQL/Pinecone result caching** with 24h TTL. Most queries on completed seasons are deterministic; hash the SQL or vector query and cache the result.
- **Smart primitive returns**. Tools return enriched results (joined names, contract status, recent form) so the agent rarely needs to chain calls just to look things up. Fewer iterations = fewer LLM calls.
- **Scratchpad summary**, not dump. If `query_stats` returns 200 rows, the agent sees "200 rows returned, columns X, here are first/last 5"; the full data lives in the scratchpad for `compute` to use. Cuts context tokens 5-10x.
- **Aggressive iteration cap**. Max 6 tool calls per query (development), tune down once we have data. Most queries finish in 2-4.
- **Structured outputs for planning**. Planning calls request brief, structured responses ("emit only the tool call, no explanation"). Free-form chain-of-thought is wasteful at the routing layer.

### Cost estimate
Without optimization, a complex analytical query is ~5 LLM calls × ~3K tokens each ≈ ~$0.05 per query at Haiku rates. With caching + routing + smart primitives + result caching, the same query is ~3 calls × ~1.5K effective tokens ≈ ~$0.005-0.012 per query. At 1-2K queries/month while developing and demoing: **$5-15/month LLM spend**.

## Phased build order

Each phase ends with a working, demonstrable artifact. Commit after each phase.

### Phase 0 — Project setup (1-2 days)
- Repo init, Python virtualenv, FastAPI scaffold
- Neon DB provisioned, Pinecone index provisioned, E2B account set up
- Schema for `players`, `seasons`, `contracts`, `teams`
- CI lint + format (ruff, black) on push
- **Deliverable**: empty FastAPI server returning health check, schema migrations applied, all third-party services reachable from the backend

### Phase 1 — Data ingestion (1 week)
- `nba_api` ingestion for current and historical NBA seasons
- Basketball-Reference scraper for pre-1996 + college historical
- collegebasketballdata.com integration for recent NCAA seasons
- Player ID resolution across sources (name + birthdate as key, fuzzy fallback)
- **Deliverable**: populated DB with ~5K NBA players, ~30K NCAA player-seasons

### Phase 2 — Translation + embedding (4-5 days)
- League translation coefficients applied at ingest
- 50-dim stat profile vectors per player-season
- `VectorStore` protocol; `PineconeVectorStore` implementation
- Bulk upsert with metadata for filterable queries
- Sanity check: Klay 2015 vector → nearest neighbors should include Booker, Middleton, etc.
- **Deliverable**: working similarity search via Pinecone with sensible comps

### Phase 3 — Comp finder UI (3-4 days)
- React frontend: player search, comp results grid, per-dimension similarity bars
- Single FastAPI endpoint wrapping the similarity function (no agent yet — this is just direct primitive access)
- **Deliverable**: working demo where you type a player and see comps. **This is v0 — already a useful tool standalone.**

### Phase 4 — Primitives + agent loop (1.5 weeks)
Build the four primitives behind clean interfaces, then wire them into a ReAct agent.

- `query_stats`: read-only Postgres role, sqlglot validation, statement timeout, row cap, enrichment joins
- `find_similar`: refactor Phase 2's similarity into the primitive contract
- `compute`: E2B integration, strict input/output contract, allowed-library list
- `write`: narrative prompt template with example outputs
- Hand-rolled ReAct loop with prompt caching enabled on the system prompt
- Iteration cap, total-time cap, total-token cap as guardrails
- Tool error handling: structured errors the agent can read and recover from

**Deliverable**: agent can answer "find me a 3-and-D wing who fits GSW's cap" and "who's the most underpaid wing in the league" without dedicated code for either.

### Phase 5 — Intent classifier + fast path (3-4 days)
- Gemini 2.5 Flash-Lite classifier: trivial / agent-required
- Trivial queries route directly to a single primitive (skip the agent loop)
- Agent loop only fires for genuinely analytical questions
- **Deliverable**: ~40% of queries should now bypass the full agent loop

### Phase 6 — Result rendering + frontend polish (4-5 days)
- Response schema with `result_type` discriminator
- React render components: `comp_list`, `ranked_table`, `narrative`, `chart`
- Frontend dispatches on `result_type`
- Loading states, query history, share links
- **Deliverable**: agent's varied outputs render appropriately in the UI

### Phase 7 — Cost optimization + production polish (ongoing)
- Verify prompt caching is firing correctly (check token usage metrics)
- SQL/Pinecone result caching with TTL
- Tune iteration cap based on observed loop lengths
- Better fuzzy player name matching
- Translation coefficient refinement from our own data
- Error handling, rate limiting, basic auth if shared publicly

## Tech stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy, Pydantic v2
- **Data**: pandas, numpy, `nba_api`, sqlglot (for SQL validation)
- **DB**: Neon (serverless Postgres, free tier)
- **Vector store**: Pinecone (serverless, free tier) behind a `VectorStore` protocol
- **Sandbox**: E2B (managed) for `compute` primitive
- **LLM**: Hand-rolled ReAct loop (no LangChain dependency). Provider-agnostic abstraction (`LLMProvider` protocol) supports Anthropic, Google, and OpenAI-compatible endpoints (DeepSeek, Together, Groq, Moonshot, etc.) interchangeably; the active model is configured per-task via `MODEL_<TASK>` env vars. See `backend/app/llm/router.py`.
- **Frontend**: React + Vite, Tailwind, shadcn/ui
- **Hosting**: Railway or Fly.io (backend), Vercel (frontend)
- **CI**: GitHub Actions

## Cost summary

| Item | Monthly cost |
|---|---|
| Neon DB (serverless Postgres) | $0 (free tier; auto-suspends when idle) |
| Pinecone serverless | $0 (free tier covers our scale ~100x over) |
| E2B sandbox | $0 (free tier sufficient at our query volume) |
| Backend hosting | $0 (free tier; ~$5 if always-on becomes needed) |
| Vercel frontend | $0 |
| LLM (Haiku + Flash routing, with prompt caching) | ~$5-15 at modest usage |
| Domain (optional) | $1/month amortized |
| **Total** | **~$5-20/month** |

## Division of labor

- **Dominic (full-stack/AI/ML)**: translation layer, embedding (vector construction), agent loop, the four primitives (modulo `query_stats` SQL generation, which depends on the schema), sandbox integration, LLM routing + prompt caching, intent classifier, result caching, cost optimization
- **Friend (frontend/SWE + DB foundation)**:
  - **Frontend**: React UI for comp finder, query input, result render components (comp cards, ranked tables, narrative display), `result_type` dispatch, loading states, query history
  - **Postgres DB foundation**: SQLAlchemy models for `players`/`seasons`/`contracts`/`teams`, Alembic migrations, the read-only Postgres role used by the agent's `query_stats` primitive, basic FastAPI endpoint plumbing once primitives exist
  - **Data ingestion** (depends on the schema being in place): `nba_api`, basketball-reference, collegebasketballdata.com, Spotrac scrapers; player ID resolution across sources

The contract between the two tracks:
- The agent's response payload has a `result_type` field and a typed payload. Frontend builds render components per `result_type` and a fallback for unknown types. The interface is stable from Phase 4 onward.
- The DB schema is owned by Friend; Dominic's primitives access it through the `VectorStore` protocol (for Pinecone) and a thin SQLAlchemy session layer (for the `query_stats` read-only role). Dominic builds and tests primitives against in-memory / fake implementations until the schema lands, then plugs in.

This split means Dominic's first work focuses on the backend scaffold (non-DB), the LLM router + provider wrappers, the translation math, the `VectorStore` protocol, and the `compute` / `write` primitives + agent loop — all of which can be built and tested without the Postgres schema existing yet.

## Open questions / risks

- **NBA.com terms of use**: `nba_api` works but stats.nba.com has terms restricting commercial use. Side-project / personal use should be fine; revisit if this becomes a product.
- **Translation coefficients are imperfect** for very recent college players (small NBA sample). Mitigation: weight by sample size, fall back to position-averaged.
- **Cap data freshness**: Spotrac scraping is fragile. Accept weekly refresh; if it breaks repeatedly, surface stale-data warnings in the UI.
- **Agent bad SQL or bad code**: caught by sqlglot validation and sandbox isolation. The agent reads structured errors and retries; logged for prompt improvement.
- **Agent loop runaway**: capped at 6 iterations, 30s wall-clock, 30K tokens. Returns partial result with "I ran out of time" caveat if hit.
- **Prompt injection via player names or query content**: low-impact since the sandbox is the only thing that runs code and it has no network or filesystem access. But worth being aware of.
- **Neon cold-start (~1s)**: acceptable; dominated by LLM latency. If annoying, ping the DB on backend startup to warm it.
- **LLM prose quality on narratives**: Haiku may produce flat reports. Mitigation: invest in good prompts with strong few-shot examples. Escalate to Sonnet for narratives only if needed (one-line config change).
- **Multi-provider integration friction**: routing across Anthropic + Google adds complexity. Worth it for the cost savings and as a portfolio talking point, but expect to spend a day debugging caching and rate limit semantics across both SDKs.
