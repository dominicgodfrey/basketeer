# Setup & Manual Tasks

Things you need to do by hand — account creation, key wiring, decisions still pending. The repo is structured so this list shrinks as decisions are made; check items off when complete.

---

## Local dev — get the test suite running

```powershell
# from repo root
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

You should see all tests pass (~56 at last commit, growing). The venv lives in `backend/.venv/` and is gitignored.

To run the app:
```powershell
uvicorn app.main:app --reload --port 8000
# then visit http://localhost:8000/health
# or POST a question:
#   curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"comps for klay thompson"}'
```

The `/ask` endpoint is wired end-to-end (classifier → agent loop → tools → response), but until you wire a real LLM provider (see below) it returns the placeholder string from `FakeProvider`.

---

## Decisions still pending

These each block a downstream module. They aren't urgent until the corresponding phase starts.

### LLM provider(s)

**Status:** Undecided. Open to DeepSeek / other Chinese models (cost), Anthropic (reliability), Google (cheap classification).

**What you need to do:**
- Decide which provider(s) you want to use for which task. The current routing table in `backend/app/llm/router.py` is a placeholder.
- Get API keys for each chosen provider.
- Add keys to `backend/.env` (see `.env.example`).
- The router supports an `openai-compatible` provider type, which covers DeepSeek, Together, Groq, Fireworks, and most Chinese providers exposing an OpenAI-shaped API — so you can swap by changing the env vars and the routing table, no code rewrite.

**When you need this:** Before the agent loop (Phase 4 of `PLAN.md`). Find_similar and translation work without it.

### Pinecone (vector store)

**Status:** Pinecone is the planned default per PLAN.md; serverless free tier sufficient at our scale.

**What you need to do:**
- Sign up at https://www.pinecone.io/ (free serverless tier — 2GB index, 1M reads/month).
- Create an index. Dimension = 50 (matches the planned stat profile schema). Metric = cosine. Cloud + region of your choice.
- Add `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` to `backend/.env`.

**When you need this:** Before Phase 2 deliverable (similarity search working end-to-end). Until then, `InMemoryVectorStore` covers all tests.

### E2B (compute sandbox)

**Status:** Planned default per CLAUDE.md.

**What you need to do:**
- Sign up at https://e2b.dev/ (free tier covers dev usage).
- Add `E2B_API_KEY` to `backend/.env`.
- Optional fallback: Docker-per-call (PLAN.md describes the contract). Pick this if E2B becomes expensive or you want to self-host.

**When you need this:** Before the `compute` primitive lands.

### Redis (result cache)

**Status:** You confirmed Redis as the cache backend.

**What you need to do:**
- Pick a Redis source: Upstash (free tier, serverless, painless), Railway Redis ($5/mo), or local Redis via Docker for dev.
- Add `REDIS_URL` to `backend/.env` (e.g. `redis://default:<pw>@<host>:<port>`).

**When you need this:** Before production result caching. `InMemoryCache` covers local dev and tests.

### Neon Postgres + DB schema (collaborator's track)

**Status:** Reserved for collaborator per PLAN.md "Division of labor". Don't start this here.

**What your collaborator will do:**
- Provision a Neon project (https://neon.tech/, free tier auto-suspends).
- Define SQLAlchemy models for `players`, `seasons`, `contracts`, `teams`.
- Set up the read-only Postgres role used by the agent's `query_stats` primitive.
- Manage Alembic migrations.
- Add `DATABASE_URL` and `DATABASE_URL_READONLY` to `backend/.env`.

**When you need this:** Before `query_stats` primitive and the ingestion pipeline.

---

## .env keys cheat sheet

See `backend/.env.example` for the template. None of these are committed to git. Add only the ones you've decided on.

| Key | Purpose | Decided? |
|---|---|---|
| `ANTHROPIC_API_KEY` | If using Claude for any task | ☐ |
| `GOOGLE_API_KEY` | If using Gemini for any task | ☐ |
| `OPENAI_COMPAT_BASE_URL` | DeepSeek / Together / Groq / etc. base URL | ☐ |
| `OPENAI_COMPAT_API_KEY` | API key for the above | ☐ |
| `PINECONE_API_KEY` | Vector store | ☐ |
| `PINECONE_INDEX_NAME` | e.g. `basketeer-stat-profiles` | ☐ |
| `E2B_API_KEY` | Compute primitive sandbox | ☐ |
| `REDIS_URL` | Result cache | ☐ |
| `DATABASE_URL` | Neon Postgres (collaborator) | ☐ |
| `DATABASE_URL_READONLY` | Agent's `query_stats` role (collaborator) | ☐ |

---

## What's already built (no action needed from you)

- FastAPI scaffold with `/health` and `/ask` endpoints
- Centralized logging (`configure_logging` + `get_logger`) — wires up on app startup
- `VectorStore` protocol + `InMemoryVectorStore` (tests, dev). Pinecone impl pending account.
- `Cache` protocol + `InMemoryCache` with TTL + `delete_prefix` (Redis impl pending `REDIS_URL`)
- Cached wrapper: `cached_find_similar` (24h TTL, sha256-stable keys)
- `Sandbox` protocol + `RaisingSandbox` (default; fails loudly) + `FakeSandbox` (tests). E2B/Docker impl pending account.
- LLM task router (env-driven via `MODEL_*` vars), `LLMProvider` protocol, `FakeProvider` with scripted multi-turn responses. Supports `anthropic`, `google`, and `openai-compatible` (DeepSeek etc.) provider types — pick at config time.
- All four primitives:
  - `find_similar` (vector + player_id paths)
  - `write` (uses LLM router, prompt at `backend/app/agents/prompts/write.md`)
  - `compute` (Pydantic contract; runs through Sandbox protocol — currently `RaisingSandbox` so calls fail until E2B/Docker is wired)
  - `query_stats` — NOT here; reserved for collaborator's DB track.
- Stat translation layer (NCAA → NBA, with placeholder coefficients)
- 50-dim stat profile schema (`STAT_DIMENSIONS`) + `build_stat_profile` (translates then embeds, tracks imputed dims). See `backend/app/data/embed/schema.py` for the full schema with source-API annotations
- Agent layer:
  - `ToolSpec` + per-provider translators (`to_anthropic_tools`, `to_openai_tools`, `to_google_tools`)
  - Tool builders for each primitive (`make_find_similar_tool`, `make_write_tool`, `make_compute_tool`)
  - **Hand-rolled ReAct loop** (`run_agent`) with guardrails (6 iterations, 30 s wall-clock, 30 K tokens) and structured tool-error recovery. No LangChain dependency.
  - **Intent classifier** (`classify`) with benign-fallback to agent path on any parse failure
- DI factories in `app/dependencies.py` (`get_vector_store`, `get_cache`, `get_sandbox`, `get_llm_provider`) — each returns the safe default impl; FastAPI's `dependency_overrides` swap them in tests.

---

## How to wire a real LLM provider (when you've picked one)

Three steps. The router and `Provider` protocol are already in place, so each new vendor is a small isolated module.

### 1. Create `backend/app/llm/providers/<vendor>.py`

Mirror the shape of `backend/app/llm/providers/fake.py`. The class must implement `complete(request: CompletionRequest) -> CompletionResponse`. Translate `request.tools` (already in the provider's native shape, courtesy of the agent layer) to whatever the SDK expects. Translate response tool calls back into our `ToolCall` records.

Per-vendor SDKs:
- **Anthropic**: `pip install anthropic`. Use `client.messages.create(...)`. Pass `cache_control={"type":"ephemeral"}` on messages where `Message.cache is True`. Populate `cache_read_input_tokens` and `cache_creation_input_tokens` on the response.
- **Google (Gemini)**: `pip install google-genai` (the new SDK; the old `google-generativeai` is being deprecated). Use `client.models.generate_content(...)`. Tool spec key: `tools=[{"function_declarations":[...]}]`. Google's tool response parsing is non-trivial; expect to spend an hour on it.
- **OpenAI-compatible** (DeepSeek / Together / Groq / Moonshot / Fireworks): `pip install openai`. Instantiate with `OpenAI(api_key=..., base_url="https://api.deepseek.com")`. Tool spec key: `tools=[{"type":"function","function":{...}}]`. Tool call response is `message.tool_calls` with `id` / `function.name` / `function.arguments` (JSON string — `json.loads` it).

### 2. Add the provider to your dependency factory

Edit `backend/app/dependencies.py`. The current `get_llm_provider()` returns `FakeProvider`. Replace with the real one. Example (DeepSeek):

```python
from app.llm.providers.openai_compatible import OpenAICompatibleProvider

@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    return OpenAICompatibleProvider(
        base_url=settings.openai_compat_base_url,
        api_key=settings.openai_compat_api_key,
    )
```

For multi-provider deployment (e.g. Gemini for classification + Anthropic for planning), `get_llm_provider` should return a `dict[str, LLMProvider]` and the agent loop will look up the right one per `ModelSpec.provider`. That's a small refactor in `app/agents/loop.py` — the comment in `dependencies.py` flags where.

### 3. Set the env vars and the routing table

In `backend/.env`:
```
# Pick a provider per task. Use ":cache" suffix for Anthropic (other providers
# auto-cache server-side and don't need the flag).
MODEL_AGENT_PLANNING=anthropic:claude-haiku-4-5-20251001:cache
MODEL_NARRATIVE_WRITE=anthropic:claude-haiku-4-5-20251001:cache
MODEL_INTENT_CLASSIFIER=openai-compatible:deepseek-chat
MODEL_TEXT_TO_SQL=openai-compatible:deepseek-chat
MODEL_CODE_GENERATION=anthropic:claude-haiku-4-5-20251001:cache

ANTHROPIC_API_KEY=...
OPENAI_COMPAT_BASE_URL=https://api.deepseek.com
OPENAI_COMPAT_API_KEY=...
```

### 4. Test

```powershell
pytest -q              # nothing should break; FakeProvider tests still cover the abstraction
# then run a live smoke test
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"comps for klay thompson"}'
```

---

## How to recover from a confused state

```powershell
# wipe venv and reinstall
Remove-Item -Recurse -Force backend\.venv
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

If a test fails after pulling new code, that's a real failure — open an issue or ping me.
