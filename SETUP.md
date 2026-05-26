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
```

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

- FastAPI scaffold with `/health` endpoint
- `VectorStore` protocol + `InMemoryVectorStore` (tests, dev)
- `LLM` task router + provider Protocol + `FakeProvider` (no real SDK wired)
- `find_similar` primitive (vector + player_id paths)
- `write` primitive (uses LLM router via FakeProvider in tests)
- Stat translation layer (NCAA → NBA, with placeholder coefficients)

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
