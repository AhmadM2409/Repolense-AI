# RepoLens AI

An AI-powered developer portfolio and job-fit analysis platform. RepoLens compares a job description with selected public GitHub repositories, explains what the inspected source demonstrates, and proposes evidence-based improvements. It does not assign arbitrary fit percentages or treat missing repository evidence as a missing personal skill.

## Workflow and features

1. Paste a job description and GitHub profile URL.
2. Discover public repositories, select up to five, and prepare verified snapshots pinned to immutable commits.
3. Extract reviewable technical requirements with exact job-description quotes.
4. Retrieve source evidence using keyword/BM25, local semantic search or hybrid reciprocal-rank fusion.
5. Inspect requirement-by-requirement results: **demonstrated**, **partially demonstrated**, **documented/claimed**, or **not evidenced**. Expand canonical repository, commit, file and line citations.
6. Generate a prioritized improvement roadmap linked to requirements, observed evidence and identified gaps, with observable completion criteria and uncertainty.

Supporting Repository Intelligence includes deterministic source analysis, grounded Q&A, persisted investigations and a local read-only MCP server. Authentication, ownership checks, revision isolation, explicit hosted-processing consent, canonical citation validation and persistent inference budgets protect these workflows. PostgreSQL-backed workers handle durable ingestion, analysis, indexing and supported investigations; Career Intelligence extraction/matching steps are explicit API actions, without a second queue or automatic model retries.

## Architecture and stack

- **Frontend:** React, TypeScript, Tailwind CSS and Vite.
- **API:** Python, FastAPI, Pydantic and JWT authentication.
- **Persistence:** PostgreSQL 16, pgvector, SQLAlchemy and Alembic migrations through **012**.
- **Retrieval:** verified source documents/chunks, BM25, Nomic-compatible local embeddings, pgvector and hybrid RRF.
- **Generation:** opt-in server-side Gemini integration; existing local LM Studio/Qwen integration supports Repository Intelligence. These providers have different acceptance limitations; none is silently substituted for another.
- **Workers:** independent PostgreSQL-backed job claiming, leases, fencing, retries, progress and cooperative cancellation.

The browser calls the API, which enforces user/repository/revision scope before retrieval or publication. Workers use the same canonical database contracts. Models select bounded evidence references; server-side validation binds published citations to stored source identity and ranges. Redis and the inherited external vector-service settings are not required for the current PostgreSQL workflow.

## Local setup

Prerequisites: Python 3.11+, Node **22.12+ within v22**, **24+ within v24**, or **26+**, npm, Git, and Docker/Compose for a dedicated PostgreSQL/pgvector database. Python3.13/Node24 are used by CI. No model service is required for deterministic unit tests. Use a separate database; never reuse another application's data.

PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,mcp]'
npm --prefix frontend ci
Copy-Item .env.example .env
Copy-Item .env.compose.example .env.compose
```

Edit the ignored environment files before starting. Replace every `CHANGE_ME` placeholder. Use a random, URL-safe database password in `.env.compose` and the corresponding password in `.env`'s `DATABASE_URL`; URL-encode reserved password characters if necessary. Generate a unique signing secret locally, keep it private, and share it only between your API/worker processes.

| Variable | Location and purpose |
|---|---|
| `DATABASE_URL` | `.env`: `postgresql+asyncpg://repolens:<PASSWORD>@127.0.0.1:5435/repolens_dev` |
| `SECRET_KEY` | `.env`: strong random signing secret, at least32bytes for production mode |
| `APP_ENV` | `.env`: local development or deliberately configured production mode; production rejects missing/placeholder signing secrets |
| `REPOLENS_DB_PASSWORD` | `.env.compose` only: database password; do not add this unknown key to application `.env` |
| `GITHUB_TOKEN` | Optional server-side token in `.env` for public-repository API quota; empty permits anonymous access subject to GitHub limits |
| `CAREER_J2_ENABLED` | `.env`: defaults `false`; explicitly set `true` only when authorizing Career Intelligence inference |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Private process environment, or ignored `.env.gemini.local`; model `gemini-3.5-flash-lite`. Never use frontend/Vite variables for secrets |
| `EMBEDDING_LMSTUDIO_BASE_URL`, `EMBEDDING_LMSTUDIO_MODEL` | Local server configuration for semantic/hybrid retrieval; default model `text-embedding-nomic-embed-text-v1.5` |

Keep API keys, database credentials, private configuration and inference ledgers out of Git. `.env.example` lists other supported bounds; unknown application `.env` keys fail validation. Do not put Gemini secrets in `.env` because the hosted adapter uses its explicit private source. Hosted keys alone do not authorize calls: provider flags, per-action UI consent and server budgets still apply. Confirm quota/billing independently before live use. Current career limits include12attempts per assessment,40per owner,100aggregate and10per minute; failures/reservations consume their applicable allowance and reopening does not reset it.

Start the dedicated local database and migrate it:

```powershell
docker compose -p repolens-public-local --env-file .env.compose up -d db
python -m alembic upgrade head
```

Port5435 must be unused. The explicit Compose project name separates this setup's volume from other checkouts. Check the configured target before migration; migrations change only the database you supply. `docker-compose.jobs.yml` is an alternative containerized API/worker stack, not required alongside the host setup; do not start both on conflicting ports.

Run these in separate activated terminals from the repository root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
python -m workers.run
npm --prefix frontend run dev
```

Open **http://localhost:5173/career**, register a local account and log in. Vite proxies `/api` to port8000. API health: **http://127.0.0.1:8000/health**; API schema: **http://127.0.0.1:8000/docs**. The `/docs` URL is FastAPI's generated interface, not a repository documentation directory. Alembic is authoritative; runtime `create_all` is not used.

Repository preparation advances sequentially and may require explicit progress actions. Review extracted requirements and explicitly authorize each matching step. Start with keyword retrieval if no local embedding model/index is ready. Semantic/hybrid modes require compatible ready revision indexes and the configured local Nomic server; they do not automatically fall back. Do not execute downloaded source. Refresh/history reopening retrieves saved results without generation.

Stop API/worker/frontend terminals with Ctrl+C. To stop only this setup's database while preserving its volume:

```powershell
docker compose -p repolens-public-local --env-file .env.compose stop db
```

## Tests and build

```powershell
$env:TIKTOKEN_CACHE_DIR = "$PWD/.test-tmp/tokenizer-cache"
python scripts/prepare_tokenizer.py --cache-dir $env:TIKTOKEN_CACHE_DIR
python scripts/prepare_tokenizer.py --cache-dir $env:TIKTOKEN_CACHE_DIR --check
python -m pytest -q tests/unit -m 'not integration'
python -m ruff check --select E9,F63,F7,F82 .
npm --prefix frontend test -- --run
node frontend/node_modules/typescript/bin/tsc --noEmit --project frontend/tsconfig.json
npm --prefix frontend run build
```

Tokenizer preparation downloads one pinned, SHA256-verified tokenizer asset on first use; `--check` is offline and fails if the cache is absent/corrupt. Do not replace production tokenization with estimated counts. The7database-backed ingestion tests under `tests/unit` are marked `integration`; they also run in the PostgreSQL command below. This selection separates infrastructure requirements, not failing assertions.

Database tests need a **disposable** PostgreSQL/pgvector database named `repolens_verify_*`. They create/migrate/drop isolated test schemas; never point them at application data:

```powershell
$env:REPOLENS_TEST_DATABASE_URL = 'postgresql+asyncpg://<USER>:<PASSWORD>@127.0.0.1:<PORT>/repolens_verify_tests'
$env:DATABASE_URL = $env:REPOLENS_TEST_DATABASE_URL
python -m pytest -q tests/integration/test_fresh_end_to_end.py
python -m pytest -q tests --deselect tests/integration/test_retrieval_quality.py::test_live_local_retrieval_quality
```

The offline end-to-end pathway uses real API/authentication, PostgreSQL/pgvector, migrations, workers, retrieval, citations, matching and roadmap persistence. **GitHub HTTP, Gemini and embeddings are deterministic test doubles**, not evidence of live model accuracy. Explicitly opt-in live local-provider tests require separate infrastructure; they are not part of normal CI. CI does not need hosted provider secrets. Broader Ruff/style/format debt remains; CI's correctness lint is distinguished from a fully clean style baseline.

## Status and limitations

**Active development.** An offline end-to-end test pathway is verified; further live-provider quality, robustness and complete fresh-user validation remain pending. Earlier bounded public-repository demonstrations do not establish general matching reliability. Retrieval can miss relevant code; documentation claims, delegation and incomplete evidence must not automatically establish a capability. Canonical citation validity is distinct from semantic claim support.

The system abstains where inspected evidence is insufficient. A roadmap is proposed work, not proof of a developer's current skills, deployed systems or future outcomes. Personal skill absence, runtime deployment success and effort estimates must not be inferred from missing repository evidence. Some local-provider compatibility, prompt-injection/completeness and broad-quality cases remain unresolved. This is a local portfolio/development project, not production-ready hiring automation.

## Attribution and licensing

RepoLens AI derives from [PathReview](https://github.com/ascherj/pathreview), through the [development fork](https://github.com/AhmadM2409/pathreview). This public copy starts with fresh Git history; original contributors retain credit for inherited work. RepoLens-specific capabilities were developed separately from the inherited scaffold.

**Licensing remains unresolved before public redistribution.** Upstream declares MIT but lacks an authoritative standalone license/copyright notice in the supplied provenance. No copyright owner or permission has been invented; no standalone LICENSE is supplied. See [NOTICE.md](NOTICE.md) for attribution, evidence limits and the precise decision required before publication. Dependencies retain their own licenses.
