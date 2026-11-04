# RepoLens AI developer shortcuts (bash/GNU make; on Windows use Git Bash or run the
# equivalent commands from README.md directly).
.PHONY: help setup db-up db-down migrate api worker frontend tokenizer test-unit test-db lint format-check frontend-check

SHELL := /bin/bash

ifeq ($(OS),Windows_NT)
  VENV_BIN := .venv/Scripts
else
  VENV_BIN := .venv/bin
endif
PYTHON := $(VENV_BIN)/python
TIKTOKEN_CACHE_DIR ?= .test-tmp/tokenizer-cache
export TIKTOKEN_CACHE_DIR

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install backend (dev+mcp) and frontend dependencies
	python -m venv .venv || python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,mcp]"
	npm --prefix frontend ci

db-up: ## Start the local pgvector database (requires .env.compose)
	docker compose --env-file .env.compose up -d db

db-down: ## Stop the local database container (data volume is kept)
	docker compose --env-file .env.compose stop db

migrate: ## Apply Alembic migrations to DATABASE_URL (from .env)
	$(PYTHON) -m alembic upgrade head

api: ## Run the FastAPI server on 127.0.0.1:8000
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

worker: ## Run one durable job worker
	$(PYTHON) -m workers.run

frontend: ## Run the Vite dev server on http://localhost:5173 (proxies /api to :8000)
	npm --prefix frontend run dev

tokenizer: ## Download and verify the pinned cl100k_base tokenizer cache
	$(PYTHON) scripts/prepare_tokenizer.py --cache-dir $(TIKTOKEN_CACHE_DIR)

test-unit: ## Unit tests (7 ingestion-pipeline cases also need REPOLENS_TEST_DATABASE_URL)
	$(PYTHON) -m pytest -q tests/unit

test-db: ## Full backend suite; requires REPOLENS_TEST_DATABASE_URL=...repolens_verify_* (disposable)
	@test -n "$$REPOLENS_TEST_DATABASE_URL" || (echo "Set REPOLENS_TEST_DATABASE_URL to a disposable repolens_verify_* database" && exit 1)
	DATABASE_URL="$$REPOLENS_TEST_DATABASE_URL" $(PYTHON) -m pytest -q tests \
		--deselect tests/integration/test_retrieval_quality.py::test_live_local_retrieval_quality

lint: ## Ruff correctness checks (syntax errors, undefined names) as enforced in CI
	$(PYTHON) -m ruff check --select E9,F63,F7,F82  .

format-check: ## Full Ruff lint + format report (repository has known inherited style debt)
	-$(PYTHON) -m ruff check  --statistics .
	-$(PYTHON) -m ruff format --check  .

frontend-check: ## Frontend tests, TypeScript check and production build
	npm --prefix frontend test -- --run
	npm --prefix frontend exec -- tsc --noEmit -p frontend/tsconfig.json
	npm --prefix frontend run build

.DEFAULT_GOAL := help
