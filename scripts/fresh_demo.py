"""Normal (writable) local RepoLens launcher for one fresh, owner-authorized end-to-end run.

Separate from scripts/local_demo.py, which stays read-only. This launcher enables career
processing; every hosted call still requires per-request consent in the UI and is limited by
the persistent budget (12 per assessment, 40 per owner, 100 aggregate, 10 per minute).

Commands:
  check                       read-only readiness report (no writes, no inference)
  migrate --confirm-database  alembic upgrade head on the configured database
  api | worker | frontend     start one component (separate terminals)
  index --input <id>          request local Nomic indexes for a prepared input's revisions
"""

import argparse
import asyncio
import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_demo import load_config  # noqa: E402  (shared loopback/RepoLens-only guard)

DEFAULT_CONFIG = ".test-tmp/phase29/private.json"
GEMINI_MODEL = "gemini-3.5-flash-lite"


def environment(value):
    """Process environment for the normal app. Secrets stay in ignored files."""
    env = dict(os.environ)
    env.update(
        APP_ENV="production",
        DATABASE_URL=value["url"],
        SECRET_KEY=value["secret_key"],
        CAREER_J2_ENABLED="true",
        QA_GEMINI_ENABLED="false",
        INVESTIGATION_ENABLED="false",
        DURABLE_INVESTIGATION_ENABLED="false",
        EMBEDDING_PROVIDER="lmstudio",
        EMBEDDING_HOSTED_SOURCE_ALLOWED="false",
        SEMANTIC_SEARCH_HOSTED_QUERY_ALLOWED="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("GEMINI_API_KEY", None)  # Read only from the ignored .env.gemini.local.
    return env


def private_file_status():
    """Shape of ignored secret files, never their values."""
    gemini = {}
    path = ROOT / ".env.gemini.local"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, val = line.partition("=")
            if sep:
                gemini[key.strip()] = val.strip().strip("\"'")
    dotenv = {}
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
            key, sep, val = line.partition("=")
            if sep:
                dotenv[key.strip()] = val.strip()
    return {
        "gemini_key_present": bool(gemini.get("GEMINI_API_KEY")),
        "gemini_model_ok": gemini.get("GEMINI_MODEL") == GEMINI_MODEL,
        "github_token_present": bool(dotenv.get("GITHUB_TOKEN")),
        "dotenv_only_known_keys": set(dotenv) <= {"GITHUB_TOKEN"},
    }


def nomic_status():
    """Read local LM Studio model metadata only (GET, no embedding request)."""
    from core.config import settings

    base = settings.embedding_lmstudio_base_url.removesuffix("/v1")
    host = base.split("://", 1)[-1].split(":", 1)[0]
    if host not in {"localhost", "127.0.0.1"}:
        return {"status": "refused_non_loopback"}
    try:
        with urllib.request.urlopen(base + "/api/v0/models", timeout=5) as response:
            models = json.loads(response.read())["data"]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return {"status": "unavailable"}
    match = [m for m in models if m.get("id") == settings.embedding_lmstudio_model]
    if len(match) != 1:
        return {"status": "model_missing"}
    m = match[0]
    ok = (
        m.get("type") == "embeddings"
        and m.get("state") == "loaded"
        and isinstance(m.get("loaded_context_length"), int)
        and m["loaded_context_length"] >= settings.embedding_max_input_tokens
        and m.get("quantization") == settings.embedding_lmstudio_quantization
    )
    return {
        "status": "ready" if ok else "mismatch_or_not_loaded",
        "model": m.get("id"),
        "state": m.get("state"),
        "loaded_context_length": m.get("loaded_context_length"),
        "quantization": m.get("quantization"),
    }


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


async def database_status(value):
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    head = ScriptDirectory.from_config(cfg).get_current_head()
    engine = create_async_engine(value["url"])
    try:
        async with engine.connect() as db:
            await db.execute(text("SET TRANSACTION READ ONLY"))
            version = (await db.execute(text("SELECT version_num FROM alembic_version"))).scalar()
            vector = (
                await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
            ).scalar()
            jobs = (
                await db.execute(
                    text("SELECT count(*) FROM reviews WHERE status IN ('pending','processing')")
                )
            ).scalar()
            budget = (
                await db.execute(text("SELECT spent, stopped, owners::text FROM career_ai_budget"))
            ).one()
            await db.rollback()
    finally:
        await engine.dispose()
    owners = json.loads(budget[2])
    return {
        "alembic_database": version,
        "alembic_code_head": head,
        "migration_needed": version != head,
        "pgvector": vector,
        "nonterminal_jobs": jobs,
        "aggregate_spent": budget[0],
        "aggregate_remaining": 100 - budget[0],
        "owner_spent": sorted(owners.values(), reverse=True),
        "stopped": budget[1],
    }


def check(config, skip_nomic, api_port, frontend_port):
    value, url = load_config(config)
    report = {
        "database": url.database,
        "host": f"{url.host}:{url.port}",
        **asyncio.run(database_status(value)),
        **private_file_status(),
        "nomic": {"status": "skipped"} if skip_nomic else nomic_status(),
        "ports_free": {
            str(api_port): port_free(api_port),
            str(frontend_port): port_free(frontend_port),
        },
    }
    report["ready"] = bool(
        not report["migration_needed"]
        and report["pgvector"]
        and not report["stopped"]
        and report["aggregate_remaining"] >= 12
        and report["gemini_key_present"]
        and report["gemini_model_ok"]
        and report["dotenv_only_known_keys"]
        and all(report["ports_free"].values())
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


def migrate(config, confirm):
    value, url = load_config(config)
    if confirm != url.database:
        raise SystemExit(f"Refusing: pass --confirm-database {url.database} to migrate it")
    from alembic.config import Config

    from alembic import command

    os.environ["DATABASE_URL"] = value["url"]
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", value["url"].replace("%", "%%"))
    command.upgrade(cfg, "head")
    print(json.dumps({"migrated": url.database, "to": "head"}))


def run(args, env):
    return subprocess.call(args, cwd=ROOT, env=env)


def index(config, input_id, email, api_port):
    load_config(config)
    base = f"http://127.0.0.1:{api_port}"
    password = getpass.getpass("RepoLens password for " + email + ": ")
    form = f"username={urllib.request.quote(email)}&password={urllib.request.quote(password)}"
    with urllib.request.urlopen(urllib.request.Request(base + "/auth/login", form.encode())) as r:
        token = json.loads(r.read())["access_token"]

    def call(path, body=None):
        request = urllib.request.Request(
            base + path,
            None if body is None else json.dumps(body).encode(),
            {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    data = call(f"/career-inputs/{input_id}")
    if data["status"] != "ready":
        raise SystemExit(f"Input status is {data['status']}; finish preparation first")
    for item in data["selection"]:
        result = call(
            "/retrieval-indexes", {"ingestion_id": item["job_id"], "configuration": "lmstudio"}
        )
        print(
            json.dumps(
                {k: result[k] for k in ("index_id", "state", "expected_chunks", "repository_id")}
            )
        )
    print(
        "Indexes queued. A running worker embeds them with local Nomic; poll GET /retrieval-indexes/<id>."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", choices=["check", "migrate", "api", "worker", "frontend", "index"]
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--skip-nomic", action="store_true")
    parser.add_argument("--confirm-database")
    parser.add_argument("--input")
    parser.add_argument("--email")
    args = parser.parse_args()
    if args.command == "check":
        raise SystemExit(check(args.config, args.skip_nomic, args.api_port, args.frontend_port))
    if args.command == "migrate":
        return migrate(args.config, args.confirm_database)
    if args.command == "index":
        if not args.input or not args.email:
            raise SystemExit("index requires --input <career input id> and --email")
        return index(args.config, args.input, args.email, args.api_port)
    value, _ = load_config(args.config)
    env = environment(value)
    if args.command == "api":
        if args.api_port != 8000:
            raise SystemExit("The Vite dev proxy targets port 8000; keep --api-port 8000")
        raise SystemExit(
            run(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                env,
            )
        )
    if args.command == "worker":
        raise SystemExit(run([sys.executable, "-m", "workers.run"], env))
    npm = shutil.which("npm") or "npm"
    raise SystemExit(
        run(
            [
                npm,
                "--prefix",
                "frontend",
                "run",
                "dev",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.frontend_port),
                "--strictPort",
            ],
            env,
        )
    )


if __name__ == "__main__":
    main()
