"""Loopback saved-case demo. No model, GitHub or other remote dispatch is permitted.

Use an ignored RepoLens-only JSON config containing url and secret_key. Does not
seed results, modify budgets, start Docker or control existing processes.
"""

import argparse
import asyncio
import http.client
import json
import mimetypes
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "Saved demo is read-only. New discovery, preparation and inference require separate authorization."


def load_config(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or "pathreview" in str(path.relative_to(ROOT)).lower():
        raise ValueError("Use an ignored configuration inside RepoLens only")
    value = json.loads(path.read_text())
    url = make_url(value["url"])
    if (
        url.drivername != "postgresql+asyncpg"
        or url.host not in {"localhost", "127.0.0.1"}
        or url.port not in {5434, 15470}
        or not (url.database or "").startswith("repolens")
        or "pathreview" in (url.database or "").lower()
        or not url.password
        or len(value.get("secret_key", "")) < 32
    ):
        raise ValueError("Dedicated loopback RepoLens database and strong secret required")
    return value, url


def configure(value):
    # Explicit values win over inherited settings; never load Gemini/GitHub secrets.
    os.environ.update(
        APP_ENV="production",
        DATABASE_URL=value["url"],
        SECRET_KEY=value["secret_key"],
        CAREER_J2_ENABLED="false",
        QA_GEMINI_ENABLED="false",
        INVESTIGATION_ENABLED="false",
        DURABLE_INVESTIGATION_ENABLED="false",
        PYTHONDONTWRITEBYTECODE="1",
        GITHUB_TOKEN="",
        GEMINI_API_KEY="",
    )


def install_network_guard(url, api_port, frontend_port):
    previous = socket.getaddrinfo
    allowed = {url.port, api_port, frontend_port}

    def guarded(host, port, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else str(host)
        if name.lower() not in {"localhost", "127.0.0.1", "::1"} or int(port) not in allowed:
            raise RuntimeError("Saved demo denies model and remote network requests")
        return previous(host, port, *args, **kwargs)

    socket.getaddrinfo = guarded


def mutation_allowed(method, path):
    return method in {"GET", "HEAD", "OPTIONS"} or method == "POST" and path == "/auth/login"


async def check(idle=False):
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import func, select, text

    from core.database import AsyncSessionLocal, engine
    from core.models import CareerAIBudget, Review

    try:
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "alembic"))
        head = ScriptDirectory.from_config(cfg).get_current_head()
        async with engine.connect() as db:
            versions = (
                (await db.execute(text("SELECT version_num FROM alembic_version"))).scalars().all()
            )
            vector = (
                await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
            ).scalar()
            if versions != [head] or not vector:
                raise RuntimeError("RepoLens needs current Alembic migrations and pgvector")
        async with AsyncSessionLocal() as db:
            pending = await db.scalar(
                select(func.count())
                .select_from(Review)
                .where(Review.status.in_(["pending", "processing"]))
            )
            budget = await db.get(CareerAIBudget, 1)
            if idle and pending:
                raise RuntimeError(
                    "Worker demo requires an idle queue; existing work was not changed"
                )
            result = {
                "status": "ready",
                "alembic_head": head,
                "pgvector": vector,
                "nonterminal_jobs": pending,
                "historical_inference_spent": budget.spent if budget else None,
                "mutations": "login_only",
                "models": "disabled",
                "remote_network": "denied",
            }
            print(json.dumps(result))
    finally:
        await engine.dispose()


def api(port):
    import uvicorn
    from fastapi.responses import JSONResponse

    from api.main import app

    @app.middleware("http")
    async def saved_only(request, call_next):
        if not mutation_allowed(request.method, request.url.path):
            return JSONResponse({"detail": MESSAGE}, status_code=423)
        return await call_next(request)

    uvicorn.run(app, host="127.0.0.1", port=port)


def frontend(port, api_port):
    root = (ROOT / "frontend/dist").resolve()
    if not (root / "index.html").is_file():
        raise RuntimeError("Run npm --prefix frontend run build first")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def proxy(self):
            path = self.path[4:]
            if not mutation_allowed(self.command, urlsplit(path).path):
                data = json.dumps({"detail": MESSAGE}).encode()
                self.send_response(423)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65536:
                self.send_error(413)
                return
            body = self.rfile.read(length) if length else None
            headers = {
                k: v for k, v in self.headers.items() if k.lower() not in {"host", "connection"}
            }
            connection = http.client.HTTPConnection("127.0.0.1", api_port, timeout=15)
            try:
                connection.request(self.command, path, body, headers)
                response = connection.getresponse()
                data = response.read()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
            except OSError:
                self.send_error(502, "Demo API unavailable; start the dedicated API")
            finally:
                connection.close()

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path.startswith("/api/"):
                return self.proxy()
            target = (root / path.lstrip("/")).resolve()
            if not target.is_relative_to(root):
                self.send_error(404)
                return
            if not target.is_file():
                if Path(path).suffix:
                    self.send_error(404)
                    return
                target = root / "index.html"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            )
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:",
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        do_HEAD = do_GET  # noqa: N815 - BaseHTTPRequestHandler protocol

        def do_POST(self):
            return self.proxy() if self.path.startswith("/api/") else self.send_error(403)

        do_PUT = do_POST  # noqa: N815 - BaseHTTPRequestHandler protocol
        do_PATCH = do_POST  # noqa: N815 - BaseHTTPRequestHandler protocol
        do_DELETE = do_POST  # noqa: N815 - BaseHTTPRequestHandler protocol

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        print(f"Saved-case demo: http://127.0.0.1:{port}/career", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "migrate", "api", "worker", "frontend"])
    parser.add_argument("--config", default=".test-tmp/phase29/private.json")
    parser.add_argument("--api-port", type=int, default=8020)
    parser.add_argument("--frontend-port", type=int, default=5190)
    parser.add_argument("--once", action="store_true", help="Idle worker startup check then exit")
    args = parser.parse_args()
    for port in [args.api_port, args.frontend_port]:
        if port not in range(1024, 65536) or port in {1234, 1235, 5433, 5434, 15470}:
            raise ValueError("Choose separate unprivileged demo HTTP ports")
    if args.api_port == args.frontend_port:
        raise ValueError("API/frontend ports must differ")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    value, url = load_config(args.config)
    configure(value)
    install_network_guard(url, args.api_port, args.frontend_port)
    if args.command == "migrate":
        from alembic.config import Config

        from alembic import command

        cfg = Config(str(ROOT / "alembic.ini"))
        command.upgrade(cfg, "head")
    elif args.command == "check":
        asyncio.run(check())
    elif args.command == "api":
        api(args.api_port)
    elif args.command == "frontend":
        frontend(args.frontend_port, args.api_port)
    else:
        asyncio.run(check(idle=True))
        from workers.run import main as worker_main

        asyncio.run(worker_main(args.once))


if __name__ == "__main__":
    main()
