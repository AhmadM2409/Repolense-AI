"""Independent PostgreSQL worker: python -m workers.run."""

import argparse
import asyncio
import signal
import socket
from contextlib import suppress
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from core.logging import configure_logging
from core.services import job_queue as queue


async def execute(factory, attempt):
    from core.services import github_ingestion, repository_analysis, retrieval_indexing

    async with factory() as db:
        if attempt.job_type == "investigation":
            from core.services import durable_investigation
            await durable_investigation.process(db, attempt.job_id, attempt.user_id, attempt=attempt)
        elif attempt.job_type == "github_ingestion":
            async with github_ingestion.make_github_client() as client:
                await github_ingestion.process(
                    db, attempt.job_id, attempt.user_id, client, attempt=attempt
                )
        elif attempt.job_type == "retrieval_index":
            await retrieval_indexing.process(db, attempt.job_id, attempt.user_id, attempt=attempt)
        elif attempt.job_type == "review":
            await repository_analysis.process(db, attempt.job_id, attempt.user_id, attempt=attempt)
        else:
            raise ValueError("Unsupported durable job type")


async def run_attempt(factory, attempt, executor=execute):
    attempt.factory = factory

    async def heartbeats():
        while True:
            await asyncio.sleep(settings.job_heartbeat_seconds)
            await attempt.checkpoint()

    task = asyncio.create_task(executor(factory, attempt))
    pulse = asyncio.create_task(heartbeats())
    try:
        done, _ = await asyncio.wait({task, pulse}, return_when=asyncio.FIRST_COMPLETED)
        for completed in done:
            completed.result()
    except asyncio.CancelledError:
        attempt.stopped.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await queue.fail(factory, attempt, category="worker_shutdown")
        raise
    except Exception as exc:
        attempt.stopped.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await queue.fail(factory, attempt, exc)
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)


async def tick(factory, worker_id, executor=execute):
    async with factory() as db:
        await queue.recover(db)
        attempt = await queue.claim(db, worker_id)
    if attempt:
        await run_attempt(factory, attempt, executor)
    return attempt is not None


async def serve(factory, stop, worker_id, once=False):
    async def slot(index):
        while not stop.is_set():
            try:
                worked = await tick(factory, worker_id + ":" + str(index))
            except Exception:
                # Never log driver exception strings (can contain credentials/SQL parameters).
                queue.log.error("worker_database_unavailable", worker_id=worker_id)
                worked = False
            if once:
                return
            if not worked:
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), settings.worker_poll_seconds)

    tasks = [asyncio.create_task(slot(i)) for i in range(settings.worker_concurrency)]
    await asyncio.gather(*tasks)


async def main(once=False):
    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    try:
        from pathlib import Path

        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from sqlalchemy import text

        cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        cfg.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "alembic"))
        async with engine.connect() as db:
            if (
                await db.execute(text("SELECT version_num FROM alembic_version"))
            ).scalar() != ScriptDirectory.from_config(cfg).get_current_head():
                raise RuntimeError("Run alembic upgrade head before starting worker")
        factory = async_sessionmaker(engine, expire_on_commit=False)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()

        def request_stop(*_):
            loop.call_soon_threadsafe(stop.set)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except NotImplementedError:
                signal.signal(sig, request_stop)
        await serve(factory, stop, socket.gethostname()[:60] + "-" + uuid4().hex, once)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Recover and execute at most one job per slot, then exit",
    )
    args = parser.parse_args()
    configure_logging()
    asyncio.run(main(args.once))
