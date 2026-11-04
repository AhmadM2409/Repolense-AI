"""Verification-only process crash after a real committed worker checkpoint."""

import asyncio
import os

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.services import durable_investigation as durable
from core.services import job_queue as queue
from core.services.investigation import Decision
from workers.run import run_attempt


async def main():
    from core.config import settings

    settings.durable_investigation_enabled = True
    engine = create_async_engine(
        os.environ["DATABASE_URL"],
        echo=False,
        connect_args={"server_settings": {"search_path": os.environ["VERIFY_SCHEMA"]}},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    identity = os.environ["VERIFY_JOB"]
    original = durable.Journal.save

    async def crash_after_commit(self, key, value):
        await original(self, key, value)
        if key == os.environ["VERIFY_CHECKPOINT"]:
            os._exit(73)  # Deliberate hard process boundary; no graceful cleanup.

    durable.Journal.save = crash_after_commit

    class Planner:
        provider = None

        def set_readiness(self, value):
            pass

        async def __call__(self, objective, steps):
            if steps and os.environ["VERIFY_CHECKPOINT"] == "plan/1":
                return Decision(operation="answer")
            return Decision(operation="search", query="guard" if not steps else "GATE")

    async def execute(factory, attempt):
        if os.environ["VERIFY_CHECKPOINT"] == "inference":

            class Provider:
                budget = durable.PrepaidBudget()

                async def generate(self, messages, before, **kwargs):
                    await before()
                    os._exit(73)

            async def before():
                pass

            journal = durable.Journal(attempt)
            journal.phase = "plan/0"
            await durable.DurableProvider(journal, Provider()).generate([], before)
        async with factory() as db:
            await durable.process(
                db,
                identity,
                attempt.user_id,
                attempt=attempt,
                make_planner=lambda journal, deadline: durable.DurablePlanner(journal, Planner()),
            )

    async with factory() as db:
        attempt = await queue.claim(db, "crash-child", identity)
    if attempt is None and os.environ.get("VERIFY_EXPECT_TERMINAL") == "1":
        from core.models import Investigation, Review

        async with factory() as db:
            assert (await db.get(Review, identity)).status == "complete"
            assert (await db.get(Investigation, identity)).outcome == "completed"
        os._exit(74)
    assert attempt is not None
    await run_attempt(factory, attempt, execute)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
