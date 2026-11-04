"""Deterministic contract double; vectors establish no semantic quality."""

import asyncio
import hashlib

from core.services import retrieval_indexing as indexing
from tests.analysis_fixtures import verified_ingestion


class FakeProvider:
    def __init__(self, config):
        self.config = config

    async def embed_documents(self, texts, before_request):
        await before_request()
        await asyncio.sleep(0)
        result = []
        for text in texts:
            vector = [0.0] * self.config.dimensions
            vector[
                int.from_bytes(hashlib.sha256(text.encode()).digest()[:2], "big")
                % self.config.dimensions
            ] = 1.0
            result.append(vector)
        return result


async def ingested(database, actors, index=0, files=None):
    async with database() as db:
        return await verified_ingestion(
            db,
            actors["profiles"][index].id,
            actors["users"][index].id,
            files
            or {"a.py": b"def cache_user():\n    return 1\n", "README.md": b"Cache project\n"},
        )


async def submitted(database, actors, index=0, files=None, configuration="default"):
    job = await ingested(database, actors, index, files)
    async with database() as db:
        ident, created = await indexing.submit(db, actors["users"][index].id, job.id, configuration)
        assert created
    return ident, job
