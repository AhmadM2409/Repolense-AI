"""Verified multi-file fixture factory: real ingestion/SQL; deterministic external HTTP only."""

import httpx

from core.services import github_ingestion
from ingestion.github import GitHubClient
from tests.github_responses import GitHubResponses

FILES = {
    "README.md": b"# Fixture project\nA deterministic test fixture.\n",
    "pyproject.toml": b'[project]\nname = "fixture-project"\nversion = "1.0"\ndependencies = ["fastapi>=0.100", "pytest>=7"]\n[project.scripts]\nfixture = "app:main"\n',  # noqa: E501 - exact source fixture
    "package.json": b'{"dependencies":{"react":"^18"},"devDependencies":{"vitest":"^1"},"scripts":{"start":"node index.js"}}\n',  # noqa: E501 - exact source fixture
    ".github/workflows/ci.yml": b"name: fixture\non: push\njobs: {}\n",
    "Dockerfile": b"FROM python:3.12-slim\nCOPY . /app\n",
    "app.py": b"import subprocess\nimport pickle\n\ndef main(value, cursor):\n    subprocess.run(value, shell=True)\n    cursor.execute(f'SELECT * FROM users WHERE name={value}')\n    pickle.loads(value)\n    try:\n        pass\n    except:\n        pass\n\nif __name__ == '__main__':\n    main(None, None)\n",  # noqa: E501 - exact source fixture
    "tests/test_app.py": b"import unittest\nclass Example(unittest.TestCase):\n    pass\n",
}


async def verified_ingestion(db, profile_id, user_id, files=None):
    fixture = GitHubResponses(files or FILES)
    job, _ = await github_ingestion.submit(db, profile_id, user_id, "example/project")
    async with GitHubClient(transport=httpx.MockTransport(fixture)) as client:
        await github_ingestion.process(db, job.id, user_id, client)
    await db.refresh(job)
    assert job.status == "complete", job.error_message
    return job
