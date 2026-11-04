"""Normal-run launcher safety; no network, database, provider or subprocess."""

import json

import pytest

from scripts import fresh_demo
from scripts.local_demo import ROOT

GOOD = "postgresql+asyncpg://repolens_eval:pw@127.0.0.1:15470/repolens_eval_phase29"


@pytest.fixture
def config(tmp_path_factory):
    folder = ROOT / ".test-tmp" / "unit-fresh-demo"
    folder.mkdir(parents=True, exist_ok=True)

    def write(url, secret="s" * 40):
        path = folder / "private.json"
        path.write_text(json.dumps({"url": url, "secret_key": secret}))
        return str(path)

    yield write
    (folder / "private.json").unlink(missing_ok=True)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://u:pw@127.0.0.1:5433/pathreview_dev",
        "postgresql+asyncpg://u:pw@10.0.0.5:15470/repolens_eval",
        "postgresql+asyncpg://u:pw@127.0.0.1:15470/other_app",
        "postgresql://u:pw@127.0.0.1:15470/repolens_eval",
    ],
)
def test_refuses_non_repolens_or_non_loopback_databases(config, url):
    with pytest.raises(ValueError):
        fresh_demo.load_config(config(url))


def test_migration_requires_exact_database_confirmation(config):
    path = config(GOOD)
    with pytest.raises(SystemExit, match="--confirm-database repolens_eval_phase29"):
        fresh_demo.migrate(path, "repolens_eval")
    with pytest.raises(SystemExit):
        fresh_demo.migrate(path, None)


def test_environment_enables_only_career_and_never_forwards_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-pass")
    env = fresh_demo.environment({"url": GOOD, "secret_key": "s" * 40})
    assert env["CAREER_J2_ENABLED"] == "true"
    assert env["QA_GEMINI_ENABLED"] == env["INVESTIGATION_ENABLED"] == "false"
    assert env["DURABLE_INVESTIGATION_ENABLED"] == "false"
    assert (
        env["EMBEDDING_HOSTED_SOURCE_ALLOWED"]
        == env["SEMANTIC_SEARCH_HOSTED_QUERY_ALLOWED"]
        == "false"
    )
    assert env["DATABASE_URL"] == GOOD and "GEMINI_API_KEY" not in env


def test_private_file_status_reports_shape_not_values():
    status = fresh_demo.private_file_status()
    assert set(status) == {
        "gemini_key_present",
        "gemini_model_ok",
        "github_token_present",
        "dotenv_only_known_keys",
    }
    assert all(isinstance(v, bool) for v in status.values())
