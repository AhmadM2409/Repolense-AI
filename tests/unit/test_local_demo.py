"""Demo config/network/mutation boundaries. No actual credentials or remote requests."""

import json
import socket

import pytest
from sqlalchemy.engine import make_url

from scripts import local_demo


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://u:p@127.0.0.1:5433/pathreview",
        "postgresql+asyncpg://u:p@127.0.0.1:15470/pathreview",
        "postgresql+asyncpg://u:p@remote.example:5434/repolens_ai",
        "sqlite:///repolens.db",
        "postgresql+asyncpg://u@127.0.0.1:15470/repolens_ai",
    ],
)
def test_foreign_or_unsafe_database_is_rejected(tmp_path, monkeypatch, url):
    monkeypatch.setattr(local_demo, "ROOT", tmp_path)
    path = tmp_path / "private.json"
    path.write_text(json.dumps({"url": url, "secret_key": "x" * 40}))
    with pytest.raises(ValueError):
        local_demo.load_config(path)


def test_config_scope_secret_and_valid_dedicated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(local_demo, "ROOT", tmp_path)
    path = tmp_path / "private.json"
    path.write_text(
        json.dumps(
            {
                "url": "postgresql+asyncpg://u:p@127.0.0.1:15470/repolens_eval_phase29",
                "secret_key": "x" * 40,
            }
        )
    )
    _, url = local_demo.load_config(path)
    assert url.database == "repolens_eval_phase29"
    path.write_text(json.dumps({"url": str(url), "secret_key": "short"}))
    with pytest.raises(ValueError):
        local_demo.load_config(path)
    with pytest.raises(ValueError):
        local_demo.load_config(tmp_path.parent / "outside.json")


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/career-inputs"),
        ("POST", "/career-assessments/x/roadmap"),
        ("PUT", "/career-inputs/x/selection"),
        ("POST", "/repositories"),
        ("POST", "/auth/register"),
        ("DELETE", "/profiles/x"),
    ],
)
def test_demo_never_dispatches_mutations(method, path):
    assert not local_demo.mutation_allowed(method, path)


def test_login_and_reads_are_allowed():
    assert local_demo.mutation_allowed("POST", "/auth/login")
    assert local_demo.mutation_allowed("GET", "/career-assessments/x/roadmap")
    assert not local_demo.mutation_allowed("POST", "/auth/login/other")


def test_dns_guard_denies_all_model_and_remote_endpoints(monkeypatch):
    forwarded = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: forwarded.append(args) or [])
    local_demo.install_network_guard(
        make_url("postgresql+asyncpg://u:p@127.0.0.1:15470/repolens_eval_phase29"), 8020, 5190
    )
    for host, port in [
        ("api.github.com", 443),
        ("generativelanguage.googleapis.com", 443),
        ("localhost", 1235),
        ("localhost", 1234),
        ("127.0.0.1", 5433),
    ]:
        with pytest.raises(RuntimeError):
            socket.getaddrinfo(host, port)
    socket.getaddrinfo("127.0.0.1", 15470)
    socket.getaddrinfo("127.0.0.1", 8020)
    assert len(forwarded) == 2
