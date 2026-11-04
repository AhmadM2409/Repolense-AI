"""Strict MCP argument contracts; no model or database needed."""

import pytest

from repolens_mcp.tools import SCHEMAS, Page


@pytest.mark.parametrize(
    "data",
    [{"owner_id": "foreign"}, {"limit": True}, {"limit": 26}, {"offset": 10001}, {"limit": "5"}],
)
def test_page_rejects_untrusted_fields(data):

    with pytest.raises(ValueError):
        Page.model_validate(data)


def test_exact_tool_set():

    assert set(SCHEMAS) == {
        "list_repositories",
        "list_revisions",
        "search_source",
        "read_source_excerpt",
    }

    assert all(m.model_json_schema()["additionalProperties"] is False for m in SCHEMAS.values())


@pytest.mark.parametrize(
    "data",
    [
        {"query": "x" * 257},
        {"query": "x", "top_k": 6},
        {"query": "x", "owner_id": "foreign"},
    ],
)
def test_search_bounds(data):

    from uuid import uuid4

    from repolens_mcp.tools import Search

    with pytest.raises(ValueError):
        Search.model_validate({"repository_id": str(uuid4()), "revision_id": str(uuid4()), **data})


@pytest.mark.parametrize("path", [".env", "dir/.env.local", "private.pem", "id.key"])
def test_secret_paths_denied(path):

    from fastapi import HTTPException

    from repolens_mcp.tools import safe

    with pytest.raises(HTTPException) as exc:
        safe(path, "synthetic text")

    assert exc.value.status_code == 404


def test_excerpt_byte_and_line_limits():

    from fastapi import HTTPException

    from repolens_mcp.tools import clip

    assert clip("a\nb\nc", 2, 3, 1, 3, 10) == "b\nc"

    with pytest.raises(HTTPException) as exc:
        clip("a\nb", 1, 3, 1, 2, 10)

    assert exc.value.status_code == 422

    with pytest.raises(HTTPException) as exc:
        clip("é", 1, 1, 1, 1, 1)

    assert exc.value.status_code == 413
