"""J1 pure contracts and deterministic genuine-shape GitHub HTTP."""

from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from api.schemas.career import InputCreate, content_hash, job_text
from ingestion.github_profile import Discovery, profile_account


@pytest.mark.parametrize("value", ["https://github.com/Some-User", "https://github.com/some-user/"])
def test_profiles(value):
    assert profile_account(value) == "some-user"


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/a",
        "https://evil.test/a",
        "https://github.com/a/b",
        "https://u:p@github.com/a",
        "https://github.com:443/a",
        "https://github.com/a?",
        "https://github.com/a#",
        "https://127.0.0.1/a",
        "file:///a",
        "https://github.com/a%2fb",
        "https://github.com/a//",
        "https://github.com/-a",
        "https://github.com/a--b",
        "https://github.com/a_b",
        "https://github.com/a\nb",
        " https://github.com/example",
        "https://github.com/" + "a" * 40,
    ],
)
def test_bad_profiles(value):
    with pytest.raises(ValueError):
        profile_account(value)


def test_job_snapshot_contract():
    text = "We need a Python developer.\nBuild reliable APIs."
    assert job_text("  " + text.replace("\n", "\r\n") + "  ") == text
    assert content_hash(job_text(text)) == content_hash(job_text(text.replace("\n", "\r\n")))
    for value in [
        "",
        " ",
        "1" * 50,
        "a" * 20001,
        "Python developer\x00 job role",
        "Python developer \ud800 job role",
    ]:
        with pytest.raises(ValueError):
            job_text(value)
    with pytest.raises(ValidationError):
        InputCreate(
            profile_id=uuid4(), github_profile_url="https://github.com/example", job_description=" "
        )


def repository(i=1, **extra):
    return {
        "id": i,
        "owner": {"login": "example"},
        "full_name": f"example/project{i}",
        "html_url": f"https://github.com/example/project{i}",
        "private": False,
        "fork": False,
        "archived": False,
        "default_branch": "main",
        **extra,
    }


@pytest.mark.asyncio
async def test_discovery_policy_and_fixed_pagination():
    calls = []

    def respond(request):
        calls.append(str(request.url))
        assert (
            request.url.host == "api.github.com"
            and request.headers["Authorization"] == "Bearer fixture-only"
        )
        if request.url.path == "/users/example":
            return httpx.Response(200, json={"login": "Example", "type": "User"})
        page = int(request.url.params["page"])
        return httpx.Response(
            200,
            json=[repository(page, fork=page == 1, archived=page == 2)],
            headers={"Link": '<https://evil.test/steal>; rel="next"'},
        )

    result = await Discovery(
        "fixture-only", transport=httpx.MockTransport(respond), pages=2
    ).discover("example")
    assert len(calls) == 3 and result["status"] == "bounded"
    assert [x["excluded_reason"] for x in result["items"]] == ["fork", "archived"]
    assert not any(x["eligible"] for x in result["items"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected",
    [
        ("empty", "complete"),
        ("rate", "unavailable"),
        ("bad", "unavailable"),
        ("timeout", "unavailable"),
        ("private", "unavailable"),
        ("duplicate", "partial"),
        ("oversized", "unavailable"),
        ("redirect", "unavailable"),
    ],
)
async def test_discovery_failures(mode, expected):
    def respond(request):
        if request.url.path == "/users/example":
            return httpx.Response(200, json={"login": "example", "type": "User"})
        if mode == "timeout":
            raise httpx.ReadTimeout("sensitive transport diagnostic", request=request)
        if mode == "rate":
            return httpx.Response(429)
        if mode == "redirect":
            return httpx.Response(302, headers={"Location": "https://evil.test"})
        if mode == "bad":
            return httpx.Response(200, json={"secret": "never echo"})
        if mode == "oversized":
            return httpx.Response(200, content=b"a" * 524289)
        rows = [] if mode == "empty" else [repository(private=mode == "private")]
        if mode == "duplicate":
            rows *= 2
        return httpx.Response(200, json=rows)

    result = await Discovery(transport=httpx.MockTransport(respond)).discover("example")
    assert result["status"] == expected
    assert "sensitive" not in str(result) and "secret" not in str(result)
    if mode == "private":
        assert not result["items"]


@pytest.mark.asyncio
async def test_discovery_manifest_bytes_are_bounded_before_persistence():
    def respond(request):
        if request.url.path == "/users/example":
            return httpx.Response(200, json={"login": "example", "type": "User"})
        page = int(request.url.params["page"])
        rows = [
            repository(
                (page - 1) * 30 + i,
                description="界" * 512,
                language="界" * 80,
                default_branch="界" * 255,
            )
            for i in range(1, 31)
        ]
        return httpx.Response(200, json=rows, headers={"Link": '<https://evil.test/>; rel="next"'})

    result = await Discovery(transport=httpx.MockTransport(respond), pages=4).discover("example")
    assert result["status"] == "partial" and result["reason"] == "manifest_limit"
    import json

    assert len(json.dumps(result).encode()) < 262144
