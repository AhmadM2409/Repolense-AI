import asyncio
import base64
import httpx
import pytest
from ingestion.github import GitHubClient, GitHubError, Limits, repository_name
from tests.github_responses import GitHubResponses

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "value",
    [
        "example/project",
        "https://github.com/example/project",
        "https://github.com/Example/Project.git/",
    ],
)
async def test_repository_input(value):
    assert repository_name(value) == "example/project"


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/a/b",
        "https://evil.test/a/b",
        "https://github.com@127.0.0.1/a/b",
        "https://github.com:443/a/b",
        "https://github.com/a/b?token=x",
        "https://github.com/a/b#x",
        "https://github.com/a/b/tree/main",
        "a/..",
        "a/%2e%2e",
        "../x",
        "https://github.com/a/b%2fc",
        "git@github.com:a/b",
        "a/b/c",
    ],
)
async def test_unsafe_repository_input(value):
    with pytest.raises(ValueError):
        repository_name(value)


async def fetch(fixture, limits=None):
    async with GitHubClient(
        "secret-fixture-token", transport=httpx.MockTransport(fixture), limits=limits
    ) as client:
        async with asyncio.timeout(client.limits.seconds):
            revision = await client.resolve("example/project")
            return await client.download(revision)


async def test_pin_hashes_lines_and_token_routing():
    fixture = GitHubResponses(branch="feature/dev")
    snapshot = await fetch(fixture)
    assert snapshot.revision.sha == fixture.sha
    assert {f.path for f in snapshot.files} == {"README.md", "src/main.py", "package.json"}
    assert snapshot.files[0].text == "# Real fixture\nContent\n"
    assert all(
        r.headers["authorization"] == "Bearer secret-fixture-token" for r in fixture.requests
    )
    assert all(r.url.host == "api.github.com" for r in fixture.requests)
    assert any(b"feature%2Fdev" in r.url.raw_path for r in fixture.requests)
    assert all(
        "/contents/" not in r.url.path and "raw.githubusercontent" not in str(r.url)
        for r in fixture.requests
    )
    assert all(f.blob_sha in fixture.blobs for f in snapshot.files)


@pytest.mark.parametrize(
    "status,code",
    [
        (404, "not_public"),
        (401, "credentials"),
        (429, "rate_limited"),
        (403, "not_public"),
        (302, "github_error"),
    ],
)
async def test_http_failures_are_bounded(status, code):
    fixture = GitHubResponses()
    fixture.overrides[""] = lambda r: httpx.Response(
        status, headers={"Location": "http://localhost/secret"}, json={}
    )
    with pytest.raises(GitHubError) as error:
        await fetch(fixture)
    assert error.value.code == code and len(fixture.requests) == 1
    assert "secret-fixture-token" not in str(error.value)


async def test_private_repository_and_secondary_rate_limit():
    fixture = GitHubResponses()
    fixture.overrides[""] = {"full_name": fixture.name, "private": True, "default_branch": "main"}
    with pytest.raises(GitHubError, match="public"):
        await fetch(fixture)
    fixture.overrides[""] = lambda r: httpx.Response(403, headers={"Retry-After": "3600"}, json={})
    with pytest.raises(GitHubError) as error:
        await fetch(fixture)
    assert error.value.code == "rate_limited"


async def test_retry_transient_response_and_network_error():
    fixture = GitHubResponses()
    attempts = 0

    def temporary(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("external fixture", request=request)
        if attempts == 2:
            return httpx.Response(503, json={})
        fixture.overrides.pop("")
        return fixture(request)

    fixture.overrides[""] = temporary
    assert (await fetch(fixture)).revision.sha == fixture.sha
    assert attempts == 3


@pytest.mark.parametrize(
    "change,code",
    [
        ("tree_hash", "integrity"),
        ("blob_hash", "integrity"),
        ("truncated", "truncated_tree"),
        ("sha", "invalid_response"),
        ("pagination", "invalid_response"),
        ("invalid_json", "invalid_response"),
    ],
)
async def test_invalid_responses_cannot_claim_provenance(change, code):
    fixture = GitHubResponses()
    if change == "tree_hash":
        fixture.trees[fixture.tree_sha]["tree"][0]["sha"] = "a" * 40
    if change == "blob_hash":
        blob = next(iter(fixture.blobs.values()))
        blob["content"] = base64.b64encode(b"x" * blob["size"]).decode()
    if change == "truncated":
        fixture.trees[fixture.tree_sha]["truncated"] = True
    if change == "sha":
        fixture.overrides["git/ref/heads/main"] = {"object": {"type": "commit", "sha": "main"}}
    if change == "pagination":
        fixture.overrides[""] = lambda r: httpx.Response(
            200, headers={"Link": '<http://127.0.0.1/>; rel="next"'}, json={}
        )
    if change == "invalid_json":
        fixture.overrides[""] = lambda r: httpx.Response(200, content=b"not JSON")
    with pytest.raises(GitHubError) as error:
        await fetch(fixture)
    assert error.value.code == code


async def test_skips_symlinks_binary_secrets_dependencies_and_oversize():
    fixture = GitHubResponses(
        {
            "README.md": b"text",
            ".env": b"secret",
            "link.py": (b"../../secret", "120000"),
            "binary.py": b"\0bad",
            "huge.py": b"x" * (128 * 1024 + 1),
            "node_modules/lib.js": b"excluded",
            "image.png": b"unsupported",
            ".github/workflows/ci.yml": b"name: fixture",
        }
    )
    result = await fetch(fixture)
    assert {f.path for f in result.files} == {"README.md", ".github/workflows/ci.yml"}
    assert {item["reason"] for item in result.skipped} == {
        "excluded",
        "symlink_or_submodule",
        "binary_or_encoding",
        "oversized",
        "unsupported",
    }


@pytest.mark.parametrize(
    "limits,code",
    [
        (Limits(files=1), "file_limit"),
        (Limits(total_bytes=1), "byte_limit"),
        (Limits(entries=1), "entry_limit"),
        (Limits(requests=1), "request_limit"),
        (Limits(response_bytes=10), "response_limit"),
        (Limits(wire_bytes=10), "response_limit"),
    ],
)
async def test_limits(limits, code):
    with pytest.raises(GitHubError) as error:
        await fetch(GitHubResponses(), limits)
    assert error.value.code == code


async def test_depth_limit_is_explicit():
    fixture = GitHubResponses({"README.md": b"text", "docs/deep/example.md": b"deep"})
    snapshot = await fetch(fixture, Limits(depth=0))
    assert [f.path for f in snapshot.files] == ["README.md"]
    assert {"path": "docs", "reason": "depth_limit"} in snapshot.skipped


async def test_unsafe_path_and_no_supported_files():
    with pytest.raises(GitHubError, match="unsafe"):
        await fetch(GitHubResponses({"../file.py": b"x"}))
    with pytest.raises(GitHubError) as error:
        await fetch(GitHubResponses({"image.png": b"x"}))
    assert error.value.code == "no_files"


async def test_request_deadline():
    async def slow(request):
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={})

    async with GitHubClient(
        transport=httpx.MockTransport(slow), limits=Limits(seconds=0.01)
    ) as client:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(client.limits.seconds):
                await client.resolve("example/project")


async def test_extensionless_readme_and_license_are_supported():
    snapshot = await fetch(
        GitHubResponses({"README": b"Hello World!\n", "LICENSE": b"Fixture license text"})
    )
    assert {file.path for file in snapshot.files} == {"README", "LICENSE"}


async def test_visibility_is_rechecked_before_accepting_snapshot():
    fixture = GitHubResponses()
    calls = 0

    def metadata(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"full_name": fixture.name, "private": calls > 1, "default_branch": "main"}
        )

    fixture.overrides[""] = metadata
    with pytest.raises(GitHubError, match="no longer public"):
        await fetch(fixture)


async def test_normalized_text_hash_retains_verified_raw_blob_identity():
    import hashlib

    fixture = GitHubResponses({"README.md": b"\xef\xbb\xbffirst\r\nsecond\r"})
    result = await fetch(fixture)
    file = result.files[0]
    assert file.text == "first\nsecond\n"
    assert file.content_hash == hashlib.sha256(file.text.encode()).hexdigest()
    assert file.blob_sha in fixture.blobs


async def test_lfs_pointer_is_not_presented_as_file_content():
    fixture = GitHubResponses(
        {
            "README.md": b"text",
            "large.py": b"version https://git-lfs.github.com/spec/v1\noid sha256:fixture\n",
        }
    )
    result = await fetch(fixture)
    assert [file.path for file in result.files] == ["README.md"]
    assert {"path": "large.py", "reason": "lfs_pointer"} in result.skipped


@pytest.mark.parametrize("token", ["bad\nkey", "bad key", "nonascii-\u00e9"])
async def test_invalid_server_credentials_fail_before_transport(token):
    with pytest.raises(GitHubError) as error:
        GitHubClient(token)
    assert error.value.code == "credentials"
