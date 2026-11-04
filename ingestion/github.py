"""Bounded GitHub-only transport. No cloning, raw URLs, archives or token forwarding."""

import asyncio
import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote, urlsplit

import httpx

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
OWNER_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,37}[a-zA-Z0-9])?$")
REPO_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,100}$")
EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".next",
    "coverage",
    ".idea",
    ".vscode",
}
EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".md",
    ".rst",
    ".go",
    ".rs",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".rb",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".xml",
    ".lock",
    ".gradle",
    ".kt",
    ".swift",
    ".php",
    ".vue",
    ".svelte",
}
NAMES = {
    "readme",
    "license",
    "licence",
    "notice",
    "changelog",
    "contributing",
    "security",
    "authors",
    "copying",
    "code_of_conduct",
    "dockerfile",
    "makefile",
    "gemfile",
    "procfile",
    ".gitignore",
    ".dockerignore",
}


def repository_name(value: str) -> str:
    value = value.strip()
    if "://" in value:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.netloc.lower() != "github.com"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use owner/repo or an HTTPS github.com repository URL")
        value = parsed.path.strip("/")
        if value.endswith(".git"):
            value = value[:-4]
    parts = value.split("/")
    if (
        len(parts) != 2
        or not OWNER_RE.fullmatch(parts[0])
        or not REPO_RE.fullmatch(parts[1])
        or parts[1] in {".", ".."}
    ):
        raise ValueError("Use owner/repo or an HTTPS github.com repository URL")
    return value.lower()


class GitHubError(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


@dataclass(frozen=True)
class Limits:
    files: int = 100
    file_bytes: int = 128 * 1024
    total_bytes: int = 4 * 1024 * 1024
    entries: int = 3000
    depth: int = 8
    requests: int = 256
    response_bytes: int = 2 * 1024 * 1024
    wire_bytes: int = 16 * 1024 * 1024
    seconds: float = 60
    request_seconds: float = 10


@dataclass(frozen=True)
class Revision:
    full_name: str
    branch: str
    sha: str
    tree_sha: str


@dataclass(frozen=True)
class Download:
    path: str
    text: str
    blob_sha: str
    raw_bytes: int
    content_hash: str


@dataclass
class Snapshot:
    revision: Revision
    files: list[Download]
    skipped: list[dict]
    entries: int


def git_hash(kind: str, data: bytes) -> str:
    return hashlib.sha1(kind.encode() + b" " + str(len(data)).encode() + b"\0" + data).hexdigest()


def valid_sha(value):
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise GitHubError("invalid_response", "GitHub returned an invalid object identity")
    return value


def safe_name(value):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or len(value.encode()) > 255
        or any(c in value for c in "/\\\0:")
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise GitHubError("unsafe_path", "Repository contains an unsafe path")
    return value


class GitHubClient:
    def __init__(self, token: str = "", *, transport=None, limits: Limits | None = None):
        if token and (
            len(token) > 4096
            or not token.isascii()
            or any(ord(c) <= 32 or ord(c) >= 127 for c in token)
        ):
            raise GitHubError("credentials", "Server GitHub credentials are invalid")
        self.checkpoint = None
        self.limits = limits or Limits()
        self.requests = self.wire_bytes = 0
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept-Encoding": "identity",
            "User-Agent": "RepoLens-AI",
        }
        if token:
            headers["Authorization"] = "Bearer " + token
        self.http = httpx.AsyncClient(
            headers=headers,
            timeout=self.limits.request_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def _json(self, name, suffix):
        # Both name and suffix are constructed/validated here; response URLs/Link headers are never followed.
        name = repository_name(name)
        url = "https://api.github.com/repos/" + name + ("/" + suffix if suffix else "")
        for attempt in range(3):
            if self.checkpoint:
                await self.checkpoint("fetching_sources" if suffix.startswith("git/blobs/") else
                                      "fetching_tree" if suffix.startswith("git/trees/") else None)
            self.requests += 1
            if self.requests > self.limits.requests:
                raise GitHubError("request_limit", "Repository exceeds the request limit")
            try:
                async with self.http.stream("GET", url) as response:
                    status = response.status_code
                    if status in {403, 429} and (
                        status == 429
                        or response.headers.get("x-ratelimit-remaining") == "0"
                        or "retry-after" in response.headers
                    ):
                        raise GitHubError("rate_limited", "GitHub rate limit reached; retry later")
                    if status in {404, 403}:
                        raise GitHubError(
                            "not_public", "Public repository or revision not found or access denied"
                        )
                    if status == 401:
                        raise GitHubError("credentials", "Server GitHub credentials were rejected")
                    if status in {502, 503, 504} and attempt < 2:
                        await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    if 500 <= status <= 599:
                        raise GitHubError("upstream_unavailable", "GitHub temporarily unavailable")
                    if status != 200:
                        raise GitHubError("github_error", "GitHub request failed")
                    if response.headers.get("link"):
                        raise GitHubError(
                            "invalid_response", "Unexpected paginated Git object response"
                        )
                    data = bytearray()
                    async for part in response.aiter_bytes():
                        data.extend(part)
                        self.wire_bytes += len(part)
                        if (
                            len(data) > self.limits.response_bytes
                            or self.wire_bytes > self.limits.wire_bytes
                        ):
                            raise GitHubError(
                                "response_limit", "GitHub response exceeds ingestion limits"
                            )
                    try:
                        result = json.loads(data)
                    except (ValueError, UnicodeError):
                        raise GitHubError(
                            "invalid_response", "GitHub returned invalid JSON"
                        ) from None
                    if not isinstance(result, dict):
                        raise GitHubError("invalid_response", "GitHub returned an invalid object")
                    return result
            except httpx.RequestError:
                if attempt == 2:
                    raise GitHubError("network", "GitHub network request failed") from None
                await asyncio.sleep(0.25 * (2**attempt))
        raise GitHubError("network", "GitHub request failed")

    async def resolve(self, name: str) -> Revision:
        name = repository_name(name)
        metadata = await self._json(name, "")
        if metadata.get("private") is not False or metadata.get("visibility", "public") != "public":
            raise GitHubError("not_public", "Only public repositories are supported")
        if metadata.get("full_name", "").lower() != name:
            raise GitHubError(
                "invalid_response", "GitHub repository identity changed; submit its canonical name"
            )
        branch = metadata.get("default_branch")
        if (
            not isinstance(branch, str)
            or not branch
            or len(branch) > 255
            or any(ord(c) < 32 for c in branch)
        ):
            raise GitHubError("invalid_response", "Repository has no valid default branch")
        ref = await self._json(name, "git/ref/heads/" + quote(branch, safe=""))
        if ref.get("ref") != "refs/heads/" + branch:
            raise GitHubError("invalid_response", "GitHub returned a different branch reference")
        obj = ref.get("object", {})
        if not isinstance(obj, dict) or obj.get("type") != "commit":
            raise GitHubError("invalid_response", "Default branch did not resolve to a commit")
        sha = valid_sha(obj.get("sha"))
        commit = await self._json(name, "git/commits/" + sha)
        if commit.get("sha") != sha or not isinstance(commit.get("tree"), dict):
            raise GitHubError(
                "invalid_response", "GitHub commit identity does not match the pinned revision"
            )
        return Revision(name, branch, sha, valid_sha(commit["tree"].get("sha")))

    async def download(self, revision: Revision) -> Snapshot:
        queue = [("", revision.tree_sha, 0)]
        files = []
        skipped = []
        entries = total = 0
        while queue:
            prefix, sha, depth = queue.pop(0)
            tree = await self._json(revision.full_name, "git/trees/" + valid_sha(sha))
            items = tree.get("tree")
            if (
                tree.get("sha") != sha
                or tree.get("truncated") is not False
                or not isinstance(items, list)
            ):
                raise GitHubError("truncated_tree", "GitHub tree is incomplete or invalid")
            entries += len(items)
            if entries > self.limits.entries:
                raise GitHubError("entry_limit", "Repository exceeds the tree entry limit")
            names = set()
            encoded = []
            for item in items:
                if not isinstance(item, dict):
                    raise GitHubError("invalid_response", "Invalid tree entry")
                name = safe_name(item.get("path"))
                if name in names:
                    raise GitHubError("invalid_response", "Duplicate tree path")
                names.add(name)
                oid = valid_sha(item.get("sha"))
                mode = item.get("mode")
                kind = item.get("type")
                if (mode, kind) not in {
                    ("040000", "tree"),
                    ("100644", "blob"),
                    ("100755", "blob"),
                    ("120000", "blob"),
                    ("160000", "commit"),
                }:
                    raise GitHubError("invalid_response", "Invalid Git tree mode")
                encoded.append(
                    (
                        name.encode() + (b"/" if kind == "tree" else b""),
                        mode.lstrip("0").encode()
                        + b" "
                        + name.encode()
                        + b"\0"
                        + bytes.fromhex(oid),
                    )
                )
            if git_hash("tree", b"".join(value for _, value in sorted(encoded))) != sha:
                raise GitHubError("integrity", "GitHub tree hash verification failed")
            for item in sorted(items, key=lambda x: x["path"]):
                name = item["path"]
                path = prefix + name
                if len(path.encode()) > 1024:
                    raise GitHubError("unsafe_path", "Repository path exceeds the path limit")
                reason = None
                if name.casefold() in EXCLUDED or name.casefold().startswith(".env"):
                    reason = "excluded"
                elif item["mode"] in {"120000", "160000"}:
                    reason = "symlink_or_submodule"
                elif item["type"] == "tree":
                    if depth >= self.limits.depth:
                        reason = "depth_limit"
                    else:
                        queue.append((path + "/", item["sha"], depth + 1))
                        continue
                elif (
                    PurePosixPath(name).suffix.lower() not in EXTENSIONS
                    and name.lower() not in NAMES
                ):
                    reason = "unsupported"
                size = item.get("size")
                if not reason and (type(size) is not int or size < 0):
                    raise GitHubError("invalid_response", "Invalid GitHub file size")
                if not reason and size > self.limits.file_bytes:
                    reason = "oversized"
                if reason:
                    skipped.append({"path": path, "reason": reason})
                    continue
                if len(files) >= self.limits.files:
                    raise GitHubError("file_limit", "Repository exceeds the supported file count")
                blob = await self._json(revision.full_name, "git/blobs/" + item["sha"])
                if (
                    blob.get("sha") != item["sha"]
                    or blob.get("encoding") != "base64"
                    or type(blob.get("size")) is not int
                    or blob.get("size") != size
                    or not isinstance(blob.get("content"), str)
                ):
                    raise GitHubError(
                        "invalid_response", "GitHub blob identity or encoding is invalid"
                    )
                try:
                    raw = base64.b64decode(blob["content"].replace("\n", ""), validate=True)
                except ValueError:
                    raise GitHubError("invalid_response", "Invalid GitHub blob encoding") from None
                if len(raw) != size or git_hash("blob", raw) != item["sha"]:
                    raise GitHubError("integrity", "GitHub file content hash verification failed")
                total += len(raw)
                if total > self.limits.total_bytes:
                    raise GitHubError("byte_limit", "Repository exceeds the total content limit")
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeError:
                    skipped.append({"path": path, "reason": "binary_or_encoding"})
                    continue
                if any(
                    (ord(c) < 32 and c not in "\t\r\n")
                    or 127 <= ord(c) <= 159
                    or c in "\u2028\u2029"
                    for c in text
                ):
                    skipped.append({"path": path, "reason": "binary_or_encoding"})
                    continue
                if text.startswith("version https://git-lfs.github.com/spec/v1"):
                    skipped.append({"path": path, "reason": "lfs_pointer"})
                    continue
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                files.append(
                    Download(
                        path, text, item["sha"], len(raw), hashlib.sha256(text.encode()).hexdigest()
                    )
                )
        if not files:
            raise GitHubError("no_files", "No supported text files were found")
        public = await self._json(revision.full_name, "")
        if (
            public.get("private") is not False
            or public.get("visibility", "public") != "public"
            or public.get("full_name", "").lower() != revision.full_name
        ):
            raise GitHubError("not_public", "Repository is no longer public at completion")
        return Snapshot(revision, files, skipped, entries)
