"""Bounded public-account metadata discovery; no source downloads or model calls."""

import asyncio
import json
import re
from urllib.parse import urlsplit

import httpx

from ingestion.github import OWNER_RE, GitHubError, repository_name


def profile_account(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 200
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Use an HTTPS github.com profile URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
        or "?" in value
        or "#" in value
        or not re.fullmatch(r"/[A-Za-z0-9-]+/?", parsed.path)
    ):
        raise ValueError("Use an HTTPS github.com profile URL")
    account = parsed.path.strip("/")
    if not OWNER_RE.fullmatch(account) or "--" in account:
        raise ValueError("Invalid GitHub account name")
    return account.lower()


class Discovery:
    def __init__(self, token="", *, transport=None, pages=2):
        if token and (not token.isascii() or any(ord(c) <= 32 or ord(c) >= 127 for c in token)):
            raise GitHubError("credentials", "Invalid server GitHub configuration")
        self.pages = min(max(pages, 1), 4)
        self.requests = 0
        self.bytes = 0
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=10,
            follow_redirects=False,
            trust_env=False,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "RepoLens-AI",
                "Accept-Encoding": "identity",
                **({"Authorization": "Bearer " + token} if token else {}),
            },
        )

    async def get(self, path):
        self.requests += 1
        if self.requests > self.pages + 1:
            raise GitHubError("request_limit", "Discovery limit reached")
        try:
            async with self.http.stream("GET", "https://api.github.com" + path) as response:
                if response.status_code in {403, 429}:
                    raise GitHubError("rate_limited", "Public discovery rate limited or denied")
                if response.status_code != 200:
                    raise GitHubError("unavailable", "Public account discovery unavailable")
                data = bytearray()
                async for part in response.aiter_bytes():
                    data.extend(part)
                    self.bytes += len(part)
                    if len(data) > 524288 or self.bytes > 2097152:
                        raise GitHubError("response_limit", "Discovery response limit reached")
                result = json.loads(data)
                return result, 'rel="next"' in response.headers.get("link", "")
        except (ValueError, UnicodeError):
            raise GitHubError("invalid_response", "Invalid GitHub discovery response") from None
        except httpx.RequestError:
            raise GitHubError("network", "Public account discovery unavailable") from None

    async def discover(self, account):
        # Revalidate even direct internal calls before constructing destinations.
        account = profile_account("https://github.com/" + account)
        items, seen = [], set()
        manifest_bytes = 0
        state, error = "complete", None
        try:
            async with asyncio.timeout(35):
                user, _ = await self.get("/users/" + account)
                if (
                    not isinstance(user, dict)
                    or user.get("type") != "User"
                    or str(user.get("login", "")).lower() != account
                ):
                    raise GitHubError("invalid_response", "Unsupported public account response")
                for page in range(1, self.pages + 1):
                    rows, more = await self.get(
                        f"/users/{account}/repos?type=owner&sort=full_name&direction=asc&per_page=30&page={page}"
                    )
                    if not isinstance(rows, list) or len(rows) > 30:
                        raise GitHubError(
                            "invalid_response", "Invalid repository discovery response"
                        )
                    for row in rows:
                        if not isinstance(row, dict) or not isinstance(row.get("full_name"), str):
                            raise GitHubError("invalid_response", "Invalid repository entry")
                        name = repository_name(row.get("full_name", ""))
                        if (
                            name.split("/")[0] != account
                            or not isinstance(row.get("owner"), dict)
                            or str(row["owner"].get("login", "")).lower() != account
                            or type(row.get("id")) is not int
                            or row["id"] <= 0
                            or any(
                                type(row.get(k)) is not bool
                                for k in ("private", "fork", "archived")
                            )
                            or row.get("html_url", "").lower() != "https://github.com/" + name
                        ):
                            raise GitHubError("invalid_response", "Invalid repository identity")
                        if name in seen:
                            raise GitHubError(
                                "duplicate_response", "Discovery changed during pagination"
                            )
                        seen.add(name)
                        reason = (
                            "private"
                            if row["private"] or row.get("visibility", "public") != "public"
                            else "fork"
                            if row["fork"]
                            else "archived"
                            if row["archived"]
                            else "no_default_branch"
                            if not row.get("default_branch")
                            else None
                        )
                        out = {
                            "full_name": name,
                            "github_id": row["id"],
                            "eligible": reason is None,
                            "excluded_reason": reason,
                            "public": not row["private"],
                            "fork": row["fork"],
                            "archived": row["archived"],
                            "url": "https://github.com/" + name,
                        }
                        for key, cap in (
                            ("description", 512),
                            ("language", 80),
                            ("default_branch", 255),
                            ("updated_at", 40),
                        ):
                            value = row.get(key)
                            if value is not None and (
                                not isinstance(value, str) or len(value) > cap
                            ):
                                raise GitHubError("invalid_response", "Invalid repository metadata")
                            out[key] = value
                        # Never retain metadata about an unexpectedly returned private repository.
                        if row["private"]:
                            raise GitHubError(
                                "invalid_response", "Unexpected nonpublic discovery data"
                            )
                        manifest_bytes += len(json.dumps(out).encode("utf-8"))
                        if manifest_bytes > 240000:
                            raise GitHubError("manifest_limit", "Discovery manifest limit reached")
                        items.append(out)
                    if not more:
                        break
                    if page == self.pages:
                        state, error = "bounded", "page_limit"
        except (GitHubError, TimeoutError, ValueError) as exc:
            state = "partial" if items else "unavailable"
            error = (
                exc.code
                if isinstance(exc, GitHubError)
                else "invalid_response"
                if isinstance(exc, ValueError)
                else "timeout"
            )
        finally:
            await self.http.aclose()
        return {
            "version": 1,
            "status": state,
            "reason": error,
            "items": items,
            "requests": self.requests,
            "private_coverage": "not_accessed_or_enumerated",
        }
