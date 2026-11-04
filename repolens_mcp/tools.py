"""Four bounded adapters; all source bytes come from canonical database snapshots."""

from uuid import UUID

from fastapi import HTTPException
from pydantic import Field, model_validator

from core.contracts import Contract
from core.services import keyword_retrieval, repository_catalog
from rag.generator.grounding import unsafe


class Page(Contract):
    limit: int = Field(default=20, strict=True, ge=1, le=25)
    offset: int = Field(default=0, strict=True, ge=0, le=10000)


class Revisions(Page):
    repository_id: UUID


class Search(Contract):
    repository_id: UUID
    revision_id: UUID
    query: str = Field(strict=True, min_length=1, max_length=256)
    top_k: int = Field(default=5, strict=True, ge=1, le=5)


class Excerpt(Contract):
    repository_id: UUID
    revision_id: UUID
    chunk_id: UUID
    line_start: int | None = Field(default=None, strict=True, ge=1)
    line_end: int | None = Field(default=None, strict=True, ge=1)

    @model_validator(mode="after")
    def range_pair(self):
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("Both bounds required")
        if (
            self.line_start is not None
            and not self.line_start <= self.line_end < self.line_start + 40
        ):
            raise ValueError("Invalid range")
        return self


SCHEMAS = {
    "list_repositories": Page,
    "list_revisions": Revisions,
    "search_source": Search,
    "read_source_excerpt": Excerpt,
}


def safe(path, text):
    if (
        any(p.startswith(".env") for p in path.lower().split("/"))
        or path.lower().endswith((".pem", ".key"))
        or unsafe(text)
    ):
        raise HTTPException(404, "Source unavailable")


def clip(text, start, end, first, last, maximum):
    if not first <= start <= end <= last:
        raise HTTPException(422, "Invalid source range")
    excerpt = "\n".join(text.splitlines()[start - first : end - first + 1])
    if not excerpt or len(excerpt.encode()) > maximum:
        raise HTTPException(413, "Excerpt exceeds limit")
    return excerpt


async def execute(db, uid, name, args):
    if name == "list_repositories":
        return await repository_catalog.repositories(db, uid, args.limit, args.offset)
    if name == "list_revisions":
        return await repository_catalog.revisions(
            db, uid, args.repository_id, args.limit, args.offset
        )
    repo, rev = str(args.repository_id), str(args.revision_id)
    if name == "search_source":
        result = await keyword_retrieval.search(
            db, uid, repo, rev, {"query": args.query, "mode": "keyword", "top_k": args.top_k}
        )
        items = []
        for match in result.matches:
            safe(match.path, match.excerpt)
            last = min(match.line_end, match.line_start + 9)
            data = match.model_dump(mode="json")
            data.update(
                line_end=last,
                excerpt=clip(
                    match.excerpt, match.line_start, last, match.line_start, match.line_end, 2048
                ),
            )
            items.append(data)
        await keyword_retrieval._scope(db, uid, repo, rev)
        return {
            "repository_id": repo,
            "revision_id": rev,
            "commit_sha": result.commit_sha,
            "matches": items,
            "untrusted_source": True,
        }
    snapshot = await keyword_retrieval._snapshot(
        db, uid, repo, rev, keyword_retrieval.SearchLimits.configured()
    )
    found = next(((c, p) for c, p in snapshot.chunks if str(c.id) == str(args.chunk_id)), None)
    if found is None:
        raise HTTPException(404, "Chunk not found")
    chunk, path = found
    safe(path, chunk.text)
    start = args.line_start or chunk.line_start
    end = args.line_end or min(chunk.line_end, start + 39)
    text = clip(chunk.text, start, end, chunk.line_start, chunk.line_end, 8192)
    return {
        "repository_id": repo,
        "revision_id": rev,
        "commit_sha": snapshot.sha,
        "snapshot_digest": snapshot.digest,
        "source_id": str(chunk.source_id),
        "chunk_id": str(chunk.id),
        "path": path,
        "line_start": start,
        "line_end": end,
        "excerpt": text,
        "untrusted_source": True,
    }
