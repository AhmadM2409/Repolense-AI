"""Canonical internal/API contracts. Line numbers are one-based and inclusive."""

from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class JobStatus(str, Enum):
    cancelled = "cancelled"
    pending = "pending"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class RepositoryContract(Contract):
    id: UUID
    profile_id: UUID
    full_name: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", max_length=255)
    provider: str = Field(default="github", pattern="^github$")

    @field_validator("full_name")
    @classmethod
    def normalize_name(cls, value):
        return value.lower()


class RevisionContract(Contract):
    id: UUID
    profile_id: UUID
    repository_id: UUID
    commit_sha: SHA


class SourceDocument(Contract):
    id: UUID
    profile_id: UUID
    source_type: str = Field(min_length=1, max_length=50)
    revision_id: UUID | None = None
    path: str | None = Field(default=None, max_length=1024)
    text: str
    content_hash: Digest
    source_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def safe_path(cls, value):
        if value is not None and (
            not value
            or "\\" in value
            or ":" in value
            or value.startswith("/")
            or ".." in PurePosixPath(value).parts
        ):
            raise ValueError("path must be repository-relative without traversal")
        return value

    @model_validator(mode="after")
    def source_integrity(self):
        import hashlib

        if (self.revision_id is None) != (self.path is None):
            raise ValueError("revision_id and path must be supplied together")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.content_hash:
            raise ValueError("content_hash must match UTF-8 text")
        return self


class TextChunk(Contract):
    id: UUID
    profile_id: UUID
    source_id: UUID
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_lines(self):
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class EvidenceSpan(Contract):
    source_id: UUID
    repository_id: UUID
    commit_sha: SHA
    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: str = Field(min_length=1)

    _safe_path = field_validator("path")(SourceDocument.safe_path.__func__)

    @model_validator(mode="after")
    def ordered_lines(self):
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class FindingContract(Contract):
    id: UUID
    profile_id: UUID
    review_id: UUID
    title: str = Field(min_length=1, max_length=255)
    detail: str = Field(min_length=1)
    evidence: list[EvidenceSpan] = Field(min_length=1)
    rule_id: str = Field(default="legacy", max_length=64)
    severity: Literal["info", "low", "medium", "high", "critical"] = "info"
    category: Literal["security", "maintainability", "configuration"] = "maintainability"
    certainty: Literal["observation", "potential"] = "observation"
    fingerprint: Digest | None = None


class AnalysisJob(Contract):
    id: UUID
    profile_id: UUID
    status: JobStatus
    revision_id: UUID | None = None
    error_message: str | None = None


class ReportContract(Contract):
    review_id: UUID
    profile_id: UUID
    findings: list[FindingContract]
    # Legacy sections/scores remain on Review; absence of findings means no verified evidence.


def normalize_github_metadata(data: dict) -> dict:
    """Accept raw REST fields and inherited GitHubTool aliases; do not infer a SHA."""
    result = dict(data)
    for canonical, alias in (
        ("language", "primary_language"),
        ("stargazers_count", "star_count"),
        ("forks_count", "fork_count"),
        ("pushed_at", "last_commit_date"),
        ("html_url", "repo_url"),
    ):
        if canonical not in result:
            result[canonical] = result.get(alias)
    result["language"] = result.get("language") or "Unknown"
    for key in ("stargazers_count", "forks_count", "open_issues_count"):
        result[key] = result.get(key) or 0
    return result
