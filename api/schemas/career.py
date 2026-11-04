"""J1 inputs only: no extraction, scoring or hosted consent grant."""

import hashlib
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from core.contracts import Contract
from ingestion.github_profile import profile_account


def job_text(value):
    if not isinstance(value, str):
        raise ValueError("Job description must be text")
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise ValueError("Invalid UTF-8 job text") from None
    if (
        not 20 <= len(value) <= 20000
        or size > 65536
        or sum(c.isalpha() for c in value) < 10
        or any(ord(c) < 32 and c not in "\n\t" for c in value)
    ):
        raise ValueError("Job text must be meaningful and within limits")
    return value


def content_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class InputCreate(Contract):
    profile_id: UUID
    github_profile_url: str = Field(strict=True, max_length=200)
    job_description: str = Field(strict=True, max_length=20000)

    @field_validator("github_profile_url")
    @classmethod
    def profile(cls, value):
        return "https://github.com/" + profile_account(value)

    @field_validator("job_description")
    @classmethod
    def description(cls, value):
        return job_text(value)


class Selection(Contract):
    repositories: list[str] = Field(min_length=1, max_length=10)


class DiscoveredRepository(Contract):
    full_name: str
    github_id: int
    eligible: bool
    excluded_reason: str | None
    public: bool
    fork: bool
    archived: bool
    url: str
    description: str | None
    language: str | None
    default_branch: str | None
    updated_at: str | None


class DiscoveryResult(Contract):
    version: Literal[1]
    status: Literal["complete", "bounded", "partial", "unavailable"]
    reason: str | None
    items: list[DiscoveredRepository] = Field(max_length=120)
    requests: int = Field(ge=0, le=5)
    private_coverage: Literal["not_accessed_or_enumerated"]


class SelectedRepository(Contract):
    full_name: str
    position: int
    job_id: UUID | None
    status: Literal[
        "not_started", "pending", "processing", "awaiting_pin", "ready", "failed", "cancelled"
    ]
    error_category: str | None
    repository_id: UUID | None
    revision_id: UUID | None
    commit_sha: str | None


class InputDetail(Contract):
    id: UUID
    profile_id: UUID
    github_profile_url: str
    job_description: str
    job_hash: str
    manifest_version: Literal[1]
    discovery: DiscoveryResult
    selection: list[SelectedRepository] = Field(max_length=10)
    status: Literal["input_created", "selection_saved", "preparing", "ready", "partial", "failed"]
    created_at: datetime
    updated_at: datetime
    future_hosted_consent: Literal["not_granted"]


class InputSummary(Contract):
    id: UUID
    profile_id: UUID
    account: str
    created_at: datetime
    started: bool


class InputList(Contract):
    items: list[InputSummary] = Field(max_length=20)
    next_offset: int | None
