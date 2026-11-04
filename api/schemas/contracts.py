"""Public DTO definitions for future source/repository APIs; no new endpoints."""

from core.contracts import (
    RepositoryContract,
    RevisionContract,
    SourceDocument,
    TextChunk,
    EvidenceSpan,
    FindingContract,
    AnalysisJob,
    ReportContract,
)

__all__ = [
    "RepositoryContract",
    "RevisionContract",
    "SourceDocument",
    "TextChunk",
    "EvidenceSpan",
    "FindingContract",
    "AnalysisJob",
    "ReportContract",
]
