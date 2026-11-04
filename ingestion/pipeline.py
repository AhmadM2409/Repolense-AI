import hashlib
import json
from uuid import uuid5, NAMESPACE_URL
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.services.profile_service import get_profile
from fastapi import HTTPException
from dataclasses import dataclass

import structlog

from core.models.ingested_source import IngestedSource
from core.models.evidence import SourceChunk
from sqlalchemy import or_

from .chunking.base import source_line_chunks
from .embeddings.batch_processor import BatchEmbeddingProcessor
from .embeddings.provider import EmbeddingProvider
from .parsers.readme_parser import ReadmeParser
from .parsers.repo_analyzer import RepoAnalyzer
from .parsers.resume_parser import ResumeParser

logger = structlog.get_logger()


@dataclass
class IngestResult:
    """Result of ingesting a source."""

    source_id: str
    chunk_count: int
    skipped: bool
    skip_reason: str | None = None


class IngestionPipeline:
    """Main orchestration for document ingestion and embedding."""

    def __init__(
        self,
        vector_db,
        db_session: AsyncSession,
        embedding_provider: EmbeddingProvider,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            vector_db: ChromaDB collection for storing embeddings
            db_session: Database session for storing metadata
            embedding_provider: EmbeddingProvider for generating embeddings
        """
        self.vector_db = vector_db
        self.db_session = db_session
        self.embedding_provider = embedding_provider

        self.batch_processor = BatchEmbeddingProcessor(embedding_provider, vector_db)

        # Initialize parsers
        self.resume_parser = ResumeParser()
        self.readme_parser = ReadmeParser()
        self.repo_analyzer = RepoAnalyzer()

    async def ingest_resume(
        self,
        profile_id: str,
        content: str | bytes,
        filename: str,
        *, user_id: str,
    ) -> IngestResult:
        """
        Ingest a resume document.

        Args:
            profile_id: ID of the profile owner
            content: Resume content (PDF bytes or markdown string)
            filename: Original filename

        Returns:
            IngestResult with ingestion status
        """
        if not isinstance(self.db_session, AsyncSession):
            raise TypeError("IngestionPipeline requires AsyncSession")
        if await get_profile(self.db_session, profile_id, user_id) is None:
            raise HTTPException(404, "Profile not found")
        content_hash = self._hash_content(content)
        source_id = f"resume_{profile_id}_{content_hash}"
        source_id = str(uuid5(NAMESPACE_URL, source_id))

        logger.info(
            "Starting resume ingestion",
            profile_id=profile_id,
            filename=filename,
            source_id=source_id,
        )

        # Check if already ingested
        skip_result = await self._check_skip(
            source_id=source_id,
            source_type="resume",
            profile_id=profile_id,
            source_url=None,
            content_hash=content_hash,
        )
        if skip_result:
            return skip_result

        try:
            # Parse resume
            parse_result = self.resume_parser.parse(content.decode("utf-8") if isinstance(content, bytes) and not filename.lower().endswith(".pdf") else content)
            logger.info(
                "Resume parsed successfully",
                sections=parse_result.metadata.get("detected_sections"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "filename": filename,
                    "source_type": "resume",
                }
            )

            # Chunk the content
            chunks = source_line_chunks(parse_result.text, metadata)
            logger.info("Resume chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("Resume embeddings stored", chunk_count=len(chunks))

            # Recheck current ownership/activity after the external embedding call.
            if await get_profile(self.db_session, profile_id, user_id) is None:
                raise HTTPException(404, "Profile not found")
            # Record in database
            await self._record_ingested_source(
                source_id=source_id,
                source_type="resume",
                profile_id=profile_id,
                source_url=None,
                content_hash=content_hash,
                chunk_count=len(chunks),
                text=parse_result.text,
                source_metadata=metadata,
                chunks=chunks,
                filename=filename,
            )

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            await self.db_session.rollback()
            try:
                self.vector_db.delete(where={"source_id": source_id})
            except Exception:
                logger.error("Vector compensation failed", source_id=source_id)
            logger.error(
                "Resume ingestion failed",
                profile_id=profile_id,
                filename=filename,
                error=str(e),
            )
            raise

    async def ingest_readme(
        self,
        profile_id: str,
        repo_name: str,
        content: str | bytes,
        *, user_id: str,
    ) -> IngestResult:
        """
        Ingest a README document.

        Args:
            profile_id: ID of the profile owner
            repo_name: Name of the repository
            content: README content (markdown string or bytes)

        Returns:
            IngestResult with ingestion status
        """
        if not isinstance(self.db_session, AsyncSession):
            raise TypeError("IngestionPipeline requires AsyncSession")
        if await get_profile(self.db_session, profile_id, user_id) is None:
            raise HTTPException(404, "Profile not found")
        content_hash = self._hash_content(content)
        source_id = f"readme_{profile_id}_{repo_name}_{content_hash}"
        source_id = str(uuid5(NAMESPACE_URL, source_id))

        logger.info(
            "Starting README ingestion",
            profile_id=profile_id,
            repo_name=repo_name,
            source_id=source_id,
        )

        # Check if already ingested
        skip_result = await self._check_skip(
            source_id=source_id,
            source_type="readme",
            profile_id=profile_id,
            source_url=repo_name,
            content_hash=content_hash,
        )
        if skip_result:
            return skip_result

        try:
            # Parse README
            parse_result = self.readme_parser.parse(content)
            logger.info(
                "README parsed successfully",
                heading_count=parse_result.metadata.get("heading_count"),
                word_count=parse_result.metadata.get("word_count"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "repo_name": repo_name,
                    "source_type": "readme",
                }
            )

            # Chunk the content
            chunks = source_line_chunks(parse_result.text, metadata)
            logger.info("README chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("README embeddings stored", chunk_count=len(chunks))

            # Recheck current ownership/activity after the external embedding call.
            if await get_profile(self.db_session, profile_id, user_id) is None:
                raise HTTPException(404, "Profile not found")
            # Record in database
            await self._record_ingested_source(
                source_id=source_id,
                source_type="readme",
                profile_id=profile_id,
                source_url=repo_name,
                content_hash=content_hash,
                chunk_count=len(chunks),
                text=parse_result.text,
                source_metadata=metadata,
                chunks=chunks,
            )

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            await self.db_session.rollback()
            try:
                self.vector_db.delete(where={"source_id": source_id})
            except Exception:
                logger.error("Vector compensation failed", source_id=source_id)
            logger.error(
                "README ingestion failed",
                profile_id=profile_id,
                repo_name=repo_name,
                error=str(e),
            )
            raise

    async def ingest_repo_metadata(
        self,
        profile_id: str,
        repo_data: dict,
        *, user_id: str,
    ) -> IngestResult:
        """
        Ingest repository metadata.

        Args:
            profile_id: ID of the profile owner
            repo_data: Repository metadata dictionary

        Returns:
            IngestResult with ingestion status
        """
        repo_name = repo_data.get("name", "unknown")
        if not isinstance(self.db_session, AsyncSession):
            raise TypeError("IngestionPipeline requires AsyncSession")
        if await get_profile(self.db_session, profile_id, user_id) is None:
            raise HTTPException(404, "Profile not found")
        content_hash = self._hash_content(json.dumps(repo_data, sort_keys=True, separators=(",", ":")))
        source_id = f"repo_{profile_id}_{repo_name}_{content_hash}"
        source_id = str(uuid5(NAMESPACE_URL, source_id))

        logger.info(
            "Starting repo metadata ingestion",
            profile_id=profile_id,
            repo_name=repo_name,
            source_id=source_id,
        )

        # Check if already ingested
        skip_result = await self._check_skip(
            source_id=source_id,
            source_type="repo",
            profile_id=profile_id,
            source_url=repo_name,
            content_hash=content_hash,
        )
        if skip_result:
            return skip_result

        try:
            # Analyze repository
            parse_result = self.repo_analyzer.parse(repo_data)
            logger.info(
                "Repository analyzed successfully",
                language=parse_result.metadata.get("primary_language"),
                tech_stack=parse_result.metadata.get("tech_stack"),
            )

            # Prepare metadata
            metadata = parse_result.metadata.copy()
            metadata.update(
                {
                    "source_id": source_id,
                    "profile_id": profile_id,
                    "source_type": "repo",
                }
            )

            # Chunk the content
            chunks = source_line_chunks(parse_result.text, metadata)
            logger.info("Repository metadata chunked successfully", chunk_count=len(chunks))

            # Generate embeddings and store
            self.batch_processor.process(chunks)
            logger.info("Repository embeddings stored", chunk_count=len(chunks))

            # Recheck current ownership/activity after the external embedding call.
            if await get_profile(self.db_session, profile_id, user_id) is None:
                raise HTTPException(404, "Profile not found")
            # Record in database
            await self._record_ingested_source(
                source_id=source_id,
                source_type="repo",
                profile_id=profile_id,
                source_url=repo_name,
                content_hash=content_hash,
                chunk_count=len(chunks),
                text=parse_result.text,
                source_metadata=metadata,
                chunks=chunks,
            )

            return IngestResult(
                source_id=source_id,
                chunk_count=len(chunks),
                skipped=False,
            )

        except Exception as e:
            await self.db_session.rollback()
            try:
                self.vector_db.delete(where={"source_id": source_id})
            except Exception:
                logger.error("Vector compensation failed", source_id=source_id)
            logger.error(
                "Repository ingestion failed",
                profile_id=profile_id,
                repo_name=repo_name,
                error=str(e),
            )
            raise

    def _hash_content(self, content: str | bytes) -> str:
        """Generate a hash of content for deduplication."""
        if isinstance(content, str):
            content = content.encode()
        return hashlib.sha256(content).hexdigest()

    async def _check_skip(
        self,
        source_id: str,
        source_type: str,
        profile_id: str,
        source_url: str | None,
        content_hash: str,
    ) -> IngestResult | None:
        """
        Check if source has already been ingested.

        Returns IngestResult if should skip, None if should proceed.
        """
        existing = await self.db_session.scalar(select(IngestedSource).where(
            IngestedSource.profile_id == profile_id,
            IngestedSource.source_type == source_type,
            IngestedSource.source_url == source_url,
            or_(IngestedSource.content_hash == content_hash,
                IngestedSource.source_metadata["raw_content_hash"].as_string() == content_hash),
        ))
        if existing:
            return IngestResult(source_id=existing.id, chunk_count=0, skipped=True,
                                skip_reason="Source content has not changed")
        return None

    async def _record_ingested_source(
        self,
        source_id: str,
        source_type: str,
        profile_id: str,
        source_url: str | None,
        content_hash: str,
        chunk_count: int,
        filename: str | None = None,
        text: str | None = None,
        source_metadata: dict | None = None,
        chunks: list | None = None,
    ) -> None:
        """
        Record that a source has been ingested.

        Args:
            source_id: Unique ID for the source
            source_type: Type of source (resume, readme, repo)
            profile_id: ID of profile owner
            source_url: Source URL or repository identity
            content_hash: Full SHA-256 hash of the source content
            chunk_count: Number of chunks created
            filename: Optional filename for uploaded sources
        """
        record = IngestedSource(
            id=source_id, profile_id=profile_id, source_type=source_type,
            source_url=source_url, filename=filename, content_hash=self._hash_content(text or ""),
            chunk_count=chunk_count, text=text,
            source_metadata={**(source_metadata or {}), "raw_content_hash": content_hash},

        )
        self.db_session.add(record)
        await self.db_session.flush()
        self.db_session.add_all(
            SourceChunk(profile_id=profile_id, source_id=source_id,
                        chunk_index=chunk.chunk_index, text=chunk.text,
                        line_start=chunk.metadata["line_start"], line_end=chunk.metadata["line_end"])
            for chunk in (chunks or [])
        )
        await self.db_session.commit()
