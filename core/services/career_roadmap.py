"""J3 synchronous fenced roadmap; J2 limits are shared, never reset or supplemented."""

import asyncio
import json
from datetime import timedelta
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select

from api.schemas.career_matching import CONSENT, CareerCitation
from api.schemas.career_roadmap import ProposalCheck, Proposals, RoadmapCreate
from core.config import settings
from core.models import CareerRoadmap, IngestedSource, SourceChunk
from core.services import career_assessment as assessment
from core.services import career_roadmap_policy as policy
from core.services.career_evidence import digest
from rag.generator.local_provider import GenerationError


def signature(parent):
    return digest(
        [
            parent.snapshot,
            parent.requirements,
            parent.matches,
            parent.provider_signature,
            parent.consent,
        ]
    )


async def parent_scope(db, uid, identity, *, lock=False):
    parent, _ = await assessment.current(db, uid, identity, lock=lock)
    if parent.status != "complete":
        raise HTTPException(409, "A completed verified assessment is required")
    policy.eligible(parent)
    for match in parent.matches:
        for raw in match["citations"]:
            c = CareerCitation.model_validate(raw)
            source = await db.get(IngestedSource, str(c.source_id))
            chunk = await db.get(SourceChunk, str(c.chunk_id))
            scope = next(
                (
                    r
                    for r in parent.snapshot["revisions"]
                    if r["repository_id"] == str(c.repository_id)
                    and r["revision_id"] == str(c.revision_id)
                    and r["commit_sha"] == c.commit_sha
                ),
                None,
            )
            if (
                scope is None
                or source is None
                or chunk is None
                or str(source.profile_id) != parent.profile_id
                or str(source.revision_id) != str(c.revision_id)
                or source.path != c.path
                or str(chunk.source_id) != str(c.source_id)
                or str(chunk.profile_id) != parent.profile_id
                or not chunk.line_start <= c.line_start <= c.line_end <= chunk.line_end
            ):
                raise HTTPException(409, "Roadmap canonical evidence changed")
            lines = source.text.splitlines()
            if not c.quote.strip() or c.quote not in "\n".join(
                lines[c.line_start - 1 : c.line_end]
            ):
                raise HTTPException(409, "Roadmap citation quote changed")
    return parent


async def current(db, uid, identity, *, token=None, lock=False):
    row = await db.scalar(
        select(CareerRoadmap)
        .where(CareerRoadmap.id == str(identity))
        .execution_options(populate_existing=True)
    )
    if row is None:
        raise HTTPException(404, "Career roadmap unavailable")
    # Parent/profile first, then roadmap: consistent lock ordering with J2 and cancellation.
    parent = await parent_scope(db, uid, row.assessment_id, lock=lock)
    if lock:
        row = await db.scalar(
            select(CareerRoadmap)
            .where(CareerRoadmap.id == str(identity))
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    if signature(parent) != row.source_signature:
        raise HTTPException(409, "Roadmap assessment snapshot changed")
    if token is not None:
        if (
            row.status != "generating"
            or row.operation_token != token
            or row.lease_until is None
            or assessment.aware(row.lease_until) <= assessment.now()
        ):
            raise HTTPException(409, "Roadmap cancelled, expired or changed")
        if (
            not settings.career_j2_enabled
            or parent.consent != CONSENT
            or assessment.provider_identity() != parent.provider_signature
        ):
            raise HTTPException(409, "Career provider policy changed")
    return row, parent


class Operation(assessment.Operation):
    async def current(self):
        _, parent = await current(self.db, self.uid, self.identity, token=self.token, lock=True)
        return parent, None


async def detail(db, uid, identity):
    row, parent = await current(db, uid, identity)
    return dict(
        id=row.id,
        assessment_id=row.assessment_id,
        version=row.version,
        status=row.status,
        error_category=row.error_category,
        result=row.result if row.status == "complete" else None,
        source_signature=row.source_signature,
        manifest=parent.snapshot,
        inference_spent=len(parent.calls),
        roadmap_inference_spent=sum(c["phase"].startswith("roadmap/") for c in parent.calls),
        policy_version="career-roadmap-v1",
        model="gemini-3.5-flash-lite",
        created_at=row.created_at,
        operation_expired=bool(
            row.operation_token
            and row.lease_until
            and assessment.aware(row.lease_until) <= assessment.now()
        ),
    )


async def fail(db, identity, token, category):
    await db.rollback()
    row = await db.scalar(
        select(CareerRoadmap).where(CareerRoadmap.id == str(identity)).with_for_update()
    )
    if row and row.status == "generating" and row.operation_token == token:
        row.status = "failed"
        row.error_category = category
        row.operation_token = None
        row.lease_until = None
        await db.commit()
    await db.rollback()


@assessment.boundary
async def create(db, uid, identity, data):
    RoadmapCreate.model_validate(data)
    parent = await parent_scope(db, uid, identity, lock=True)
    row = await db.scalar(
        select(CareerRoadmap).where(CareerRoadmap.assessment_id == str(identity)).with_for_update()
    )
    if row:
        if (
            row.status == "generating"
            and row.lease_until
            and assessment.aware(row.lease_until) <= assessment.now()
        ):
            row.status = "failed"
            row.error_category = "interrupted_unknown"
            row.operation_token = None
            row.lease_until = None
            await db.commit()
        return await detail(db, uid, row.id)
    if (
        not settings.career_j2_enabled
        or parent.consent != CONSENT
        or assessment.provider_identity() != parent.provider_signature
    ):
        raise HTTPException(409, "Career hosted processing unavailable or changed")
    if policy.eligible(parent):
        # Before the single roadmap row exists, so a refusal (including a transient rate-limit
        # 429) never leaves a permanently failed roadmap for this assessment.
        await assessment.preflight(db, uid, parent.calls, "roadmap/generate")
    token = str(uuid4())
    row = CareerRoadmap(
        assessment_id=parent.id,
        source_signature=signature(parent),
        operation_token=token,
        lease_until=assessment.now() + timedelta(seconds=120),
    )
    db.add(row)
    await db.commit()
    roadmap_id = row.id
    try:
        async with asyncio.timeout(120):
            row, parent = await current(db, uid, roadmap_id, token=token)
            if not policy.eligible(parent):
                final = policy.result(parent, [])
                await db.rollback()
            else:
                payload = policy.payload(parent)
                await db.rollback()
                operation = Operation(db, uid, roadmap_id, token)
                raw = await operation.call(
                    policy.GENERATE,
                    payload,
                    Proposals.model_json_schema(),
                    "roadmap/generate",
                    needed=2,
                )
                Proposals.model_validate_json(raw)
                checked = await operation.call(
                    policy.CHECK,
                    {**payload, "candidate": json.loads(raw)},
                    ProposalCheck.model_json_schema(),
                    "roadmap/check",
                )
                _, parent = await current(db, uid, roadmap_id, token=token)
                final = policy.bind(parent, raw, checked)
                await db.rollback()
            row, _ = await current(db, uid, roadmap_id, token=token, lock=True)
            row.result = final
            row.status = "complete"
            row.operation_token = None
            row.lease_until = None
            await db.commit()
    except HTTPException:
        await fail(db, roadmap_id, token, "scope_changed")
        raise
    except GenerationError as exc:
        await fail(
            db,
            roadmap_id,
            token,
            exc.code
            if exc.code in {"capacity", "request_budget", "quota_or_billing", "authorization"}
            else "provider_failure",
        )
    except (ValueError, KeyError, TypeError):
        await fail(db, roadmap_id, token, "validation_failure")
    except TimeoutError:
        await fail(db, roadmap_id, token, "timeout_unknown")
    except asyncio.CancelledError:
        await asyncio.shield(fail(db, roadmap_id, token, "interrupted_unknown"))
        raise
    except Exception:
        await fail(db, roadmap_id, token, "execution_failure")
        raise HTTPException(503, "Roadmap execution unavailable") from None
    return await detail(db, uid, roadmap_id)


@assessment.boundary
async def cancel(db, uid, identity):
    row, _ = await current(db, uid, identity, lock=True)
    if row.status == "complete":
        raise HTTPException(409, "Roadmap already complete")
    if row.status == "generating":
        row.status = "cancelled"
        row.operation_token = None
        row.lease_until = None
        await db.commit()
    return await detail(db, uid, identity)


async def by_assessment(db, uid, identity):
    await parent_scope(db, uid, identity)
    row = await db.scalar(select(CareerRoadmap).where(CareerRoadmap.assessment_id == str(identity)))
    if row is None:
        raise HTTPException(404, "Career roadmap unavailable")
    return await detail(db, uid, row.id)
