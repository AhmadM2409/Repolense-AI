"""Explicit bounded J2 stages; no queue, automatic retry, fallback or historical ledger writes."""

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from functools import wraps
from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select

from api.schemas.career_matching import (
    CONSENT,
    AssessmentCreate,
    Extraction,
    ExtractionCheck,
    MatchCheck,
    MatchDraft,
    MatchNext,
)
from core.config import settings
from core.models import CareerAIBudget, CareerAssessment
from core.services import (
    career_context,
    career_input,
    career_retrieval,
    career_structural,
    hybrid_retrieval,
    keyword_retrieval,
    semantic_retrieval,
)
from core.services import career_evidence as policy
from ingestion.github import GitHubError
from rag.generator import hosted
from rag.generator.gemini_provider import MODEL, URL, Gemini
from rag.generator.grounding import unsafe
from rag.generator.local_provider import GenerationError


def boundary(fn):
    @wraps(fn)
    async def wrapped(db, *args, **kwargs):
        if db.new or db.dirty or db.deleted:
            raise RuntimeError("Career assessments require a clean session")
        try:
            return await fn(db, *args, **kwargs)
        except Exception:
            await db.rollback()
            raise

    return wrapped


def now():
    return datetime.now(UTC)


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def provider_identity():
    return policy.digest(
        {
            "model": MODEL,
            "endpoint": URL,
            "policy": "career-evidence-v1",
            "credential_digest": hashlib.sha256(hosted.credential().encode()).hexdigest(),
        }
    )


def make_provider(budget):
    return Gemini(hosted.credential(), budget)


async def snapshot(db, uid, identity, mode, fixed=None):
    try:
        data = await career_input.detail(db, uid, identity)
    except GitHubError:
        raise HTTPException(409, "Canonical portfolio evidence changed") from None
    if data["status"] != "ready" or not 1 <= len(data["selection"]) <= 5:
        raise HTTPException(409, "A ready portfolio of at most five selected revisions is required")
    manifest = []
    for item in data["selection"]:
        repo, rev = str(item["repository_id"]), str(item["revision_id"])
        snap = await keyword_retrieval._snapshot(
            db, str(uid), repo, rev, keyword_retrieval.SearchLimits.configured()
        )
        row = dict(
            repository_id=repo,
            revision_id=rev,
            commit_sha=snap.sha,
            digest=snap.digest,
            full_name=item["full_name"],
        )
        if mode != "keyword":
            config = semantic_retrieval.configuration("lmstudio")
            selected = next(
                (x["index_id"] for x in (fixed or []) if x["repository_id"] == repo), None
            )
            view = await semantic_retrieval.view(
                db,
                str(uid),
                repo,
                rev,
                config,
                keyword_retrieval.SearchLimits.configured(),
                selected,
            )
            row.update(index_id=view.index_id, index_signature=view.signature)
        manifest.append(row)
    return {
        "input_id": str(identity),
        "job_hash": data["job_hash"],
        "revisions": manifest,
        "discovery_status": data["discovery"]["status"],
        "version": 1,
    }, data["job_description"]


async def current(db, uid, identity, *, token=None, lock=False):
    row = await db.scalar(
        select(CareerAssessment)
        .where(CareerAssessment.id == str(identity), CareerAssessment.owner_id == str(uid))
        .execution_options(populate_existing=True)
    )
    if row is None:
        raise HTTPException(404, "Career assessment unavailable")
    await career_input.owned(db, uid, row.input_id, lock=lock)
    snap, text = await snapshot(db, uid, row.input_id, row.mode, row.snapshot["revisions"])
    if policy.digest(snap) != row.signature:
        raise HTTPException(409, "Portfolio evidence changed")
    if lock:
        row = await db.scalar(
            select(CareerAssessment)
            .where(CareerAssessment.id == str(identity))
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    if token is not None:
        if (
            row.operation_token != token
            or row.status not in {"extracting", "matching"}
            or row.lease_until is None
            or aware(row.lease_until) <= now()
        ):
            raise HTTPException(409, "Assessment operation cancelled, expired or changed")
        if (
            not settings.career_j2_enabled
            or row.consent != CONSENT
            or provider_identity() != row.provider_signature
        ):
            raise HTTPException(409, "Assessment provider policy changed")
    return row, text


# Hosted-inference budget. Owner and aggregate limits are unchanged lifetime caps. One assessment
# may use 12 attempts: J2 extraction/matching keeps its original 10 (2 + 4 requirements x 2), and
# the J3 roadmap gets its own 2, so matching can never consume the roadmap's attempts.
MAX_ASSESSMENT_CALLS = 12
MAX_MATCHING_CALLS = 10
MAX_ROADMAP_CALLS = 2
MAX_OWNER_CALLS = 40
MAX_AGGREGATE_CALLS = 100


def roadmap_phase(phase):
    return phase.startswith("roadmap/")


def remaining(calls, phase):
    """Attempts still available to this assessment for a call in `phase`."""
    used = sum(roadmap_phase(c["phase"]) for c in calls)
    scoped = (
        MAX_ROADMAP_CALLS - used
        if roadmap_phase(phase)
        else MAX_MATCHING_CALLS - (len(calls) - used)
    )
    return max(0, min(scoped, MAX_ASSESSMENT_CALLS - len(calls)))


def budget_remaining(budget, uid):
    """Attempts still available under the shared owner and aggregate caps."""
    return max(
        0,
        min(
            MAX_AGGREGATE_CALLS - budget.spent,
            MAX_OWNER_CALLS - budget.owners.get(str(uid), 0),
        ),
    )


MAX_CALLS_PER_MINUTE = 10


async def preflight(db, uid, calls, phase, needed=2):
    """Refuse a two-call step BEFORE creating rows or changing status when it cannot complete.

    Advisory only: the locked reservation in Operation.call stays authoritative. Budget refusals
    are 409; a full rolling-minute window is a transient 429 with Retry-After, so nothing is
    marked failed and the same step can simply be retried later.
    """
    budget = await db.scalar(select(CareerAIBudget).where(CareerAIBudget.id == 1))
    if budget is None or budget.stopped:
        raise HTTPException(409, "Career hosted processing is stopped (quota or billing latch)")
    if remaining(calls, phase) < needed:
        raise HTTPException(409, "Assessment has fewer than two remaining inference attempts")
    if budget_remaining(budget, uid) < needed:
        raise HTTPException(
            409, "Owner or shared hosted budget has fewer than two remaining attempts"
        )
    current_time = time.time()
    recent = sorted(x for x in budget.recent if x > current_time - 60)
    if len(recent) + needed > MAX_CALLS_PER_MINUTE:
        wait = max(
            1, int(recent[len(recent) + needed - MAX_CALLS_PER_MINUTE - 1] + 61 - current_time)
        )
        raise HTTPException(
            429,
            f"Hosted request rate limit reached; retry in about {wait} seconds",
            headers={"Retry-After": str(wait)},
        )


class Prepaid:
    stopped = False
    row = None

    def reserve(self):
        if self.row is None or self.stopped:
            raise GenerationError("request_budget")
        value, self.row = self.row, None
        return value

    def save(self):
        pass  # Adapter's temporary raw output/receipt never enters persistent accounting.


class Operation:
    def __init__(self, db, uid, identity, token):
        self.db, self.uid, self.identity, self.token = db, str(uid), str(identity), token
        self.deadline = asyncio.get_running_loop().time() + 110

    async def current(self):
        return await current(self.db, self.uid, self.identity, token=self.token, lock=True)

    async def call(self, system, payload, schema, phase, needed=1):
        """Reserve then send one call. `needed=2` on the first call of a two-call step
        (extract+verify, match+check, generate+check): the step starts only when both calls are
        affordable at every level, so a step never spends one call and then fails on budget."""
        schema = policy.compact(schema)
        user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        size = len(system.encode()) + len(user.encode()) + len(json.dumps(schema).encode())
        if size > 4096:
            raise ValueError("Bounded model context exceeded")
        prepaid = Prepaid()
        provider = make_provider(prepaid)
        provider.deadline = self.deadline

        async def reserve():
            row, _ = await self.current()
            budget = await self.db.scalar(
                select(CareerAIBudget)
                .where(CareerAIBudget.id == 1)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if budget is None or budget.stopped:
                raise GenerationError("quota_or_billing")
            recent = [x for x in budget.recent if x > time.time() - 60]
            if min(remaining(row.calls, phase), budget_remaining(budget, self.uid)) < needed:
                raise GenerationError("request_budget")
            if len(recent) >= 10:
                raise GenerationError("capacity")
            stamp = time.time()
            budget.spent += 1
            budget.owners = {**budget.owners, self.uid: budget.owners.get(self.uid, 0) + 1}
            budget.recent = recent + [stamp]
            row.calls = [*row.calls, {"phase": phase, "at": stamp, "state": "reserved"}]
            await self.db.commit()  # persisted before possible external POST
            prepaid.row = {"number": len(row.calls), "state": "reserved_before_post"}

        try:
            result = await provider.generate(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                reserve,
                response_schema=schema,
            )
        except GenerationError as exc:
            if prepaid.stopped or exc.code in {"authorization", "quota_or_billing"}:
                await self.db.rollback()
                budget = await self.db.scalar(
                    select(CareerAIBudget).where(CareerAIBudget.id == 1).with_for_update()
                )
                if budget:
                    budget.stopped = True
                    await self.db.commit()
            raise
        row, _ = await self.current()
        calls = list(row.calls)
        if not calls or calls[-1]["phase"] != phase:
            raise RuntimeError("Provider bypassed reservation")
        calls[-1] = {
            **calls[-1],
            "state": "response_received",
            "model": result.model,
            "usage": result.usage,
            "latency_seconds": result.latency_seconds,
        }
        row.calls = calls
        await self.db.commit()
        return result.content


async def fail(db, uid, identity, token, category):
    await db.rollback()
    row = await db.scalar(
        select(CareerAssessment)
        .where(CareerAssessment.id == str(identity), CareerAssessment.owner_id == str(uid))
        .with_for_update()
    )
    if row and row.operation_token == token:
        row.status = "failed"
        row.error_category = category
        row.operation_token = None
        row.lease_until = None
        await db.commit()
    await db.rollback()


async def operate(db, uid, identity, token, stage):
    operation = Operation(db, uid, identity, token)
    try:
        async with asyncio.timeout(120):
            row, text = await current(db, uid, identity, token=token)
            await db.rollback()
            if stage == "extract":
                raw = await operation.call(
                    policy.EXTRACT,
                    {"job": text},
                    Extraction.model_json_schema(),
                    "extract",
                    needed=2,
                )
                row, text = await current(db, uid, identity, token=token)
                values = policy.requirements(raw, text, row.snapshot["job_hash"])
                await db.rollback()
                check = await operation.call(
                    policy.EXTRACT_CHECK,
                    {"job": text, "requirements": values},
                    ExtractionCheck.model_json_schema(),
                    "verify_extraction",
                )
                policy.validate_extraction(check, values)
                row, _ = await current(db, uid, identity, token=token, lock=True)
                row.requirements = values
                row.status = "review"
            else:
                row, _ = await current(db, uid, identity, token=token)
                requirement = row.requirements[len(row.matches)]
                manifest = row.snapshot["revisions"]
                mode = row.mode
                await db.rollback()
                selected, scopes, context, coverage = await gather(
                    db, uid, manifest, mode, requirement
                )
                if not selected:
                    match = {
                        "classification": "not evidenced",
                        "explanation": (
                            "No safe matching excerpts were found "
                            "within the selected retrieval bounds."
                        ),
                        "uncertainty": (
                            "This does not establish that the developer lacks the skill."
                        ),
                        "citations": [],
                        "support_verdict": "insufficient",
                    }
                else:
                    payload = {"requirement": requirement, "evidence": context}
                    raw = await operation.call(
                        policy.MATCH,
                        payload,
                        MatchDraft.model_json_schema(),
                        "match/" + requirement["id"],
                        needed=2,
                    )
                    MatchDraft.model_validate_json(raw)
                    check = await operation.call(
                        policy.MATCH_CHECK,
                        {**payload, "candidate": json.loads(raw)},
                        MatchCheck.model_json_schema(),
                        "check/" + requirement["id"],
                    )
                    match = policy.accepted(raw, check, selected, scopes)
                row, _ = await current(db, uid, identity, token=token, lock=True)
                row.matches = [
                    *row.matches,
                    {"requirement_id": requirement["id"], **match, "coverage": coverage},
                ]
                row.status = "complete" if len(row.matches) == len(row.requirements) else "review"
            row.operation_token = None
            row.lease_until = None
            await db.commit()
    except HTTPException:
        await fail(db, uid, identity, token, "scope_changed")
        raise
    except GenerationError as exc:
        await fail(
            db,
            uid,
            identity,
            token,
            exc.code
            if exc.code in {"capacity", "request_budget", "quota_or_billing", "authorization"}
            else "provider_failure",
        )
    except (ValueError, KeyError, TypeError):
        await fail(db, uid, identity, token, "validation_failure")
    except TimeoutError:
        await fail(db, uid, identity, token, "timeout_unknown")
    except asyncio.CancelledError:
        await asyncio.shield(fail(db, uid, identity, token, "interrupted_unknown"))
        raise
    except Exception:
        await fail(db, uid, identity, token, "execution_failure")
        raise HTTPException(503, "Assessment execution unavailable") from None
    return await detail(db, uid, identity)


@boundary
async def create(db, uid, input_id, data):
    data = AssessmentCreate.model_validate(data)
    await career_input.owned(db, uid, input_id, lock=True)
    existing = await db.scalar(
        select(CareerAssessment).where(
            CareerAssessment.owner_id == str(uid),
            CareerAssessment.request_id == str(data.request_id),
        )
    )
    if existing:
        if existing.input_id != str(input_id) or existing.mode != data.retrieval_mode:
            raise HTTPException(409, "Request identity already used with different inputs")
        return await detail(db, uid, existing.id)
    if not settings.career_j2_enabled:
        raise HTTPException(409, "Career hosted processing disabled")
    snap, text = await snapshot(db, uid, input_id, data.retrieval_mode)
    if len(text.encode()) > 1500 or unsafe(text):
        raise HTTPException(422, "Job text exceeds safe bounded extraction context")
    await preflight(db, uid, [], "extract")
    signature = provider_identity()
    record = await career_input.owned(db, uid, input_id, lock=True)
    token = str(uuid4())
    row = CareerAssessment(
        owner_id=str(uid),
        input_id=str(input_id),
        profile_id=record.profile_id,
        request_id=str(data.request_id),
        snapshot=snap,
        signature=policy.digest(snap),
        provider_signature=signature,
        mode=data.retrieval_mode,
        consent=data.hosted_processing_consent,
        status="extracting",
        operation_token=token,
        lease_until=now() + timedelta(seconds=120),
    )
    db.add(row)
    await db.commit()
    return await operate(db, uid, row.id, token, "extract")


@boundary
async def match_next(db, uid, identity, data):
    data = MatchNext.model_validate(data)
    row, _ = await current(db, uid, identity, lock=True)
    if data.requirement_id not in {r["id"] for r in row.requirements}:
        raise HTTPException(422, "Unknown assessment requirement")
    if data.requirement_id in {r["requirement_id"] for r in row.matches}:
        return await detail(db, uid, identity)
    if row.status == "complete":
        return await detail(db, uid, identity)
    if row.operation_token:
        if aware(row.lease_until) <= now():
            row.status = "failed"
            row.error_category = "interrupted_unknown"
            row.operation_token = None
            await db.commit()
            return await detail(db, uid, identity)
        raise HTTPException(409, "Assessment operation already in progress")
    if row.status != "review" or not row.requirements:
        raise HTTPException(409, "Assessment is not ready for reviewed matching")
    if not settings.career_j2_enabled or provider_identity() != row.provider_signature:
        raise HTTPException(409, "Career hosted processing unavailable or changed")
    if data.requirement_id != row.requirements[len(row.matches)]["id"]:
        raise HTTPException(409, "Match the next reviewed requirement in order")
    await preflight(db, uid, row.calls, "match/" + data.requirement_id)
    token = str(uuid4())
    row.status = "matching"
    row.operation_token = token
    row.lease_until = now() + timedelta(seconds=120)
    await db.commit()
    return await operate(db, uid, identity, token, "match")


async def detail(db, uid, identity):
    row, _ = await current(db, uid, identity)
    return {
        "id": row.id,
        "input_id": row.input_id,
        "status": row.status,
        "error_category": row.error_category,
        "manifest": row.snapshot,
        "requirements": row.requirements,
        "matches": row.matches if row.status in {"review", "complete"} else [],
        "inference_spent": len(row.calls),
        "policy_version": "career-evidence-v1",
        "model": "gemini-3.5-flash-lite",
        "retrieval_mode": row.mode,
        "created_at": row.created_at,
        "operation_expired": bool(
            row.operation_token and row.lease_until and aware(row.lease_until) <= now()
        ),
        "coverage_notice": (
            "Only selected, bounded GitHub evidence is assessed; "
            "not evidenced is not absent ability."
        ),
    }


@boundary
async def cancel(db, uid, identity):
    row, _ = await current(db, uid, identity, lock=True)
    if row.status == "complete":
        raise HTTPException(409, "Assessment already complete")
    if row.status != "failed":
        row.status = "cancelled"
        row.operation_token = None
        row.lease_until = None
        await db.commit()
    return await detail(db, uid, identity)


async def listing(db, uid, input_id):
    await career_input.owned(db, uid, input_id)
    rows = (
        await db.scalars(
            select(CareerAssessment)
            .where(
                CareerAssessment.owner_id == str(uid), CareerAssessment.input_id == str(input_id)
            )
            .order_by(CareerAssessment.created_at.desc())
            .limit(20)
        )
    ).all()
    return {
        "items": [
            {
                "id": r.id,
                "status": r.status,
                "created_at": r.created_at,
                "inference_spent": len(r.calls),
            }
            for r in rows
        ]
    }


async def gather(db, uid, manifest, mode, requirement):
    selected, scopes, context = {}, {}, {}
    truncated = filtered = budget_conflicts = 0
    pools, candidates_count = [], 0
    structural = {
        "structural_scanned_files": 0,
        "structural_scanned_bytes": 0,
        "structural_hints": 0,
        "structural_seed_candidates": 0,
        "structural_replaced_lexical_candidates": 0,
        "structural_scan_limited": False,
    }
    # Every mode uses the same requirement query, structural hints, implementation groups and
    # window rules. Semantic/hybrid only change where candidates come from (local Nomic vectors,
    # fused with BM25 for hybrid); similarity is never proof, and "demonstrated" still requires
    # a complete, validated canonical implementation group.
    lexical = career_retrieval.query(requirement)
    semantic = mode != "keyword"
    query = lexical.text
    for row in manifest:
        request = {"query": query, "mode": mode, "top_k": 5}
        if semantic:
            request.update(configuration="lmstudio", index_id=row["index_id"])
        services = {
            "keyword": keyword_retrieval,
            "semantic": semantic_retrieval,
            "hybrid": hybrid_retrieval,
        }
        results = await services[mode].search(
            db, uid, row["repository_id"], row["revision_id"], request
        )
        await db.rollback()
        candidates = list(results.matches)
        retrieved = frozenset(str(c.chunk_id) for c in candidates) if semantic else frozenset()
        groups, parsed, hints = [], set(), []
        if lexical:
            limits = keyword_retrieval.SearchLimits.configured()
            canonical = await keyword_retrieval._snapshot(
                db, str(uid), row["repository_id"], row["revision_id"], limits
            )
            await db.rollback()  # No row locks across bounded static analysis.
            hint_seeds, hints, hint_stats = career_structural.discover(canonical, lexical)
            for key, value in hint_stats.items():
                structural[key] = (
                    structural[key] or value if isinstance(value, bool) else structural[key] + value
                )
            structural["structural_seed_candidates"] += len(hint_seeds)
            # Same five-seed bound; at most two source-verified structural seeds.
            seen = {str(c.chunk_id) for c in hint_seeds}
            fused = [*hint_seeds, *(c for c in candidates if str(c.chunk_id) not in seen)]
            structural["structural_replaced_lexical_candidates"] += max(0, len(fused) - 5)
            candidates = fused[:5]
            groups, stats = career_context.groups(
                canonical, candidates, lexical, hints, retrieved=retrieved
            )
            current_snapshot = await keyword_retrieval._snapshot(
                db, str(uid), row["repository_id"], row["revision_id"], limits
            )
            if current_snapshot.digest != canonical.digest:
                raise keyword_retrieval._conflict()
            await db.rollback()
            parsed = stats["parsed_chunks"]
            budget_conflicts += stats["budget_conflicts"] + int(stats["routine_limit"])
        candidates_count += len(candidates)
        for position, candidate in enumerate(candidates):
            if (
                any(p.startswith(".env") for p in candidate.path.lower().split("/"))
                or candidate.path.lower().endswith((".pem", ".key"))
                or unsafe(candidate.excerpt)
                or career_structural.excluded_asset(candidate.path, lexical)
            ):
                filtered += 1
                continue
            if str(candidate.chunk_id) in parsed:
                continue  # No clipped fallback bypass of an incomplete function group.
            found = career_retrieval.window(candidate, lexical)
            if found is not None:
                score, start, end, excerpt = found
            elif not semantic:
                filtered += 1
                continue
            else:
                # Semantic candidate with no lexical overlap: bounded leading window, ranked below
                # every lexically supported window. It is evidence to inspect, not proof.
                lines = []
                for line in candidate.excerpt.splitlines()[:8]:
                    if len("\n".join(lines + [line]).encode()) > 300:
                        break
                    lines.append(line)
                if not lines or not "\n".join(lines).strip():
                    filtered += 1
                    continue
                start, end, excerpt = 0, len(lines) - 1, "\n".join(lines)
                score = (False, False, 0, -1 - position)
            item = candidate.model_copy(
                update={
                    "excerpt": excerpt,
                    "line_start": candidate.line_start + start,
                    "line_end": candidate.line_start + end,
                }
            )
            groups.append(
                career_context.Group(
                    tuple(score[:4]),
                    (item,),
                    None,
                    clipped=(start > 0 or end + 1 < len(candidate.excerpt.splitlines()),),
                )
            )
        if lexical:
            groups.sort(key=lambda group: (group.priority, -len(group.windows)), reverse=True)
        pools.append((row, groups))
    used, chosen_groups = set(), 0
    for _ in range(2 if lexical else 1):
        for row, groups in pools:
            while groups and len(selected) < 5:
                group = groups.pop(0)
                identities = [
                    (
                        row["repository_id"],
                        row["revision_id"],
                        str(w.chunk_id),
                        w.line_start,
                        w.line_end,
                    )
                    for w in group.windows
                ]
                if any(key in used for key in identities):
                    continue
                if len(selected) + len(group.windows) > 5:
                    budget_conflicts += 1
                    continue
                ids = ["E" + str(len(selected) + n + 1) for n in range(len(group.windows))]
                proposed = dict(context)
                for n, (key, window) in enumerate(zip(ids, group.windows, strict=True)):
                    value = {
                        "repository": row["full_name"],
                        "path": window.path,
                        "kind": policy.kind(window.path, window.excerpt),
                        "text": window.excerpt,
                    }
                    if n == 0 and group.complete is not None:
                        value["requires"] = ids
                        if not group.complete:
                            value["missing_local_implementation"] = list(group.missing)
                    proposed[key] = value
                if (
                    len(
                        json.dumps(
                            {"requirement": requirement, "evidence": proposed}, ensure_ascii=False
                        ).encode()
                    )
                    > 1800
                ):
                    budget_conflicts += 1
                    continue  # Entire implementation group or nothing; no hidden clipping.
                for index, (key, window) in enumerate(zip(ids, group.windows, strict=True)):
                    selected[key] = SimpleNamespace(
                        **{
                            **window.model_dump(),
                            "implementation_complete": group.complete,
                            "required_evidence": ids if group.complete is not None else [],
                        }
                    )
                    scopes[key] = (row["repository_id"], row["revision_id"])
                    truncated += int(
                        getattr(
                            window, "is_clipped", group.clipped[index] if group.clipped else False
                        )
                    )
                context = proposed
                used.update(identities)
                chosen_groups += 1
                break
    return (
        selected,
        scopes,
        context,
        {
            "searched_revisions": len(manifest),
            "included_excerpts": len(selected),
            "omitted_candidates": max(0, candidates_count - chosen_groups),
            "context_limited": bool(
                budget_conflicts
                or candidates_count > chosen_groups
                or truncated
                or structural["structural_scan_limited"]
                or structural["structural_replaced_lexical_candidates"]
            ),
            **structural,
            "truncated_excerpts": truncated,
            "relevance_filtered_candidates": filtered,
            "implementation_budget_conflicts": budget_conflicts,
            "mode": mode,
        },
    )
