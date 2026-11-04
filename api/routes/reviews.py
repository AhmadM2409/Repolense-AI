from fastapi import APIRouter, Depends, HTTPException, status
from uuid import UUID
import structlog

from api.schemas.review import (
    ReviewCreate,
    ReviewResponse,
    ReviewListResponse,
    ReviewStatusResponse,
)
from api.middleware.auth import get_current_user
from core.models.user import User
from core.database import get_db
from core.services.review_service import (
    create_review,
    get_review,
    list_reviews,
)

log = structlog.get_logger()

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.post("", response_model=ReviewResponse)
async def create_review_endpoint(
    data: ReviewCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Start or reuse deterministic analysis for a verified ingested revision.
    Profile-only legacy review requests are unsupported (422).
    """
    try:
        # Create review with status="pending"
        review = await create_review(
            db=db,
            profile_id=data.profile_id,
            user_id=current_user.id,
            revision_id=data.revision_id,
        )


        log.info(
            "review_created",
            review_id=str(review.id),
            profile_id=str(data.profile_id),
            user_id=str(current_user.id),
        )

        if not review.analysis_result:
            raise HTTPException(
                410,
                "Historical placeholder review is unsupported; analyze a verified repository ingestion",
            )
        return ReviewResponse.model_validate(review)

    except HTTPException:
        raise
    except Exception as exc:
        log.error("review_creation_error", error=str(exc))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create review",
        )


@router.get("/{review_id}", response_model=ReviewResponse)
async def get_review_endpoint(
    review_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Get a review by ID.
    Returns 404 if not found or not owned by current user.
    """
    try:
        review = await get_review(db=db, review_id=review_id, user_id=current_user.id)

        if not review:
            log.warning(
                "review_not_found",
                review_id=str(review_id),
                user_id=str(current_user.id),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review not found",
            )

        if not review.analysis_result:
            raise HTTPException(
                410,
                "Historical placeholder review is unsupported; analyze a verified repository ingestion",
            )
        return ReviewResponse.model_validate(review)

    except HTTPException:
        raise
    except Exception as exc:
        log.error("get_review_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve review",
        )


@router.get("", response_model=ReviewListResponse)
async def list_reviews_endpoint(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    List reviews for current user with pagination.
    """
    try:
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 20

        reviews, total = await list_reviews(
            db=db,
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            supported_only=True,
        )

        return ReviewListResponse(
            items=[ReviewResponse.model_validate(r) for r in reviews if r.analysis_result],
            total=total,
            page=page,
            page_size=page_size,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("list_reviews_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list reviews",
        )


@router.get("/{review_id}/status", response_model=ReviewStatusResponse)
async def get_review_status(
    review_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Get supported deterministic analysis status; expired leases become failed.
    Historical placeholder jobs are unsupported (410).
    """
    try:
        review = await get_review(db=db, review_id=review_id, user_id=current_user.id)

        if not review:
            log.warning(
                "review_not_found",
                review_id=str(review_id),
                user_id=str(current_user.id),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review not found",
            )

        if not review.analysis_result:
            raise HTTPException(
                410,
                "Historical placeholder review is unsupported; analyze a verified repository ingestion",
            )
        from core.services.repository_analysis import get_job

        review = await get_job(db, review.id, current_user.id)
        return {
            "review_id": str(review.id),
            "status": review.status,
            "progress_pct": review.progress_pct,
            "progress_stage": review.progress_stage,
            "error_message": review.error_message,
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.error("get_review_status_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve review status",
        )
