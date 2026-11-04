from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from uuid import UUID
import structlog
import mimetypes
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from ingestion.parsers.resume_parser import ResumeParser

from api.schemas.profile import ProfileCreate, ProfileResponse, ProfileUpdate
from api.middleware.auth import get_current_user
from core.models.user import User
from core.models.profile import Profile
from core.database import get_db
from core.services.profile_service import (
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
)

log = structlog.get_logger()

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.post("", response_model=ProfileResponse)
async def create_profile_endpoint(
    github_username: str = Form(default=None, max_length=255),
    portfolio_url: str = Form(default=None, max_length=500),
    resume_file: UploadFile = File(default=None),
    resume: UploadFile = File(default=None),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Create a new profile with optional resume upload.
    Resume must be PDF or Markdown.
    Returns 422 if file is not PDF or markdown.
    """
    try:
        resume_filename = None
        resume_text = None

        if resume_file and resume:
            raise HTTPException(status_code=422, detail="Supply only one resume upload field")
        resume_file = resume_file or resume
        if resume_file:
            # Check file type
            file_mime = resume_file.content_type or mimetypes.guess_type(resume_file.filename or "")[0]

            if file_mime not in ["application/pdf", "text/markdown", "text/plain"]:
                log.warning(
                    "profile_creation_invalid_file_type",
                    file_type=file_mime,
                    user_id=str(current_user.id),
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Resume must be a PDF or Markdown file",
                )

            content = await resume_file.read(5 * 1024 * 1024 + 1)
            if len(content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Resume exceeds 5 MiB")
            try:
                if file_mime == "application/pdf":
                    parsed = await run_in_threadpool(ResumeParser().parse, content)
                    resume_text = parsed.text
                else:
                    resume_text = content.decode("utf-8")
            except (ValueError, UnicodeError):
                raise HTTPException(status_code=422, detail="Failed to parse resume")
            if not resume_text.strip():
                raise HTTPException(status_code=422, detail="Resume contains no extractable text")

            resume_filename = resume_file.filename

        # Create profile
        profile_data = ProfileCreate(
            github_username=github_username,
            portfolio_url=portfolio_url,
        )

        new_profile = await create_profile(
            db=db,
            user_id=current_user.id,
            data=profile_data,
            resume_filename=resume_filename,
            resume_text=resume_text,
        )

        log.info(
            "profile_created",
            profile_id=str(new_profile.id),
            user_id=str(current_user.id),
        )

        return ProfileResponse.model_validate(new_profile)

    except HTTPException:
        raise
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid profile fields")
    except Exception as exc:
        log.error("profile_creation_error", error=str(exc))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create profile",
        )


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile_endpoint(
    profile_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Get a profile by ID.
    Returns 404 if not found or not owned by current user.
    """
    try:
        profile = await get_profile(db=db, profile_id=profile_id, user_id=current_user.id)

        if not profile:
            log.warning(
                "profile_not_found",
                profile_id=str(profile_id),
                user_id=str(current_user.id),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found",
            )

        return ProfileResponse.model_validate(profile)

    except HTTPException:
        raise
    except Exception as exc:
        log.error("get_profile_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile",
        )


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile_endpoint(
    profile_id: UUID,
    data: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Update a profile.
    Returns 404 if not found or not owned by current user.
    """
    try:
        updated_profile = await update_profile(
            db=db,
            profile_id=profile_id,
            user_id=current_user.id,
            data=data,
        )

        if not updated_profile:
            log.warning(
                "profile_not_found",
                profile_id=str(profile_id),
                user_id=str(current_user.id),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found",
            )

        log.info(
            "profile_updated",
            profile_id=str(profile_id),
            user_id=str(current_user.id),
        )

        return ProfileResponse.model_validate(updated_profile)

    except HTTPException:
        raise
    except Exception as exc:
        log.error("profile_update_error", error=str(exc))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        )


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile_endpoint(
    profile_id: UUID,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Delete a profile and cascade delete reviews and ingested sources.
    Returns 404 if not found or not owned by current user.
    """
    try:
        success = await delete_profile(
            db=db,
            profile_id=profile_id,
            user_id=current_user.id,
        )

        if not success:
            log.warning(
                "profile_not_found",
                profile_id=str(profile_id),
                user_id=str(current_user.id),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found",
            )

        log.info(
            "profile_deleted",
            profile_id=str(profile_id),
            user_id=str(current_user.id),
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("profile_deletion_error", error=str(exc))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete profile",
        )
