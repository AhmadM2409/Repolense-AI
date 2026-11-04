"""Actual multipart/JSON routes, parser and persisted fields."""

from io import BytesIO
from uuid import UUID
import pytest
from sqlalchemy import select, func
from pypdf import PdfWriter
from core.models import Profile
from core.security import create_access_token

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def auth(actors):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][0].id})}


@pytest.mark.parametrize("field", ["resume_file", "resume"])
async def test_upload_field_alias_persists_text(client, database, actors, field):
    response = await client.post(
        "/profiles",
        headers=auth(actors),
        data={"github_username": "example"},
        files={field: ("resume.md", b"# Experience\nActual submitted text", "text/markdown")},
    )
    assert response.status_code == 200, response.text
    async with database() as db:
        row = await db.get(Profile, str(UUID(response.json()["id"])))
        assert row.resume_text == "# Experience\nActual submitted text"
        assert row.resume_filename == "resume.md"


def text_pdf():
    # A minimal synthetic PDF with a genuine text stream, parsed by real pypdf.
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 100 Td (Synthetic resume text) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def test_pdf_upload_uses_real_parser(client, database, actors):
    response = await client.post(
        "/profiles",
        headers=auth(actors),
        files={"resume_file": ("resume.pdf", text_pdf(), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    async with database() as db:
        row = await db.get(Profile, str(UUID(response.json()["id"])))
        assert "Synthetic resume text" in row.resume_text


@pytest.mark.parametrize(
    "name,body,mime,code",
    [
        ("bad.pdf", b"not a pdf", "application/pdf", 422),
        ("bad.md", b"\xff", "text/markdown", 422),
        ("empty.md", b"  ", "text/markdown", 422),
        ("bad.exe", b"binary", "application/octet-stream", 422),
        ("large.md", b"x" * (5 * 1024 * 1024 + 1), "text/markdown", 413),
    ],
    ids=["bad-pdf", "bad-utf8", "empty", "unsupported", "oversize"],
)
async def test_invalid_upload_has_no_partial_profile(
    client, database, actors, name, body, mime, code
):
    async with database() as db:
        before = await db.scalar(select(func.count()).select_from(Profile))
    response = await client.post(
        "/profiles", headers=auth(actors), files={"resume_file": (name, body, mime)}
    )
    assert response.status_code == code, response.text
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(Profile)) == before


async def test_upload_conflict_and_form_limits(client, actors):
    response = await client.post(
        "/profiles",
        headers=auth(actors),
        files={
            "resume_file": ("a.md", b"a", "text/markdown"),
            "resume": ("b.md", b"b", "text/markdown"),
        },
    )
    assert response.status_code == 422
    response = await client.post(
        "/profiles", headers=auth(actors), data={"github_username": "x" * 256}
    )
    assert response.status_code == 422


async def test_status_is_separate_typed_contract(client, actors):
    response = await client.get(
        "/reviews/" + actors["reviews"][0].id + "/status", headers=auth(actors)
    )
    assert response.status_code==410
    assert "unsupported" in response.json()["detail"]


async def test_nullable_profile_update_distinguishes_omitted_from_null(client, database, actors):
    pid = actors["profiles"][0].id
    response = await client.put(
        "/profiles/" + pid,
        headers=auth(actors),
        json={"github_username": "example", "portfolio_url": "https://example.invalid"},
    )
    assert response.status_code == 200
    response = await client.put(
        "/profiles/" + pid, headers=auth(actors), json={"github_username": None}
    )
    assert response.status_code == 200
    assert response.json()["github_username"] is None
    assert response.json()["portfolio_url"] == "https://example.invalid"


async def test_revision_request_and_response(client, database, actors):
    from uuid import uuid4
    from core.contracts import RepositoryContract, RevisionContract
    from core.services.source_service import register_revision

    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    repo = RepositoryContract(id=uuid4(), profile_id=pid, full_name="example/project")
    rev = RevisionContract(id=uuid4(), repository_id=repo.id, profile_id=pid, commit_sha="a" * 40)
    async with database() as db:
        await register_revision(db, uid, repo, rev)
        from tests.analysis_fixtures import verified_ingestion
        source=await verified_ingestion(db,pid,uid)
        revision_id=source.revision_id
    response = await client.post(
        "/reviews", headers=auth(actors), json={"profile_id": pid, "revision_id": revision_id}
    )
    assert response.status_code == 200, response.text
    assert response.json()["revision_id"] == revision_id
    assert response.json()["status"] == "pending"
    response = await client.get("/reviews/" + response.json()["id"], headers=auth(actors))
    assert response.json()["report_kind"] == "evidence"
    assert response.json()["revision_id"] == revision_id
    response = await client.post(
        "/reviews", headers=auth(actors), json={"profile_id": pid, "revision_id": "main"}
    )
    assert response.status_code == 422
