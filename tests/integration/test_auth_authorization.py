"""Two-user HTTP security tests with real ORM persistence and JWT/password code."""

from datetime import timedelta
from uuid import uuid4

import pytest
from jose import jwt
from sqlalchemy import func, select, update

from core.config import settings
from core.models import Profile, Review, User
from core.security import create_access_token
from core.services import review_service
from tests.auth_fixtures import PASSWORD

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def bearer(uid):
    return {"Authorization": "Bearer " + create_access_token({"sub": uid})}


def endpoints(actors, owner=0):
    pid = actors["profiles"][owner].id
    rid = actors["reviews"][owner].id
    return [
        ("POST", "/profiles", {"data": {"github_username": "alice"}}),
        ("GET", "/profiles/" + pid, {}),
        ("PUT", "/profiles/" + pid, {"json": {"github_username": "updated"}}),
        ("DELETE", "/profiles/" + pid, {}),
        ("POST", "/reviews", {"json": {"profile_id": pid}}),
        ("GET", "/reviews/" + rid, {}),
        ("GET", "/reviews/" + rid + "/status", {}),
        ("GET", "/reviews", {}),
    ]


async def test_authorized_user_can_create_read_update_delete(client, database, actors):
    headers = bearer(actors["users"][0].id)
    response = await client.post("/profiles", data={"github_username": "owned"}, headers=headers)
    assert response.status_code == 200
    profile = response.json()
    assert profile["user_id"] == actors["users"][0].id
    pid = profile["id"]
    assert (await client.get("/profiles/" + pid, headers=headers)).status_code == 200
    assert (await client.put(
        "/profiles/" + pid, json={"github_username": "updated"}, headers=headers,
    )).json()["github_username"] == "updated"
    from tests.analysis_fixtures import verified_ingestion
    async with database() as db:
        source = await verified_ingestion(db,pid,actors["users"][0].id)
    response = await client.post("/reviews", json={"profile_id": pid,"revision_id":source.revision_id}, headers=headers)
    assert response.status_code == 200 and response.json()["status"] == "pending"
    rid = response.json()["id"]
    # Independent worker executes the committed job after the request has returned.
    from workers.run import tick
    await tick(database,"auth-regression-worker")
    assert (await client.get("/reviews/" + rid, headers=headers)).json()["status"] == "complete"
    assert (await client.get("/reviews/" + rid + "/status", headers=headers)).status_code == 200
    listed = (await client.get("/reviews", headers=headers)).json()
    assert listed["total"] == 1
    assert all(row["profile_id"] != actors["profiles"][1].id for row in listed["items"])
    assert (await client.delete("/profiles/" + pid, headers=headers)).status_code == 204
    assert (await client.get("/reviews/" + rid, headers=headers)).status_code == 404
    async with database() as db:
        assert await db.get(Review, actors["reviews"][1].id) is not None


@pytest.mark.parametrize("index", range(6))
async def test_cross_user_resource_requests_are_not_found(client, database, actors, index):
    foreign = endpoints(actors, 1)[1:7]
    method, path, kwargs = foreign[index]
    headers = bearer(actors["users"][0].id)
    response = await client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 404
    async with database() as db:
        assert (await db.get(Profile, actors["profiles"][1].id)).github_username is None
        assert (await db.get(Review, actors["reviews"][1].id)).status == "pending"
        assert await db.scalar(select(func.count()).select_from(Review)) == 2


@pytest.mark.parametrize("index", range(8))
async def test_unauthenticated_requests_are_rejected(client, actors, index):
    method, path, kwargs = endpoints(actors)[index]
    response = await client.request(method, path, **kwargs)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("index", range(8))
async def test_inactive_tokens_cannot_access_any_protected_route(client, database, actors, index):
    uid = actors["users"][0].id
    headers = bearer(uid)  # Issued while user is still active.
    async with database() as db:
        await db.execute(update(User).where(User.id == uid).values(is_active=False))
        await db.commit()
    method, path, kwargs = endpoints(actors)[index]
    response = await client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 401


async def test_missing_and_foreign_ids_have_same_error(client, actors):
    headers = bearer(actors["users"][0].id)
    foreign = await client.post("/reviews", json={"profile_id": actors["profiles"][1].id},
                                headers=headers)
    missing = await client.post("/reviews", json={"profile_id": str(uuid4())}, headers=headers)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()


async def test_profile_owner_cannot_be_supplied_or_updated_by_client(client, actors):
    headers = bearer(actors["users"][0].id)
    response = await client.post(
        "/profiles", data={"user_id": actors["users"][1].id}, headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == actors["users"][0].id
    response = await client.put(
        "/profiles/" + actors["profiles"][0].id,
        json={"user_id": actors["users"][1].id}, headers=headers,
    )
    assert response.json()["user_id"] == actors["users"][0].id


@pytest.mark.parametrize("case", ["malformed", "wrong_signature", "expired", "missing_exp",
                                 "missing_sub", "bad_uuid", "numeric_sub", "unknown_user"])
async def test_invalid_tokens_fail_without_database_type_errors(client, actors, case):
    uid = actors["users"][0].id
    if case == "malformed":
        token = "invalid.token.here"
    elif case == "wrong_signature":
        token = jwt.encode({"sub": uid, "exp": 9999999999}, "wrong-key", algorithm="HS256")
    elif case == "expired":
        token = create_access_token({"sub": uid}, expires_delta=timedelta(seconds=-1))
    elif case == "missing_exp":
        token = jwt.encode({"sub": uid}, settings.secret_key, algorithm=settings.jwt_algorithm)
    elif case == "missing_sub":
        token = create_access_token({})
    elif case == "bad_uuid":
        token = create_access_token({"sub": "not-a-uuid"})
    elif case == "numeric_sub":
        token = jwt.encode({"sub": 123, "exp": 9999999999}, settings.secret_key, algorithm="HS256")
    else:
        token = create_access_token({"sub": str(uuid4())})
    response = await client.get("/reviews", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_register_login_and_duplicate_registration(client, database):
    payload = {"email": "new@example.com", "password": PASSWORD}
    response = await client.post("/auth/register", json=payload)
    assert response.status_code == 200 and response.json()["token_type"] == "bearer"
    async with database() as db:
        user = await db.scalar(select(User).where(User.email == payload["email"]))
        assert user is not None and user.hashed_password != PASSWORD
    response = await client.post(
        "/auth/login", data={"username": payload["email"], "password": PASSWORD},
    )
    assert response.status_code == 200
    assert (await client.post("/auth/register", json=payload)).status_code == 400


@pytest.mark.parametrize("password", ["short", "a" * 73, "é" * 37])
async def test_registration_rejects_invalid_passwords(client, database, password):
    response = await client.post("/auth/register", json={
        "email": "bad@example.com", "password": password,
    })
    assert response.status_code == 422
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 0


@pytest.mark.parametrize("password", ["a" * 72, "é" * 36])
async def test_registration_accepts_bcrypt_byte_boundary(client, password):
    email = "boundary@example.com"
    assert (await client.post("/auth/register", json={
        "email": email, "password": password,
    })).status_code == 200
    assert (await client.post("/auth/login", data={
        "username": email, "password": password,
    })).status_code == 200
    assert (await client.post("/auth/login", data={
        "username": email, "password": password + "x",
    })).status_code == 401


@pytest.mark.parametrize("case", ["wrong_password", "unknown_email", "inactive", "bad_hash",
                                 "oversized_password"])
async def test_invalid_credentials_return_401(client, database, actors, case):
    email = actors["users"][0].email
    password = PASSWORD
    async with database() as db:
        if case == "inactive":
            await db.execute(update(User).where(User.id == actors["users"][0].id)
                             .values(is_active=False))
        elif case == "bad_hash":
            await db.execute(update(User).where(User.id == actors["users"][0].id)
                             .values(hashed_password="not-a-valid-hash"))
        await db.commit()
    if case == "wrong_password":
        password = "WrongPassword123"
    elif case == "unknown_email":
        email = "missing@example.com"
    elif case == "oversized_password":
        password = "x" * 1000
    response = await client.post("/auth/login", data={"username": email, "password": password})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_deactivated_queued_owner_does_not_run_job(client, database, actors):
    uid = actors["users"][0].id
    async with database() as db:
        await db.execute(update(User).where(User.id == uid).values(is_active=False))
        await db.commit()
    await review_service.run_review_job(
        actors["reviews"][0].id, actors["profiles"][0].id, uid,
    )
    async with database() as db:
        assert (await db.get(Review, actors["reviews"][0].id)).status == "pending"


async def test_invalid_registration_input_has_no_side_effect(client, database):
    response = await client.post("/auth/register", json={"email": "bad", "password": PASSWORD})
    assert response.status_code == 422
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 0


async def test_deleted_users_token_is_revoked(client, database, actors):
    headers = bearer(actors["users"][0].id)
    async with database() as db:
        user = await db.get(User, actors["users"][0].id)
        await db.delete(user)
        await db.commit()
    assert (await client.get("/reviews", headers=headers)).status_code == 401


async def test_database_constraint_registration_failure_is_safe(client, database):
    from sqlalchemy import text
    async with database() as db:
        from tests.database_probes import reject_insert
        await reject_insert(db, "reject_registration", "users", "test constraint violation")
        await db.commit()
    response = await client.post("/auth/register", json={
        "email": "new@example.com", "password": PASSWORD,
    })
    assert response.status_code == 400
    assert response.json() == {"detail": "Registration could not be completed"}
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 0


async def test_database_failure_does_not_authenticate_or_leak_details(client, database):
    from sqlalchemy import text
    async with database() as db:
        await db.execute(text("DROP TABLE users" + (" CASCADE" if db.bind.dialect.name == "postgresql" else "")))
        await db.commit()
    response = await client.get("/reviews", headers=bearer(str(uuid4())))
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "SELECT" not in response.text and "password" not in response.text


async def test_blank_password_form_is_rejected_by_oauth_contract(client):
    response = await client.post("/auth/login", data={
        "username": "person@example.com", "password": "",
    })
    assert response.status_code == 422
    assert any(error["loc"][-1] == "password" for error in response.json()["detail"])
    assert "access_token" not in response.json()


async def test_direct_login_rejects_legacy_empty_password_hash(database, actors):
    from fastapi import HTTPException
    from fastapi.security import OAuth2PasswordRequestForm
    from api.routes.auth import login
    from core.security import hash_password
    async with database() as db:
        await db.execute(update(User).where(User.id == actors["users"][0].id)
                         .values(hashed_password=hash_password("")))
        await db.commit()
        with pytest.raises(HTTPException) as denied:
            await login(
                form_data=OAuth2PasswordRequestForm(
                    username=actors["users"][0].email, password="",
                ), db=db,
            )
        assert denied.value.status_code == 401
