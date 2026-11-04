"""Signing-key and password boundary contracts."""

import secrets

import pytest
from pydantic import ValidationError

from api.schemas.user import UserCreate
from core.config import Settings
from core.security import hash_password, verify_password


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for key in ("SECRET_KEY", "secret_key", "APP_ENV", "app_env"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("environment", ["production", " Production ", "prod", "staging", "typo"])
def test_non_development_requires_explicit_key(environment):
    with pytest.raises(ValidationError, match="explicit SECRET_KEY"):
        Settings(_env_file=None, app_env=environment)


@pytest.mark.parametrize("key", ["", "short", "dev-secret-key-change-in-production",
                                "CHANGE_ME_GENERATE_A_UNIQUE_LOCAL_SECRET"])
def test_non_development_rejects_unsafe_keys(key):
    with pytest.raises(ValidationError, match="explicit SECRET_KEY"):
        Settings(_env_file=None, app_env="production", secret_key=key)


def test_explicit_generated_production_key_is_accepted():
    key = secrets.token_urlsafe(48)
    assert Settings(_env_file=None, app_env="production", secret_key=key).secret_key == key


def test_development_default_is_ephemeral_not_public_literal():
    one = Settings(_env_file=None, app_env="development")
    two = Settings(_env_file=None, app_env="development")
    assert len(one.secret_key) >= 32 and one.secret_key != two.secret_key


def test_multibyte_password_limit_and_no_truncation_alias():
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        UserCreate(email="person@example.com", password="é" * 37)
    hashed = hash_password("a" * 72)
    assert verify_password("a" * 72, hashed)
    assert not verify_password("a" * 72 + "b", hashed)
    with pytest.raises(ValueError, match="72"):
        hash_password("a" * 73)
