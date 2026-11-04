"""Application configuration loaded from environment variables and .env file."""

import secrets
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with support for .env file and environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5435/repolens_dev"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    vector_db_url: str = Field(default="http://localhost:8001")

    # LLM Configuration
    llm_provider: str = Field(default="mock")
    openai_api_key: str = Field(default="")
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="google/gemma-3-27b-it:free")

    # Approved local generation; distinct from embeddings and legacy review utilities.
    generation_protocol: Literal["openai", "lmstudio-native"] = "lmstudio-native"
    generation_base_url: str = "http://localhost:1235/v1"
    generation_model: str = "qwen/qwen3.5-9b"
    generation_timeout_seconds: float = Field(default=90, ge=0.1, le=120)
    generation_max_input_bytes: int = Field(default=4096, ge=256, le=4096)
    generation_max_output_tokens: int = Field(default=512, ge=16, le=1024)
    generation_max_response_bytes: int = Field(default=65536, ge=1024, le=65536)
    generation_concurrency: int = Field(default=1, ge=1, le=2)
    generation_temperature: float = Field(default=0, ge=0, le=1)
    generation_local_api_key: SecretStr = SecretStr("")
    qa_timeout_seconds: float = Field(default=180, ge=0.1, le=240)

    # Hosted Q&A remains explicit opt-in until independently accepted.
    investigation_planner_thinking: Literal["minimal", "low"] = "minimal"
    durable_investigation_enabled: bool = False
    investigation_enabled: bool = False
    investigation_ledger_path: str = ".test-tmp/phase23-gemini-ledger.sqlite3"
    investigation_budget_stage: Literal["development", "acceptance"] = "development"

    qa_provider: Literal["lmstudio", "gemini"] = "lmstudio"
    qa_gemini_enabled: bool = False
    qa_gemini_primary_accepted: bool = False
    qa_gemini_ledger_path: str = ".test-tmp/gemini-production-ledger.sqlite3"
    qa_gemini_budget_stage: Literal["development", "acceptance"] = "development"

    # Application
    app_env: str = Field(default="development")
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    log_level: str = Field(default="INFO")

    # Durable workers: retries include the first attempt.
    worker_concurrency: int = Field(default=1, ge=1, le=8)
    worker_poll_seconds: float = Field(default=1, ge=0.05, le=30)
    job_lease_seconds: int = Field(default=60, ge=5, le=3600)
    job_heartbeat_seconds: float = Field(default=10, ge=0.1, le=300)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_retry_base_seconds: float = Field(default=5, ge=0, le=3600)
    job_retry_max_seconds: float = Field(default=300, ge=0, le=86400)

    @model_validator(mode="after")
    def validate_worker_timing(self):
        if self.job_heartbeat_seconds >= self.job_lease_seconds / 2:
            raise ValueError("JOB_HEARTBEAT_SECONDS must be less than half JOB_LEASE_SECONDS")
        if self.job_retry_max_seconds < self.job_retry_base_seconds:
            raise ValueError("JOB_RETRY_MAX_SECONDS must be >= JOB_RETRY_BASE_SECONDS")
        return self

    # GitHub
    github_token: str = Field(default="")

    # JWT
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)

    career_j2_enabled: bool = False
    career_max_selected_repositories: int = Field(default=5, ge=1, le=10)
    career_discovery_pages: int = Field(default=2, ge=1, le=4)

    # Feature Limits
    max_repos_per_profile: int = Field(default=5)
    max_chunks_per_query: int = Field(default=10)
    min_relevance_score: float = Field(default=0.3)

    # Keyword-only retrieval limits. No vector/provider infrastructure.
    keyword_search_max_query_characters: int = Field(default=1024, ge=1, le=1024)
    keyword_search_max_top_k: int = Field(default=20, ge=5, le=20)
    keyword_search_max_sources: int = Field(default=100, ge=1, le=100)
    keyword_search_max_chunks: int = Field(default=2000, ge=1, le=10000)
    keyword_search_max_corpus_bytes: int = Field(default=4194304, ge=1, le=4194304)
    keyword_search_max_metadata_bytes: int = Field(default=524288, ge=1, le=524288)
    keyword_search_max_tokens: int = Field(default=100000, ge=1, le=100000)
    keyword_search_max_excerpt_bytes: int = Field(default=16384, ge=1, le=16384)
    keyword_search_max_response_bytes: int = Field(default=131072, ge=1, le=131072)
    keyword_search_concurrency: int = Field(default=2, ge=1, le=8)
    keyword_search_timeout_seconds: float = Field(default=5, ge=0.01, le=10)

    # Approved embedding configurations: hosted transmission is disabled by default.
    embedding_provider: Literal["lmstudio", "mistral"] = "lmstudio"
    embedding_lmstudio_base_url: str = "http://localhost:1235/v1"
    embedding_lmstudio_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_lmstudio_dimensions: int = 768
    embedding_lmstudio_quantization: str = "Q4_K_M"
    embedding_lmstudio_version: str = Field(
        default="nomic-v1.5-q4_k_m-v1", min_length=1, max_length=64
    )
    embedding_lmstudio_api_key: SecretStr = SecretStr("")
    embedding_lmstudio_max_requests: int = Field(default=512, ge=1, le=512)
    embedding_mistral_base_url: str = "https://api.mistral.ai/v1"
    embedding_mistral_model: str = "mistral-embed"
    embedding_mistral_dimensions: int = 1024
    embedding_mistral_version: str = Field(
        default="mistral-embed-2312-v1", min_length=1, max_length=64
    )
    embedding_mistral_api_key: SecretStr = SecretStr("")
    embedding_mistral_max_requests: int = Field(default=0, ge=0, le=512)
    embedding_hosted_source_allowed: bool = False
    embedding_batch_size: int = Field(default=8, ge=1, le=32)
    embedding_max_input_tokens: int = Field(default=2048, ge=32, le=2048)
    embedding_max_batch_tokens: int = Field(default=8192, ge=32, le=8192)
    embedding_max_chunks: int = Field(default=2000, ge=1, le=2000)
    embedding_max_index_tokens: int = Field(default=500000, ge=32, le=4000000)
    embedding_timeout_seconds: float = Field(default=30, ge=0.1, le=120)
    embedding_retries: int = Field(default=2, ge=0, le=3)

    # Semantic reads: independent hosted QUERY consent, never inferred from source consent.
    hybrid_search_timeout_seconds: float = Field(default=55, ge=0.1, le=120)
    semantic_search_hosted_query_allowed: bool = False
    semantic_search_concurrency: int = Field(default=2, ge=1, le=8)
    semantic_search_timeout_seconds: float = Field(default=45, ge=0.1, le=120)
    semantic_search_max_provider_requests: int = Field(default=3, ge=1, le=4)

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)

    @model_validator(mode="after")
    def validate_signing_secret(self) -> "Settings":
        """Non-development deployments must explicitly supply a non-placeholder key."""
        self.app_env = self.app_env.strip().lower()
        if self.app_env not in {"development", "test", "testing"} and (
            "secret_key" not in self.model_fields_set
            or len(self.secret_key.encode("utf-8")) < 32
            or any(
                marker in self.secret_key.lower()
                for marker in ("change_me", "change-me", "your-secret", "dev-secret")
            )
        ):
            raise ValueError(
                "Non-development environments require an explicit SECRET_KEY of at least "
                "32 bytes without development/example placeholders"
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


# Singleton instance
settings = Settings()
