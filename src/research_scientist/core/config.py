"""Cấu hình ứng dụng, đọc từ biến môi trường / file .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"

    # LLM
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    llm_default_model: str = "claude-opus-5-5"
    run_budget_usd: float = 2.0
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    # Persistence
    database_url: str = "postgresql+asyncpg://ars:ars@localhost:5432/ars"
    checkpoint_database_url: str = "postgresql://ars:ars@localhost:5432/ars"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "papers"
    minio_secure: bool = False

    # AI services
    grobid_url: str = "http://localhost:8070"
    tei_embed_url: str = "http://localhost:8081"
    tei_rerank_url: str = "http://localhost:8082"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # Scholarly APIs
    semantic_scholar_api_key: SecretStr | None = None
    crossref_mailto: str | None = None

    # Web
    cors_origins: list[str] = ["http://localhost:3000"]

    @staticmethod
    def _secret(value: SecretStr | None) -> str | None:
        """Trả None nếu chưa cấu hình hoặc để trống trong .env."""
        return (value.get_secret_value().strip() or None) if value is not None else None

    def anthropic_key(self) -> str | None:
        return self._secret(self.anthropic_api_key)

    def gemini_key(self) -> str | None:
        return self._secret(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
