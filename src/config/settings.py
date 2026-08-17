"""Central settings loaded from environment / .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: src/config/settings.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the recommendation graph and agentic service."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    catalog_path: Path = Field(
        default=REPO_ROOT / "data" / "catalog.example.json",
        alias="CATALOG_PATH",
    )
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
