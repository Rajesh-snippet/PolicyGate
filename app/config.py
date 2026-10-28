from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    app_name: str = "PolicyGate"

    # LLM judge (Day 6). Pick a provider and set its API key in .env.
    judge_provider: Literal["groq", "anthropic", "openai"] = "groq"
    judge_model: str | None = None  # None -> provider default
    judge_timeout_s: float = 20.0
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None

    # Decision engine and audit log (Day 7)
    low_confidence_threshold: float = 0.6
    audit_log_path: str = "data/audit_log.jsonl"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()