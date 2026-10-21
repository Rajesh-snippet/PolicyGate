from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    app_name: str = "PolicyGate"

    # LLM judge (Day 6). Pick a provider and set its API key in .env.
    judge_provider: Literal["anthropic", "openai"] = "anthropic"
    judge_model: str | None = None  # None -> provider default
    judge_timeout_s: float = 20.0
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
