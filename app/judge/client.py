"""Provider-agnostic client for the LLM judge.

The judge only needs one capability: send a system + user prompt, get text
back. Adapters import their SDK lazily so you only install the one you use,
and tests use a fake client with no network.
"""
from typing import Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class JudgeConfigError(Exception):
    """Raised when the judge provider is misconfigured."""


class JudgeClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicJudgeClient:
    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL, timeout: float = 20.0):
        try:
            import anthropic
        except ImportError as exc:
            raise JudgeConfigError("pip install anthropic to use the anthropic judge") from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIJudgeClient:
    """Chat-completions client. Also serves any OpenAI-compatible API via base_url."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        timeout: float = 20.0,
        base_url: str | None = None,
    ):
        try:
            import openai
        except ImportError as exc:
            raise JudgeConfigError("pip install openai to use the openai/groq judge") from exc
        # max_retries lets the SDK back off and retry on rate limits (429), which
        # matters on free tiers.
        self._client = openai.OpenAI(
            api_key=api_key, timeout=timeout, base_url=base_url, max_retries=3
        )
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class GroqJudgeClient(OpenAIJudgeClient):
    """Groq's API is OpenAI-compatible, so it reuses the OpenAI SDK with Groq's base URL."""

    def __init__(self, api_key: str, model: str = DEFAULT_GROQ_MODEL, timeout: float = 20.0):
        super().__init__(api_key, model, timeout, base_url=GROQ_BASE_URL)


class UnavailableJudgeClient:
    """Placeholder used when no judge is configured; every call fails."""

    def __init__(self, reason: str):
        self.reason = reason

    def complete(self, system: str, user: str) -> str:
        raise JudgeConfigError(self.reason)


def build_judge_client(settings) -> JudgeClient:
    provider = settings.judge_provider
    if provider == "groq":
        if not settings.groq_api_key:
            raise JudgeConfigError("GROQ_API_KEY is not set")
        return GroqJudgeClient(
            settings.groq_api_key,
            settings.judge_model or DEFAULT_GROQ_MODEL,
            settings.judge_timeout_s,
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise JudgeConfigError("ANTHROPIC_API_KEY is not set")
        return AnthropicJudgeClient(
            settings.anthropic_api_key,
            settings.judge_model or DEFAULT_ANTHROPIC_MODEL,
            settings.judge_timeout_s,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise JudgeConfigError("OPENAI_API_KEY is not set")
        return OpenAIJudgeClient(
            settings.openai_api_key,
            settings.judge_model or DEFAULT_OPENAI_MODEL,
            settings.judge_timeout_s,
        )
    raise JudgeConfigError(f"unknown judge provider: {provider}")