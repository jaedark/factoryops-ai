import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class ConfigurationError(ValueError):
    """필수 runtime 설정이 없을 때 발생하는 오류."""


class AppSettings(BaseModel):
    app_env: str = Field(default="development", min_length=1)
    factory_agent_host: str = Field(default="0.0.0.0", min_length=1)
    factory_agent_port: int = Field(default=8000, ge=1, le=65535)
    factory_agent_api_key: SecretStr | None = None
    api_max_request_bytes: int = Field(default=1_048_576, ge=1_024)

    gemini_api_key: SecretStr | None = None
    gemini_model: str = Field(default="gemini-2.5-flash", min_length=1)
    database_url: str = Field(
        default="sqlite:///./factoryops.db",
        min_length=1,
        repr=False,
    )

    agent_max_steps: int = Field(default=5, ge=1, le=10)
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0)
    tool_timeout_seconds: float = Field(default=5.0, gt=0.0)
    retry_max_attempts: int = Field(default=3, ge=1)
    retry_base_delay_seconds: float = Field(default=0.1, ge=0.0)
    retry_max_delay_seconds: float = Field(default=0.5, ge=0.0)
    retry_backoff_multiplier: float = Field(default=2.0, ge=1.0)
    circuit_failure_threshold: int = Field(default=3, ge=1)
    circuit_recovery_timeout_seconds: float = Field(default=30.0, gt=0.0)

    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        min_length=1,
    )
    reranker_model: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        min_length=1,
    )
    reranker_top_n: int = Field(default=5, ge=1)
    retrieval_top_k: int = Field(default=3, ge=1)

    @field_validator("app_env", "factory_agent_host", "gemini_model", "database_url")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("configuration value must not be blank")
        return value

    @model_validator(mode="after")
    def validate_retry_delays(self):
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError(
                "RETRY_MAX_DELAY_SECONDS must be greater than or equal to "
                "RETRY_BASE_DELAY_SECONDS"
            )
        return self

    @classmethod
    def from_environment(cls) -> "AppSettings":
        load_dotenv(override=False)
        environment_names = {
            "app_env": ("APP_ENV",),
            "factory_agent_host": ("FACTORY_AGENT_HOST",),
            "factory_agent_port": ("FACTORY_AGENT_PORT",),
            "factory_agent_api_key": ("FACTORY_AGENT_API_KEY",),
            "api_max_request_bytes": ("API_MAX_REQUEST_BYTES",),
            "gemini_api_key": ("GEMINI_API_KEY",),
            "gemini_model": ("GEMINI_MODEL",),
            "database_url": ("DATABASE_URL",),
            "agent_max_steps": ("AGENT_MAX_STEPS",),
            "llm_timeout_seconds": ("LLM_TIMEOUT_SECONDS",),
            "tool_timeout_seconds": ("TOOL_TIMEOUT_SECONDS",),
            "retry_max_attempts": ("RETRY_MAX_ATTEMPTS",),
            "retry_base_delay_seconds": ("RETRY_BASE_DELAY_SECONDS",),
            "retry_max_delay_seconds": ("RETRY_MAX_DELAY_SECONDS",),
            "retry_backoff_multiplier": ("RETRY_BACKOFF_MULTIPLIER",),
            "circuit_failure_threshold": ("CIRCUIT_FAILURE_THRESHOLD",),
            "circuit_recovery_timeout_seconds": (
                "CIRCUIT_RECOVERY_TIMEOUT_SECONDS",
            ),
            "embedding_model": ("EMBEDDING_MODEL",),
            "reranker_model": ("RERANKER_MODEL", "RERANKER_MODEL_NAME"),
            "reranker_top_n": ("RERANKER_TOP_N", "RERANK_RETRIEVER_TOP_N"),
            "retrieval_top_k": ("RETRIEVAL_TOP_K", "RERANK_FINAL_TOP_K"),
        }
        values = {}
        for field_name, aliases in environment_names.items():
            for environment_name in aliases:
                if environment_name in os.environ:
                    values[field_name] = os.environ[environment_name]
                    break

        for secret_name in ("gemini_api_key", "factory_agent_api_key"):
            if values.get(secret_name) == "":
                values[secret_name] = None
        return cls.model_validate(values)

    def get_factory_agent_api_key(self) -> str | None:
        if self.factory_agent_api_key is None:
            return None
        value = self.factory_agent_api_key.get_secret_value().strip()
        return value or None

    def require_gemini_api_key(self) -> str:
        if self.gemini_api_key is None:
            raise ConfigurationError("Gemini API key is not configured")
        value = self.gemini_api_key.get_secret_value().strip()
        if not value:
            raise ConfigurationError("Gemini API key is not configured")
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings.from_environment()


def validate_startup_settings() -> AppSettings:
    return get_settings()
