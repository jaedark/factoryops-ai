from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import (
    AppSettings,
    ConfigurationError,
    get_settings,
)
from backend.app.agents import INCIDENT_ANALYSIS_AGENT
from backend.app.core.database import SessionLocal
from backend.app.resilience import (
    AgentExecutionConfig,
    CircuitBreaker,
    ResilientExecutionService,
    RetryPolicy,
)
from backend.app.services.llm_service import LlmService
from backend.app.services.agent_service import AgentService
from backend.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app, headers={"X-API-Key": "test-api-key"})


@pytest.fixture(autouse=True)
def reset_cached_configuration():
    get_settings.cache_clear()
    LlmService.reset_client()
    ResilientExecutionService.reset_default_llm_circuit_breaker()
    yield
    get_settings.cache_clear()
    LlmService.reset_client()
    ResilientExecutionService.reset_default_llm_circuit_breaker()


def test_settings_load_safe_defaults_without_secret():
    settings = AppSettings()

    assert settings.factory_agent_api_key is None
    assert settings.api_max_request_bytes == 1_048_576
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.database_url == "sqlite:///./factoryops.db"
    assert settings.agent_max_steps == 5


def test_settings_representation_masks_gemini_secret():
    settings = AppSettings(
        factory_agent_api_key="factory-api-secret",
        gemini_api_key="do-not-log-this-secret",
        database_url="postgresql://user:password@db/app",
    )

    assert "do-not-log-this-secret" not in repr(settings)
    assert "do-not-log-this-secret" not in settings.model_dump_json()
    assert "factory-api-secret" not in repr(settings)
    assert "factory-api-secret" not in settings.model_dump_json()
    assert "postgresql://user:password@db/app" not in repr(settings)


def test_settings_environment_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("FACTORY_AGENT_PORT", "8100")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("FACTORY_AGENT_API_KEY", "configured-api-key")
    monkeypatch.setenv("API_MAX_REQUEST_BYTES", "2097152")

    settings = get_settings()

    assert settings.app_env == "test"
    assert settings.factory_agent_port == 8100
    assert settings.gemini_model == "gemini-test-model"
    assert settings.get_factory_agent_api_key() == "configured-api-key"
    assert settings.api_max_request_bytes == 2_097_152


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("TOOL_TIMEOUT_SECONDS", "-1"),
        ("RETRY_MAX_ATTEMPTS", "0"),
        ("CIRCUIT_FAILURE_THRESHOLD", "0"),
        ("CIRCUIT_RECOVERY_TIMEOUT_SECONDS", "0"),
        ("RETRIEVAL_TOP_K", "0"),
        ("API_MAX_REQUEST_BYTES", "100"),
    ],
)
def test_invalid_runtime_settings_are_rejected(
    monkeypatch,
    environment_name,
    value,
):
    monkeypatch.setenv(environment_name, value)

    with pytest.raises(ValidationError):
        get_settings()


def test_blank_database_url_is_rejected(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", " ")

    with pytest.raises(ValidationError):
        get_settings()


def test_missing_gemini_key_does_not_break_health(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_gemini_key_fails_before_llm_client_creation(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(ConfigurationError, match="Gemini API key is not configured"):
        LlmService.generate_content("hello")


def test_llm_service_uses_configured_model(monkeypatch):
    calls = []
    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: calls.append(kwargs) or "response"
        )
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-configured")
    LlmService._client = fake_client
    LlmService._client_api_key = "test-key"

    assert LlmService.generate_content("hello") == "response"
    assert calls[0]["model"] == "gemini-configured"


def test_database_url_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./configured.db")

    assert get_settings().database_url == "sqlite:///./configured.db"


def test_agent_execution_config_uses_settings(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "8")

    config = AgentExecutionConfig.from_settings(get_settings())

    assert config.max_steps == 7
    assert config.llm_timeout_seconds == 20
    assert config.tool_timeout_seconds == 8


def test_explicit_max_steps_overrides_settings(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")

    config = AgentExecutionConfig.from_settings(
        get_settings(),
        max_steps=2,
    )

    assert config.max_steps == 2


def test_agent_service_runtime_uses_settings_and_explicit_override(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "21")
    db = SessionLocal()
    response = SimpleNamespace(function_calls=None, candidates=[], text="done")

    try:
        with patch.object(
            ResilientExecutionService,
            "call_llm",
            return_value=response,
        ) as call_llm:
            AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="status",
                max_steps=2,
            )

        runtime = call_llm.call_args.kwargs["runtime"]
        assert runtime.config.max_steps == 2
        assert runtime.config.llm_timeout_seconds == 21
    finally:
        db.close()


def test_retry_policy_uses_settings(monkeypatch):
    monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("RETRY_BASE_DELAY_SECONDS", "0.2")
    monkeypatch.setenv("RETRY_MAX_DELAY_SECONDS", "1.0")
    monkeypatch.setenv("RETRY_BACKOFF_MULTIPLIER", "3")

    policy = RetryPolicy.from_settings(get_settings())

    assert policy.max_attempts == 4
    assert policy.base_delay_seconds == 0.2
    assert policy.max_delay_seconds == 1.0
    assert policy.backoff_multiplier == 3


def test_default_circuit_breaker_uses_settings(monkeypatch):
    monkeypatch.setenv("CIRCUIT_FAILURE_THRESHOLD", "4")
    monkeypatch.setenv("CIRCUIT_RECOVERY_TIMEOUT_SECONDS", "45")

    breaker = CircuitBreaker.from_settings(get_settings())

    assert breaker.failure_threshold == 4
    assert breaker.recovery_timeout_seconds == 45


def test_settings_cache_can_be_cleared(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "first-model")
    first = get_settings()
    monkeypatch.setenv("GEMINI_MODEL", "second-model")

    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().gemini_model == "second-model"


def test_env_example_contains_placeholders_not_secret_values():
    values = dict(
        line.split("=", 1)
        for line in (PROJECT_ROOT / ".env.example").read_text(
            encoding="utf-8"
        ).splitlines()
        if line and not line.startswith("#")
    )

    assert values["GEMINI_API_KEY"] == ""
    assert values["FACTORY_AGENT_API_KEY"] == ""
    assert values["API_MAX_REQUEST_BYTES"] == "1048576"
    assert values["DATABASE_URL"] == "sqlite:///./factoryops.db"


def test_git_and_docker_ignore_protect_local_env_files():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert ".env.*" in gitignore
    assert "!.env.example" in gitignore
    assert ".env" in dockerignore
    assert ".env.*" in dockerignore


def test_compose_environment_names_match_settings():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    )
    environment = compose["services"]["factory-agent"]["environment"]

    expected_names = {
        "GEMINI_API_KEY",
        "FACTORY_AGENT_API_KEY",
        "API_MAX_REQUEST_BYTES",
        "GEMINI_MODEL",
        "DATABASE_URL",
        "AGENT_MAX_STEPS",
        "LLM_TIMEOUT_SECONDS",
        "TOOL_TIMEOUT_SECONDS",
        "RETRY_MAX_ATTEMPTS",
        "CIRCUIT_FAILURE_THRESHOLD",
        "RERANKER_MODEL",
    }
    assert expected_names <= environment.keys()


def test_container_runner_uses_host_and_port_settings(monkeypatch):
    from backend import run

    settings = AppSettings(
        factory_agent_host="127.0.0.1",
        factory_agent_port=8123,
    )
    monkeypatch.setattr(run, "get_settings", lambda: settings)

    with patch.object(run.uvicorn, "run") as run_server:
        run.main()

    run_server.assert_called_once_with(
        "backend.main:app",
        host="127.0.0.1",
        port=8123,
        workers=1,
    )
