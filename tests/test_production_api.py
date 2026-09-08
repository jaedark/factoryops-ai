import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import AppSettings, get_settings
from backend.app.services.incident_service import IncidentService
from backend.app.resilience import ResilientExecutionService
from backend.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_KEY = "test-api-key"
AUTH_HEADERS = {"X-API-Key": API_KEY}
client = TestClient(app, headers=AUTH_HEADERS)
public_client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_health_and_readiness_are_public_in_test_environment():
    health = public_client.get("/health")
    ready = public_client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_protected_endpoint_requires_api_key():
    response = public_client.get("/incidents")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"


def test_invalid_api_key_is_rejected():
    response = public_client.get(
        "/incidents",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_API_KEY"
    assert "wrong-key" not in response.text


def test_valid_api_key_allows_protected_read():
    response = client.get("/incidents")

    assert response.status_code == 200


def test_admin_and_approval_endpoints_are_protected():
    admin = public_client.post("/admin/seed")
    approval = public_client.post("/agent/approvals/APR-0001/approve")

    assert admin.status_code == 401
    assert approval.status_code == 401


def test_unconfigured_api_auth_returns_service_error(monkeypatch):
    monkeypatch.setenv("FACTORY_AGENT_API_KEY", "")
    get_settings.cache_clear()

    response = public_client.get("/incidents")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "API_AUTH_NOT_CONFIGURED"


def test_request_id_is_generated_and_returned():
    response = public_client.get("/health")

    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"req-[0-9a-f-]{36}", request_id)


def test_safe_request_id_is_reused():
    response = public_client.get(
        "/health",
        headers={"X-Request-ID": "client-request-123"},
    )

    assert response.headers["X-Request-ID"] == "client-request-123"


def test_unsafe_request_id_is_replaced():
    response = public_client.get(
        "/health",
        headers={"X-Request-ID": "unsafe request id"},
    )

    assert response.headers["X-Request-ID"].startswith("req-")


def test_validation_error_uses_common_envelope_and_request_id():
    response = client.post(
        "/agent/chat",
        headers={"X-Request-ID": "validation-request"},
        json={"message": ""},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "request_id": "validation-request",
        }
    }
    assert response.headers["X-Request-ID"] == "validation-request"


def test_known_not_found_error_uses_common_envelope():
    response = client.get("/incidents/999999999")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert response.json()["error"]["message"] == "Incident not found"


def test_unknown_route_uses_common_error_envelope():
    response = public_client.get("/route-that-does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert response.json()["error"]["message"] == "Not Found"


def test_unexpected_error_is_sanitized():
    isolated_client = TestClient(
        app,
        headers=AUTH_HEADERS,
        raise_server_exceptions=False,
    )
    with patch.object(
        IncidentService,
        "get_incidents",
        side_effect=RuntimeError("database password=do-not-expose"),
    ):
        response = isolated_client.get("/incidents")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert response.json()["error"]["message"] == "Internal server error"
    assert "do-not-expose" not in response.text


def test_content_length_over_limit_is_rejected_before_endpoint():
    response = client.post(
        "/agent/chat",
        headers={"Content-Length": str(1_048_577)},
        json={"message": "small body"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_request_log_excludes_body_and_api_key(caplog):
    sensitive_message = "sensitive-production-message"
    with caplog.at_level(
        logging.INFO,
        logger="backend.app.api.middleware",
    ):
        response = client.post(
            "/admin/seed",
            json={"message": sensitive_message},
        )

    assert response.status_code == 200
    assert "API request completed" in caplog.text
    assert sensitive_message not in caplog.text
    assert API_KEY not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {"message": " "},
        {"message": "x" * 8001},
        {"message": "valid", "session_id": " "},
        {"message": "valid", "session_id": "x" * 129},
    ],
)
def test_agent_request_bounds_are_enforced(payload):
    response = client.post("/agent/chat", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_tool_and_rag_message_bounds_are_enforced():
    tool_response = client.post("/tools/chat", json={"message": " "})
    rag_response = client.post("/rag/analyze", json={"query": " "})

    assert tool_response.status_code == 422
    assert rag_response.status_code == 422


def test_approval_id_and_incident_query_bounds_are_enforced():
    approval = client.post(f"/agent/approvals/{'x' * 129}/approve")
    list_response = client.get("/incidents", params={"limit": 101})
    search = client.get(
        "/incidents/search",
        params={"keyword": "x" * 1001},
    )

    assert approval.status_code == 422
    assert list_response.status_code == 422
    assert search.status_code == 422


def test_production_readiness_requires_both_runtime_secrets(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FACTORY_AGENT_API_KEY", API_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()

    response = public_client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_NOT_READY"


def test_production_readiness_succeeds_with_runtime_secrets(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FACTORY_AGENT_API_KEY", API_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    get_settings.cache_clear()

    response = public_client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_database_readiness_failure_is_sanitized():
    with patch(
        "backend.main.engine.connect",
        side_effect=RuntimeError("database unavailable at private path"),
    ):
        response = public_client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_NOT_READY"
    assert "private path" not in response.text


def test_openapi_declares_api_key_security_only_for_protected_routes():
    schema = public_client.get("/openapi.json").json()

    scheme = schema["components"]["securitySchemes"]["FactoryAgentApiKey"]
    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-API-Key"
    assert schema["paths"]["/agent/chat"]["post"]["security"] == [
        {"FactoryAgentApiKey": []}
    ]
    assert "security" not in schema["paths"]["/health"]["get"]
    assert "security" not in schema["paths"]["/ready"]["get"]
    for path, operations in schema["paths"].items():
        if path in {"/health", "/ready"}:
            continue
        for operation in operations.values():
            if isinstance(operation, dict) and "responses" in operation:
                assert operation["security"] == [{"FactoryAgentApiKey": []}]
    assert (
        schema["paths"]["/agent/chat"]["post"]["responses"]["401"]
        ["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ApiErrorResponse"
    )


def test_openapi_docs_remain_public():
    assert public_client.get("/docs").status_code == 200
    assert public_client.get("/openapi.json").status_code == 200


def test_http_request_id_and_agent_trace_id_remain_distinct():
    llm_response = SimpleNamespace(
        function_calls=None,
        candidates=[],
        text="final answer",
    )
    with patch.object(
        ResilientExecutionService,
        "call_llm",
        return_value=llm_response,
    ):
        response = client.post(
            "/agent/chat",
            headers={"X-Request-ID": "http-request-id"},
            json={"message": "status"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "http-request-id"
    assert response.json()["trace_id"] != "http-request-id"


def test_api_settings_defaults_and_validation():
    settings = AppSettings()

    assert settings.factory_agent_api_key is None
    assert settings.api_max_request_bytes == 1_048_576
    with pytest.raises(ValidationError):
        AppSettings(api_max_request_bytes=100)


def test_api_secret_environment_and_files_are_safe(monkeypatch):
    monkeypatch.setenv("FACTORY_AGENT_API_KEY", "configured-api-secret")
    monkeypatch.setenv("API_MAX_REQUEST_BYTES", "2097152")
    settings = get_settings()

    assert settings.get_factory_agent_api_key() == "configured-api-secret"
    assert settings.api_max_request_bytes == 2_097_152
    assert "configured-api-secret" not in repr(settings)
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "FACTORY_AGENT_API_KEY=" in env_example
    assert "FACTORY_AGENT_API_KEY=configured-api-secret" not in env_example


def test_compose_and_cloud_build_inject_api_key_by_environment_reference():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    )
    cloud_build = (PROJECT_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")

    environment = compose["services"]["factory-agent"]["environment"]
    assert environment["FACTORY_AGENT_API_KEY"] == "${FACTORY_AGENT_API_KEY:-}"
    assert environment["API_MAX_REQUEST_BYTES"] == (
        "${API_MAX_REQUEST_BYTES:-1048576}"
    )
    assert (
        "FACTORY_AGENT_API_KEY=${_API_SECRET_NAME}:${_API_SECRET_VERSION}"
        in cloud_build
    )
    assert "configured-api-secret" not in cloud_build
