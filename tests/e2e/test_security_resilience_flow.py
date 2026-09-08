from unittest.mock import Mock, patch

import pytest

from backend.app.observability import ObservabilityEventType
from backend.app.services.llm_service import LlmService
from backend.app.services.tool_calling_service import ToolCallingService
from tests.e2e.helpers import build_text_response, build_tool_response


pytestmark = pytest.mark.e2e


def test_e2e_08_agent_api_authentication_boundary(e2e_environment):
    missing = e2e_environment.public_client.post(
        "/agent/chat",
        headers={"X-Request-ID": "e2e-missing-auth"},
        json={"message": "Robot-01 상태를 확인해줘."},
    )
    wrong = e2e_environment.public_client.post(
        "/agent/chat",
        headers={
            "X-API-Key": "wrong-key",
            "X-Request-ID": "e2e-wrong-auth",
        },
        json={"message": "Robot-01 상태를 확인해줘."},
    )
    with patch.object(
        LlmService,
        "generate_content",
        return_value=build_text_response("인증 후 Agent 실행 완료"),
    ):
        valid = e2e_environment.client.post(
            "/agent/chat",
            headers={"X-Request-ID": "e2e-valid-auth"},
            json={"message": "Robot-01 상태를 확인해줘."},
        )

    assert missing.status_code == 401
    assert missing.json()["error"] == {
        "code": "AUTHENTICATION_REQUIRED",
        "message": "X-API-Key header is required",
        "request_id": "e2e-missing-auth",
    }
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "INVALID_API_KEY"
    assert wrong.json()["error"]["request_id"] == "e2e-wrong-auth"
    assert valid.status_code == 200
    assert valid.headers["X-Request-ID"] == "e2e-valid-auth"


def test_e2e_09_agent_tool_allowlist_blocks_write_before_guardrail(
    e2e_environment,
    monkeypatch,
):
    execution_spy = Mock(
        wraps=ToolCallingService._TOOL_REGISTRY[
            "create_maintenance_request"
        ]
    )
    monkeypatch.setitem(
        ToolCallingService._TOOL_REGISTRY,
        "create_maintenance_request",
        execution_spy,
    )
    with patch.object(
        LlmService,
        "generate_content",
        return_value=build_tool_response(
            "create_maintenance_request",
            {
                "equipment_id": "Robot-01",
                "reason": "Unapproved write attempt",
            },
        ),
    ):
        response = e2e_environment.client.post(
            "/agent/chat",
            json={"message": "Robot-01 정비 요청을 생성해줘."},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert "Tool not allowed for agent" in response.json()["error"][
        "message"
    ]
    assert execution_spy.call_count == 0
    assert not e2e_environment.approval_store._requests
    event_types = [
        event.event_type
        for event in e2e_environment.observability_sink.events
    ]
    assert ObservabilityEventType.AGENT_FAILED in event_types
    assert ObservabilityEventType.APPROVAL_REQUIRED not in event_types
    assert ObservabilityEventType.TOOL_CALL_STARTED not in event_types


def test_e2e_11_transient_llm_failure_recovers_with_retry(
    e2e_environment,
):
    with patch.object(
        LlmService,
        "generate_content",
        side_effect=[
            TimeoutError("temporary LLM timeout"),
            build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
            build_text_response("Robot-01 상태 조회를 복구 후 완료했습니다."),
        ],
    ) as mock_generate:
        response = e2e_environment.client.post(
            "/agent/chat",
            json={"message": "Robot-01 현재 상태를 확인해줘."},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["steps"][0]["tool_called"] == (
        "get_equipment_status"
    )
    assert mock_generate.call_count == 3
    retry_events = [
        event
        for event in e2e_environment.observability_sink.events
        if event.event_type == ObservabilityEventType.RETRY_SCHEDULED
    ]
    assert len(retry_events) == 1
    assert retry_events[0].metadata["component"] == "llm"
    assert retry_events[0].metadata["error_category"] == "timeout"
    assert ObservabilityEventType.AGENT_COMPLETED in {
        event.event_type
        for event in e2e_environment.observability_sink.events
    }


def test_e2e_12_permanent_invalid_tool_fails_without_retry(
    e2e_environment,
):
    with patch.object(
        LlmService,
        "generate_content",
        return_value=build_tool_response(
            "delete_all_incidents",
            {},
        ),
    ) as mock_generate:
        response = e2e_environment.client.post(
            "/agent/chat",
            json={"message": "모든 장애를 삭제해줘."},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert "Unsupported tool requested" in response.json()["error"][
        "message"
    ]
    assert mock_generate.call_count == 1
    event_types = [
        event.event_type
        for event in e2e_environment.observability_sink.events
    ]
    assert ObservabilityEventType.RETRY_SCHEDULED not in event_types
    assert ObservabilityEventType.AGENT_FAILED in event_types
