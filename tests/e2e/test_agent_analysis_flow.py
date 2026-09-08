from unittest.mock import patch

import pytest

from backend.app.observability import ObservabilityEventType
from backend.app.services.llm_service import LlmService
from tests.e2e.helpers import build_text_response, build_tool_response


pytestmark = pytest.mark.e2e


def test_e2e_01_current_state_and_historical_incident_analysis(
    e2e_environment,
):
    final_answer = (
        "Robot-01 현재 상태와 과거 장애를 종합하면 베어링 마모가 "
        "가능성 높은 원인이며 정비 시 베어링과 축 정렬을 점검해야 합니다."
    )
    with patch.object(
        LlmService,
        "generate_content",
        side_effect=[
            build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
            build_tool_response(
                "search_incidents",
                {
                    "query": "Robot-01 bearing vibration temperature",
                    "top_k": 2,
                },
            ),
            build_tool_response(
                "get_incident",
                {"incident_id": e2e_environment.incident_ids[0]},
            ),
            build_text_response(final_answer),
        ],
    ) as mock_generate:
        response = e2e_environment.client.post(
            "/agent/chat",
            headers={"X-Request-ID": "e2e-analysis-request"},
            json={
                "message": (
                    "Robot-01 현재 상태를 확인하고 과거 유사 장애를 "
                    "찾아서 가능성 높은 원인과 정비 방향을 알려줘."
                )
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert response.headers["X-Request-ID"] == "e2e-analysis-request"
    assert data["trace_id"].startswith("trc-")
    assert data["trace_id"] != "e2e-analysis-request"
    assert data["status"] == "completed"
    assert data["termination_reason"] == "final_answer"
    assert data["total_steps"] == 3
    assert [step["tool_called"] for step in data["steps"]] == [
        "get_equipment_status",
        "search_incidents",
        "get_incident",
    ]
    assert data["steps"][0]["tool_result"]["equipment"][
        "equipment_id"
    ] == "Robot-01"
    assert data["steps"][0]["tool_result"]["latest_telemetry"]
    assert len(data["steps"][1]["tool_result"]) == 2
    assert data["steps"][2]["tool_result"]["incident_id"] == (
        e2e_environment.incident_ids[0]
    )
    assert all(step["success"] for step in data["steps"])
    assert all(not step["approval_required"] for step in data["steps"])
    assert "create_maintenance_request" not in {
        step["tool_called"] for step in data["steps"]
    }
    for keyword in ("Robot-01", "현재 상태", "과거 장애", "원인", "정비"):
        assert keyword in data["answer"]
    assert mock_generate.call_count == 4

    events = e2e_environment.observability_sink.events
    event_types = [event.event_type for event in events]
    for expected in (
        ObservabilityEventType.AGENT_STARTED,
        ObservabilityEventType.LLM_CALL_STARTED,
        ObservabilityEventType.LLM_CALL_COMPLETED,
        ObservabilityEventType.TOOL_CALL_STARTED,
        ObservabilityEventType.TOOL_CALL_COMPLETED,
        ObservabilityEventType.AGENT_COMPLETED,
    ):
        assert expected in event_types
    assert ObservabilityEventType.RETRY_SCHEDULED not in event_types
    assert {event.trace_id for event in events} == {data["trace_id"]}


def test_e2e_10_observability_event_order(e2e_environment):
    with patch.object(
        LlmService,
        "generate_content",
        side_effect=[
            build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
            build_text_response("Robot-01 상태 확인 완료"),
        ],
    ):
        response = e2e_environment.client.post(
            "/agent/chat",
            json={"message": "Robot-01 현재 상태를 확인해줘."},
        )

    assert response.status_code == 200
    assert [event.event_type.value for event in e2e_environment.observability_sink.events] == [
        "agent_started",
        "llm_call_started",
        "llm_call_completed",
        "tool_call_started",
        "tool_call_completed",
        "llm_call_started",
        "llm_call_completed",
        "agent_completed",
    ]
