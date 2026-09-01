from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.genai import types

from backend.main import app
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)


client = TestClient(app)


def _build_tool_response(
    tool_name: str,
    tool_args: dict,
):
    function_call = types.FunctionCall(
        name=tool_name,
        args=tool_args,
    )
    content = types.Content(
        role="model",
        parts=[
            types.Part.from_function_call(
                name=tool_name,
                args=tool_args,
            )
        ],
    )
    return SimpleNamespace(
        function_calls=[function_call],
        candidates=[SimpleNamespace(content=content)],
        text=None,
    )


def _build_text_response(text: str):
    return SimpleNamespace(
        function_calls=None,
        candidates=[],
        text=text,
    )


def test_agent_chat_runs_single_tool_then_returns_answer():
    client.post("/admin/seed")

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
            _build_text_response("Robot-01 장애 이력을 확인했습니다."),
        ],
    ) as mock_generate:
        response = client.post(
            "/agent/chat",
            json={"message": "Robot-01 장애 이력 알려줘"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["total_steps"] == 1
    assert data["status"] == "completed"
    assert data["termination_reason"] == "final_answer"
    assert data["steps"][0]["tool_called"] == "get_equipment_incidents"
    assert data["steps"][0]["tool_arguments"] == {
        "equipment_name": "Robot-01"
    }
    assert data["steps"][0]["success"] is True
    assert data["answer"] == "Robot-01 장애 이력을 확인했습니다."
    assert mock_generate.call_count == 2


def test_agent_chat_runs_two_tools_then_returns_answer():
    client.post("/admin/seed")
    incidents = client.get("/incidents").json()
    robot_incident = next(
        incident
        for incident in incidents
        if incident["equipment_name"] == "Robot-01"
    )

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
            _build_tool_response(
                "get_incident",
                {"incident_id": robot_incident["incident_id"]},
            ),
            _build_text_response("Robot-01의 원인과 조치까지 확인했습니다."),
        ],
    ):
        response = client.post(
            "/agent/chat",
            json={
                "message": "Robot-01 장애를 확인하고 상세 원인까지 알려줘"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["total_steps"] == 2
    assert data["status"] == "completed"
    assert data["termination_reason"] == "final_answer"
    assert data["steps"][0]["tool_called"] == "get_equipment_incidents"
    assert data["steps"][0]["success"] is True
    assert data["steps"][1]["tool_called"] == "get_incident"
    assert data["steps"][1]["success"] is True
    assert (
        data["steps"][1]["tool_arguments"]["incident_id"]
        == robot_incident["incident_id"]
    )
    assert data["answer"] == "Robot-01의 원인과 조치까지 확인했습니다."


def test_agent_chat_runs_industrial_then_incident_tools():
    client.post("/admin/seed")

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
            _build_tool_response(
                "search_incidents",
                {"query": "Robot-01 overheating risk", "top_k": 3},
            ),
            _build_text_response(
                "현재 상태와 유사 장애 이력을 함께 확인했습니다."
            ),
        ],
    ):
        response = client.post(
            "/agent/chat",
            json={
                "message": (
                    "Robot-01의 현재 상태를 확인하고 과거 비슷한 "
                    "장애이력을 찾아줘"
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["total_steps"] == 2
    assert [step["tool_called"] for step in data["steps"]] == [
        "get_equipment_status",
        "search_incidents",
    ]
    assert data["steps"][0]["success"] is True
    assert data["steps"][1]["success"] is True
    assert data["status"] == "completed"
    assert data["termination_reason"] == "final_answer"


def test_agent_chat_records_three_step_agentic_rag_trace():
    client.post("/admin/seed")
    incidents = client.get("/incidents").json()
    robot_incident = next(
        incident
        for incident in incidents
        if incident["equipment_name"] == "Robot-01"
    )

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
            _build_tool_response(
                "search_incidents",
                {"query": "Robot-01 servo drift overheating", "top_k": 3},
            ),
            _build_tool_response(
                "get_incident",
                {"incident_id": robot_incident["incident_id"]},
            ),
            _build_text_response(
                "현재 위험 원인과 과거 조치 방향을 종합했습니다."
            ),
        ],
    ):
        response = client.post(
            "/agent/chat",
            json={
                "message": (
                    "Robot-01의 현재 상태를 확인하고 과거 비슷한 "
                    "장애이력을 찾아서 위험 원인과 조치 방향을 알려줘"
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["total_steps"] == 3
    assert [step["tool_called"] for step in data["steps"]] == [
        "get_equipment_status",
        "search_incidents",
        "get_incident",
    ]
    assert data["steps"][0]["tool_arguments"] == {
        "equipment_id": "Robot-01"
    }
    assert data["steps"][2]["tool_arguments"]["incident_id"] == (
        robot_incident["incident_id"]
    )
    assert data["answer"] == (
        "현재 위험 원인과 과거 조치 방향을 종합했습니다."
    )


def test_agent_chat_returns_direct_answer_without_tool_call():
    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        return_value=_build_text_response("안녕하세요."),
    ):
        response = client.post(
            "/agent/chat",
            json={"message": "안녕"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "안녕하세요."
    assert data["steps"] == []
    assert data["total_steps"] == 0
    assert data["status"] == "completed"
    assert data["termination_reason"] == "final_answer"


def test_agent_chat_blocks_unsupported_tool():
    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "delete_all_incidents",
            {},
        ),
    ):
        response = client.post(
            "/agent/chat",
            json={"message": "이상한 툴을 써봐"},
        )

    assert response.status_code == 400
    assert "Unsupported tool requested" in response.json()["detail"]


def test_agent_chat_rejects_invalid_arguments():
    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "get_incident",
            {"incident_id": "wrong-type"},
        ),
    ):
        response = client.post(
            "/agent/chat",
            json={"message": "Incident 상세 알려줘"},
        )

    assert response.status_code == 400
    assert "Invalid arguments" in response.json()["detail"]


def test_agent_chat_stops_when_max_steps_exceeded():
    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
        ],
    ):
        response = client.post(
            "/agent/chat",
            json={
                "message": "계속 반복해",
                "max_steps": 2,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Agent step limit exceeded"


def test_agent_service_records_state_when_max_steps_exceeded():
    db = None

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
        ],
    ):
        try:
            from backend.app.core.database import SessionLocal

            db = SessionLocal()
            AgentService.chat(
                db=db,
                message="계속 반복해",
                max_steps=2,
            )
            assert False, "Expected AgentExecutionError"
        except AgentExecutionError as exc:
            assert exc.state.status == "failed"
            assert (
                exc.state.termination_reason
                == "max_steps_exceeded"
            )
            assert len(exc.state.steps) == 2
            assert exc.state.error == "Agent step limit exceeded"
        finally:
            if db is not None:
                db.close()


def test_agent_service_records_tool_execution_error_state():
    client.post("/admin/seed")
    db = None

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "get_equipment_incidents",
            {"equipment_name": "Robot-01"},
        ),
    ), patch(
        "backend.app.services.agent_service.ToolCallingService.execute_tool",
        side_effect=RuntimeError("tool exploded"),
    ):
        try:
            from backend.app.core.database import SessionLocal

            db = SessionLocal()
            AgentService.chat(
                db=db,
                message="Robot-01 장애 이력 알려줘",
            )
            assert False, "Expected AgentExecutionError"
        except AgentExecutionError as exc:
            assert exc.state.status == "failed"
            assert exc.state.termination_reason == "tool_error"
            assert len(exc.state.steps) == 1
            assert exc.state.steps[0].success is False
            assert exc.state.steps[0].error == "tool exploded"
        finally:
            if db is not None:
                db.close()
