from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    KNOWLEDGE_SEARCH_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT,
    REPORT_AGENT,
)
from backend.app.core.database import SessionLocal
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)
from backend.app.services.incident_service import IncidentService


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


def test_agent_definitions_are_configured():
    assert INCIDENT_ANALYSIS_AGENT.name == "incident_analysis"
    assert "search_incidents" in INCIDENT_ANALYSIS_AGENT.allowed_tools
    assert "get_equipment_status" in INCIDENT_ANALYSIS_AGENT.allowed_tools
    assert "get_high_risk_equipment" in INCIDENT_ANALYSIS_AGENT.allowed_tools

    assert KNOWLEDGE_SEARCH_AGENT.name == "knowledge_search"
    assert KNOWLEDGE_SEARCH_AGENT.allowed_tools == []

    assert (
        MAINTENANCE_RECOMMENDATION_AGENT.name
        == "maintenance_recommendation"
    )
    assert MAINTENANCE_RECOMMENDATION_AGENT.allowed_tools == [
        "search_incidents",
        "get_incident",
        "get_equipment_status",
        "get_equipment_telemetry",
        "get_high_risk_equipment",
    ]

    assert REPORT_AGENT.name == "report"
    assert REPORT_AGENT.allowed_tools == []


def test_incident_analysis_agent_allows_search_tool():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)

        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "search_incidents",
                    {"query": "모터 과열", "top_k": 3},
                ),
                _build_text_response("검색을 마쳤습니다."),
            ],
        ):
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="모터가 너무 뜨거워졌어",
            )

        assert result.status == "completed"
        assert result.steps[0].tool_called == "search_incidents"
    finally:
        db.close()


def test_incident_analysis_agent_allows_industrial_tool():
    db = SessionLocal()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_status",
                    {"equipment_id": "Robot-01"},
                ),
                _build_text_response("현재 상태를 확인했습니다."),
            ],
        ):
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="Robot-01 현재 상태 알려줘",
            )

        assert result.status == "completed"
        assert result.steps[0].tool_called == "get_equipment_status"
    finally:
        db.close()


def test_incident_analysis_agent_runs_multi_step_loop():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)
        robot_incidents = [
            incident
            for incident in IncidentService.get_incidents(db)
            if incident.equipment_name == "Robot-01"
        ]
        robot_incident_id = robot_incidents[0].incident_id

        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_incidents",
                    {"equipment_name": "Robot-01"},
                ),
                _build_tool_response(
                    "get_incident",
                    {"incident_id": robot_incident_id},
                ),
                _build_text_response("원인과 조치를 확인했습니다."),
            ],
        ):
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="Robot-01 장애를 확인하고 상세 원인까지 알려줘",
            )

        assert result.total_steps == 2
        assert [step.tool_called for step in result.steps] == [
            "get_equipment_incidents",
            "get_incident",
        ]
    finally:
        db.close()


def test_report_agent_blocks_industrial_tools():
    db = SessionLocal()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "get_equipment_status",
                {"equipment_id": "Robot-01"},
            ),
        ):
            try:
                AgentService.run(
                    db=db,
                    agent_definition=REPORT_AGENT,
                    message="보고서용으로 현재 상태를 찾아줘",
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError as exc:
                assert exc.state.termination_reason == "invalid_tool"
                assert exc.state.steps[0].success is False
                assert (
                    "Tool not allowed for agent"
                    in exc.state.steps[0].error
                )
    finally:
        db.close()


def test_knowledge_search_agent_can_be_created_without_tools():
    assert KNOWLEDGE_SEARCH_AGENT.allowed_tools == []
