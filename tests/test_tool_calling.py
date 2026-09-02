from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.genai import types

from backend.main import app
from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    KNOWLEDGE_SEARCH_AGENT,
    REPORT_AGENT,
)
from backend.app.services.tool_calling_service import (
    ToolCallingService,
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


def test_build_tool_schemas_contains_all_agentic_rag_tools():
    tools = ToolCallingService.build_tool_schemas()

    assert len(tools) == 1

    declarations = tools[0].function_declarations
    names = [declaration.name for declaration in declarations]

    assert names == [
        "search_incidents",
        "get_incident",
        "get_equipment_incidents",
        "get_equipment_status",
        "get_equipment_telemetry",
        "get_high_risk_equipment",
        "create_maintenance_request",
    ]


def test_build_tool_schemas_none_returns_all_tools():
    tools = ToolCallingService.build_tool_schemas(None)

    assert len(tools) == 1

    names = [
        declaration.name
        for declaration in tools[0].function_declarations
    ]

    assert names == [
        "search_incidents",
        "get_incident",
        "get_equipment_incidents",
        "get_equipment_status",
        "get_equipment_telemetry",
        "get_high_risk_equipment",
        "create_maintenance_request",
    ]


def test_build_tool_schemas_empty_returns_no_tools():
    assert ToolCallingService.build_tool_schemas([]) == []


def test_report_agent_generation_config_exposes_no_incident_tools():
    config = ToolCallingService.build_generation_config(
        system_instruction=REPORT_AGENT.system_instruction,
        allowed_tools=REPORT_AGENT.allowed_tools,
    )

    assert config.tools == []


def test_knowledge_search_agent_generation_config_exposes_no_tools():
    config = ToolCallingService.build_generation_config(
        system_instruction=KNOWLEDGE_SEARCH_AGENT.system_instruction,
        allowed_tools=KNOWLEDGE_SEARCH_AGENT.allowed_tools,
    )

    assert config.tools == []


def test_incident_analysis_agent_generation_config_exposes_incident_and_industrial_tools():
    config = ToolCallingService.build_generation_config(
        system_instruction=INCIDENT_ANALYSIS_AGENT.system_instruction,
        allowed_tools=INCIDENT_ANALYSIS_AGENT.allowed_tools,
    )

    assert len(config.tools) == 1

    names = [
        declaration.name
        for declaration in config.tools[0].function_declarations
    ]

    assert names == [
        "search_incidents",
        "get_incident",
        "get_equipment_incidents",
        "get_equipment_status",
        "get_equipment_telemetry",
        "get_high_risk_equipment",
    ]


def test_industrial_tool_schemas_are_available():
    tools = ToolCallingService.build_tool_schemas(
        [
            "get_equipment_status",
            "get_equipment_telemetry",
            "get_high_risk_equipment",
            "create_maintenance_request",
        ]
    )

    declarations = tools[0].function_declarations
    names = [declaration.name for declaration in declarations]

    assert names == [
        "get_equipment_status",
        "get_equipment_telemetry",
        "get_high_risk_equipment",
        "create_maintenance_request",
    ]

    status_schema = declarations[0].parameters_json_schema
    telemetry_schema = declarations[1].parameters_json_schema
    high_risk_schema = declarations[2].parameters_json_schema
    maintenance_schema = declarations[3].parameters_json_schema

    assert status_schema["required"] == ["equipment_id"]
    assert telemetry_schema["required"] == ["equipment_id"]
    assert high_risk_schema["properties"] == {}
    assert maintenance_schema["required"] == [
        "equipment_id",
        "reason",
    ]


def test_tool_chat_calls_get_equipment_incidents():
    client.post("/admin/seed")

    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
            _build_text_response("Robot-01 장애 이력을 정리했습니다."),
        ],
    ) as mock_generate:
        response = client.post(
            "/tools/chat",
            json={"message": "Robot-01 장애 이력 알려줘"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["tool_called"] == "get_equipment_incidents"
    assert data["tool_arguments"] == {
        "equipment_name": "Robot-01"
    }
    assert len(data["tool_result"]) >= 1
    assert data["tool_result"][0]["equipment_name"] == "Robot-01"
    assert data["answer"] == "Robot-01 장애 이력을 정리했습니다."
    assert mock_generate.call_count == 2


def test_tool_chat_calls_get_incident():
    client.post("/admin/seed")
    incidents = client.get("/incidents").json()
    incident_id = incidents[0]["incident_id"]

    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "get_incident",
                {"incident_id": incident_id},
            ),
            _build_text_response("Incident 3 상세 정보를 확인했습니다."),
        ],
    ):
        response = client.post(
            "/tools/chat",
            json={"message": "Incident 3번 상세 내용 알려줘"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["tool_called"] == "get_incident"
    assert data["tool_arguments"] == {"incident_id": incident_id}
    assert data["tool_result"]["incident_id"] == incident_id
    assert data["answer"] == "Incident 3 상세 정보를 확인했습니다."


def test_tool_chat_calls_search_incidents():
    client.post("/admin/seed")

    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        side_effect=[
            _build_tool_response(
                "search_incidents",
                {
                    "query": "모터가 너무 뜨거워졌어. 비슷한 장애 찾아줘",
                    "top_k": 3,
                },
            ),
            _build_text_response("비슷한 장애 3건을 찾았습니다."),
        ],
    ):
        response = client.post(
            "/tools/chat",
            json={
                "message": "모터가 너무 뜨거워졌어. 비슷한 장애 찾아줘"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["tool_called"] == "search_incidents"
    assert data["tool_arguments"]["top_k"] == 3
    assert len(data["tool_result"]) >= 1
    assert data["answer"] == "비슷한 장애 3건을 찾았습니다."


def test_tool_chat_blocks_unsupported_tool():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "delete_all_incidents",
            {},
        ),
    ):
        response = client.post(
            "/tools/chat",
            json={"message": "이상한 툴을 써봐"},
        )

    assert response.status_code == 400
    assert "Unsupported tool requested" in response.json()["detail"]


def test_tool_chat_rejects_invalid_arguments():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "get_incident",
            {"incident_id": "wrong-type"},
        ),
    ):
        response = client.post(
            "/tools/chat",
            json={"message": "Incident 상세 알려줘"},
        )

    assert response.status_code == 400
    assert "Invalid arguments" in response.json()["detail"]


def test_tool_chat_returns_plain_answer_without_tool_call():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_text_response(
            "안녕하세요. 무엇을 도와드릴까요?"
        ),
    ):
        response = client.post(
            "/tools/chat",
            json={"message": "안녕"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "안녕하세요. 무엇을 도와드릴까요?"
    assert data["tool_called"] is None
    assert data["tool_arguments"] is None
    assert data["tool_result"] is None


def test_tool_chat_creates_pending_approval_for_write_tool():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "create_maintenance_request",
            {
                "equipment_id": "Robot-01",
                "reason": "servo drift requires inspection",
            },
        ),
    ), patch(
        "backend.app.services.tool_calling_service.ToolCallingService.execute_tool",
    ) as mock_execute:
        response = client.post(
            "/tools/chat",
            json={"message": "Robot-01 정비 요청 생성해줘"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "waiting_approval"
    assert data["termination_reason"] == "approval_required"
    assert data["tool_called"] == "create_maintenance_request"
    assert data["tool_result"] is None
    assert data["approval_request"]["status"] == "pending"
    assert data["approval_request"]["tool_name"] == (
        "create_maintenance_request"
    )
    assert mock_execute.called is False
