from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.genai import types

from backend.main import app
from backend.app.core.database import SessionLocal
from backend.app.schemas.agent import MemoryMessage
from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT,
)
from backend.app.services.guardrail_service import (
    InMemoryApprovalStore,
)
from backend.app.services.memory_service import (
    InMemoryMemoryStore,
)
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
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


def test_agent_chat_without_session_id_keeps_existing_stateless_behavior():
    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        return_value=_build_text_response("세션 없이 답변합니다."),
    ):
        response = client.post(
            "/agent/chat",
            json={"message": "안녕"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "세션 없이 답변합니다."


def test_agent_service_saves_memory_after_first_request():
    db = SessionLocal()
    memory_store = InMemoryMemoryStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_status",
                    {"equipment_id": "Robot-01"},
                ),
                _build_text_response(
                    "Robot-01은 현재 high risk 상태입니다."
                ),
            ],
        ):
            result = AgentService.chat(
                db=db,
                message="Robot-01 현재 상태 알려줘",
                session_id="demo-001",
                memory_store=memory_store,
            )

        messages = memory_store.get_messages("demo-001")

        assert result.answer == "Robot-01은 현재 high risk 상태입니다."
        assert [message.role for message in messages] == [
            "user",
            "assistant",
        ]
        assert messages[0].content == "Robot-01 현재 상태 알려줘"
        assert messages[1].content == "Robot-01은 현재 high risk 상태입니다."
    finally:
        db.close()


def test_agent_service_injects_previous_conversation_for_follow_up_request():
    db = SessionLocal()
    memory_store = InMemoryMemoryStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_status",
                    {"equipment_id": "Robot-01"},
                ),
                _build_text_response(
                    "Robot-01은 현재 high risk 상태입니다."
                ),
                _build_tool_response(
                    "get_equipment_incidents",
                    {"equipment_name": "Robot-01"},
                ),
                _build_text_response(
                    "Robot-01의 과거 장애 이력을 확인했습니다."
                ),
            ],
        ) as mock_generate:
            AgentService.chat(
                db=db,
                message="Robot-01 현재 상태 알려줘",
                session_id="demo-001",
                memory_store=memory_store,
            )
            AgentService.chat(
                db=db,
                message="그럼 과거 장애는?",
                session_id="demo-001",
                memory_store=memory_store,
            )

        follow_up_context = mock_generate.call_args_list[2].kwargs[
            "contents"
        ][0].parts[0].text

        assert "Previous Conversation:" in follow_up_context
        assert "User: Robot-01 현재 상태 알려줘" in follow_up_context
        assert (
            "Assistant: Robot-01은 현재 high risk 상태입니다."
            in follow_up_context
        )
        assert "Current Request:\n그럼 과거 장애는?" in follow_up_context
    finally:
        db.close()


def test_agent_service_applies_recent_memory_limit():
    db = SessionLocal()
    memory_store = InMemoryMemoryStore()

    for index in range(8):
        memory_store.append_message(
            "demo-001",
            MemoryMessage(
                role=(
                    "user"
                    if index % 2 == 0
                    else "assistant"
                ),
                content=f"message-{index}",
            ),
        )

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_text_response("최근 대화만 참고했습니다."),
            ],
        ) as mock_generate:
            AgentService.chat(
                db=db,
                message="follow-up",
                session_id="demo-001",
                memory_store=memory_store,
                max_memory_messages=4,
            )

        context_text = mock_generate.call_args.kwargs["contents"][0].parts[
            0
        ].text

        assert "message-0" not in context_text
        assert "message-3" not in context_text
        assert "message-4" in context_text
        assert "message-7" in context_text
    finally:
        db.close()



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


def test_agent_write_tool_waits_for_human_approval():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "servo drift requires inspection",
                },
            ),
        ), patch(
            "backend.app.services.tool_calling_service.ToolCallingService.execute_tool"
        ) as mock_execute:
            result = AgentService.run(
                db=db,
                agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                message="Robot-01 정비 요청 생성해줘",
                approval_store=approval_store,
            )

        assert result.status == "waiting_approval"
        assert result.termination_reason == "approval_required"
        assert result.approval_request is not None
        assert result.approval_request.status == "pending"
        assert result.steps[0].tool_called == "create_maintenance_request"
        assert result.steps[0].approval_required is True
        assert result.steps[0].approval_id == (
            result.approval_request.approval_id
        )
        assert mock_execute.called is False
    finally:
        db.close()


def test_agent_invalid_arguments_are_rejected_before_guardrail():
    db = SessionLocal()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                },
            ),
        ), patch(
            "backend.app.services.tool_calling_service.ToolCallingService.evaluate_guardrail"
        ) as mock_guardrail:
            try:
                AgentService.run(
                    db=db,
                    agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                    message="Robot-01 정비 요청 생성해줘",
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError as exc:
                assert exc.state.termination_reason == "invalid_arguments"
                assert mock_guardrail.called is False
    finally:
        db.close()


def test_agent_blocks_disallowed_write_tool_before_guardrail():
    db = SessionLocal()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "servo drift requires inspection",
                },
            ),
        ), patch(
            "backend.app.services.tool_calling_service.ToolCallingService.evaluate_guardrail"
        ) as mock_guardrail:
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="Robot-01 정비 요청 생성해줘",
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError as exc:
                assert exc.state.termination_reason == "invalid_tool"
                assert mock_guardrail.called is False
    finally:
        db.close()


def test_approve_executes_pending_tool():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "servo drift requires inspection",
                },
            ),
        ):
            pending = AgentService.run(
                db=db,
                agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                message="Robot-01 정비 요청 생성해줘",
                approval_store=approval_store,
            )

        approval_id = pending.approval_request.approval_id
        executed = AgentService.approve_tool_execution(
            db=db,
            approval_id=approval_id,
            approval_store=approval_store,
        )

        assert executed.approval_request.status == "executed"
        assert executed.tool_result["equipment_id"] == "Robot-01"
        assert executed.tool_result["status"] == "created"
    finally:
        db.close()


def test_same_approval_id_cannot_be_approved_twice():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()
    original_execute_tool = ToolCallingService.execute_tool

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "servo drift requires inspection",
                },
            ),
        ), patch(
            "backend.app.services.tool_calling_service.ToolCallingService.execute_tool",
            wraps=original_execute_tool,
        ) as mock_execute:
            pending = AgentService.run(
                db=db,
                agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                message="Robot-01 정비 요청 생성해줘",
                approval_store=approval_store,
            )
            approval_id = pending.approval_request.approval_id

            first_result = AgentService.approve_tool_execution(
                db=db,
                approval_id=approval_id,
                approval_store=approval_store,
            )

            assert first_result.approval_request.status == "executed"
            assert mock_execute.call_count == 1

            try:
                AgentService.approve_tool_execution(
                    db=db,
                    approval_id=approval_id,
                    approval_store=approval_store,
                )
                assert False, "Expected ValueError"
            except ValueError as exc:
                assert "already executed" in str(exc)

            assert mock_execute.call_count == 1
    finally:
        db.close()


def test_reject_blocks_tool_execution():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "servo drift requires inspection",
                },
            ),
        ):
            pending = AgentService.run(
                db=db,
                agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                message="Robot-01 정비 요청 생성해줘",
                approval_store=approval_store,
            )

        approval_id = pending.approval_request.approval_id
        rejected = AgentService.reject_tool_execution(
            approval_id=approval_id,
            approval_store=approval_store,
        )

        assert rejected.approval_request.status == "rejected"

        try:
            AgentService.approve_tool_execution(
                db=db,
                approval_id=approval_id,
                approval_store=approval_store,
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "rejected" in str(exc)
    finally:
        db.close()


def test_approved_request_cannot_be_rejected():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "servo drift requires inspection",
                },
            ),
        ):
            pending = AgentService.run(
                db=db,
                agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                message="Robot-01 정비 요청 생성해줘",
                approval_store=approval_store,
            )

        approval_id = pending.approval_request.approval_id
        AgentService.approve_tool_execution(
            db=db,
            approval_id=approval_id,
            approval_store=approval_store,
        )

        try:
            AgentService.reject_tool_execution(
                approval_id=approval_id,
                approval_store=approval_store,
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "already executed" in str(exc)
    finally:
        db.close()


def test_approval_api_executes_approved_request():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "create_maintenance_request",
            {
                "equipment_id": "Robot-01",
                "reason": "servo drift requires inspection",
            },
        ),
    ):
        pending_response = client.post(
            "/tools/chat",
            json={"message": "Robot-01 정비 요청 생성해줘"},
        )

    approval_id = pending_response.json()["approval_request"][
        "approval_id"
    ]
    response = client.post(
        f"/agent/approvals/{approval_id}/approve"
    )

    assert response.status_code == 200
    assert response.json()["approval_request"]["status"] == "executed"
    assert response.json()["tool_result"]["equipment_id"] == "Robot-01"


def test_approval_api_blocks_second_approve_call():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "create_maintenance_request",
            {
                "equipment_id": "Robot-01",
                "reason": "servo drift requires inspection",
            },
        ),
    ):
        pending_response = client.post(
            "/tools/chat",
            json={"message": "Robot-01 정비 요청 생성해줘"},
        )

    approval_id = pending_response.json()["approval_request"][
        "approval_id"
    ]
    first_response = client.post(
        f"/agent/approvals/{approval_id}/approve"
    )
    second_response = client.post(
        f"/agent/approvals/{approval_id}/approve"
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 400
    assert "already executed" in second_response.json()["detail"]


def test_approval_api_rejects_request():
    with patch(
        "backend.app.services.tool_calling_service.LlmService.generate_content",
        return_value=_build_tool_response(
            "create_maintenance_request",
            {
                "equipment_id": "Robot-01",
                "reason": "servo drift requires inspection",
            },
        ),
    ):
        pending_response = client.post(
            "/tools/chat",
            json={"message": "Robot-01 정비 요청 생성해줘"},
        )

    approval_id = pending_response.json()["approval_request"][
        "approval_id"
    ]
    response = client.post(
        f"/agent/approvals/{approval_id}/reject"
    )

    assert response.status_code == 200
    assert response.json()["approval_request"]["status"] == "rejected"
    assert response.json()["tool_result"] is None
