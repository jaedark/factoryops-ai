import json
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT,
)
from backend.app.core.database import SessionLocal
from backend.app.observability import (
    InMemoryObservabilitySink,
    LoggingObservabilitySink,
    ObservabilityEventType,
)
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)
from backend.app.services.guardrail_service import (
    InMemoryApprovalStore,
)


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


def _event_types(sink: InMemoryObservabilitySink) -> list[str]:
    return [
        event.event_type.value
        for event in sink.events
    ]


def test_agent_run_generates_trace_id_and_emits_started_event():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_text_response("안녕하세요."),
        ):
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="안녕",
                observability_sink=sink,
            )

        assert result.trace_id.startswith("trc-")
        assert sink.events[0].event_type == (
            ObservabilityEventType.AGENT_STARTED
        )
        assert sink.events[0].trace_id == result.trace_id
        assert sink.events[0].status == "running"
    finally:
        db.close()


def test_different_runs_generate_different_trace_ids():
    db = SessionLocal()

    try:
        first_sink = InMemoryObservabilitySink()
        second_sink = InMemoryObservabilitySink()

        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_text_response("안녕하세요."),
        ):
            first = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="안녕",
                observability_sink=first_sink,
            )
            second = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="안녕",
                observability_sink=second_sink,
            )

        assert first.trace_id != second.trace_id
    finally:
        db.close()


def test_same_run_events_share_same_trace_id():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_incidents",
                    {"equipment_name": "Robot-01"},
                ),
                _build_text_response("Robot-01 장애 이력을 확인했습니다."),
            ],
        ):
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="Robot-01 장애 이력 알려줘",
                observability_sink=sink,
            )

        assert {event.trace_id for event in sink.events} == {
            result.trace_id
        }
    finally:
        db.close()


def test_agent_run_emits_llm_and_tool_events_with_latency():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_incidents",
                    {"equipment_name": "Robot-01"},
                ),
                _build_text_response("Robot-01 장애 이력을 확인했습니다."),
            ],
        ):
            AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="Robot-01 장애 이력 알려줘",
                observability_sink=sink,
            )

        event_types = _event_types(sink)
        assert event_types == [
            "agent_started",
            "llm_call_started",
            "llm_call_completed",
            "tool_call_started",
            "tool_call_completed",
            "llm_call_started",
            "llm_call_completed",
            "agent_completed",
        ]
        llm_completed = [
            event
            for event in sink.events
            if event.event_type
            == ObservabilityEventType.LLM_CALL_COMPLETED
        ]
        tool_completed = [
            event
            for event in sink.events
            if event.event_type
            == ObservabilityEventType.TOOL_CALL_COMPLETED
        ]
        assert all(
            event.latency_ms is not None
            and event.latency_ms >= 0
            for event in llm_completed
        )
        assert all(
            event.latency_ms is not None
            and event.latency_ms >= 0
            for event in tool_completed
        )
    finally:
        db.close()


def test_approval_required_emits_event_without_tool_start():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
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
            result = AgentService.run(
                db=db,
                agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                message="Robot-01 정비 요청 생성해줘",
                approval_store=approval_store,
                observability_sink=sink,
            )

        assert result.status == "waiting_approval"
        assert _event_types(sink) == [
            "agent_started",
            "llm_call_started",
            "llm_call_completed",
            "approval_required",
        ]
        approval_event = sink.events[-1]
        assert approval_event.tool_name == "create_maintenance_request"
        assert approval_event.status == "waiting_approval"
    finally:
        db.close()


def test_tool_failure_emits_tool_failed_and_agent_failed():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "get_equipment_incidents",
                {"equipment_name": "Robot-01"},
            ),
        ), patch(
            "backend.app.services.tool_calling_service.ToolCallingService.execute_tool",
            side_effect=RuntimeError("tool exploded"),
        ):
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="Robot-01 장애 이력 알려줘",
                    observability_sink=sink,
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError:
                pass

        assert _event_types(sink) == [
            "agent_started",
            "llm_call_started",
            "llm_call_completed",
            "tool_call_started",
            "tool_call_failed",
            "agent_failed",
        ]
        assert sink.events[-2].error == "tool exploded"
        assert sink.events[-1].status == "failed"
    finally:
        db.close()


def test_llm_failure_emits_llm_failed_and_agent_failed():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=RuntimeError("llm exploded"),
        ):
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="안녕",
                    observability_sink=sink,
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError:
                pass

        assert _event_types(sink) == [
            "agent_started",
            "llm_call_started",
            "llm_call_failed",
            "agent_failed",
        ]
        assert sink.events[-2].error == "llm exploded"
        assert sink.events[-1].error == "llm exploded"
    finally:
        db.close()


def test_max_steps_exceeded_emits_agent_failed():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
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
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="계속 반복해",
                    max_steps=2,
                    observability_sink=sink,
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError:
                pass

        assert sink.events[-1].event_type == (
            ObservabilityEventType.AGENT_FAILED
        )
        assert sink.events[-1].metadata["termination_reason"] == (
            "max_steps_exceeded"
        )
    finally:
        db.close()


def test_in_memory_sink_isolation():
    first_sink = InMemoryObservabilitySink()
    second_sink = InMemoryObservabilitySink()

    assert first_sink.events == []
    assert second_sink.events == []
    assert first_sink.events is not second_sink.events


def test_observability_sink_is_optional():
    db = SessionLocal()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_text_response("안녕하세요."),
        ):
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="안녕",
            )

        assert result.answer == "안녕하세요."
        assert result.trace_id.startswith("trc-")
    finally:
        db.close()


def test_sensitive_data_is_not_stored_in_event_metadata():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    secret_message = "비밀 prompt 123"

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_status",
                    {"equipment_id": "Robot-01"},
                ),
                _build_text_response("상태를 확인했습니다."),
            ],
        ):
            AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message=secret_message,
                observability_sink=sink,
            )

        dumped = [event.model_dump(mode="json") for event in sink.events]
        assert all(
            secret_message not in str(event_dump)
            for event_dump in dumped
        )
        assert all(
            "tool_result" not in event_dump["metadata"]
            for event_dump in dumped
        )
    finally:
        db.close()


def test_logging_observability_sink_emits_json_log():
    target = []
    sink = LoggingObservabilitySink(
        logger=SimpleNamespace(info=target.append)
    )
    sink.emit(
        event=SimpleNamespace(
            model_dump=lambda mode="json": {
                "trace_id": "trc-123",
                "event_type": "agent_started",
                "timestamp": "2026-09-04T00:00:00+00:00",
                "agent_name": "incident_analysis",
                "step": None,
                "tool_name": None,
                "latency_ms": None,
                "success": True,
                "status": "running",
                "error": None,
                "metadata": {"query_length": 4},
            }
        )
    )

    assert len(target) == 1
    payload = json.loads(target[0])
    assert payload["trace_id"] == "trc-123"
    assert payload["metadata"]["query_length"] == 4


def test_trace_id_is_exposed_through_chat_result():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_text_response("안녕하세요."),
        ):
            result = AgentService.chat(
                db=db,
                message="안녕",
                observability_sink=sink,
            )

        assert result.trace_id.startswith("trc-")
    finally:
        db.close()


def test_follow_up_memory_and_observability_work_together():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_text_response("첫 답변입니다."),
                _build_text_response("후속 답변입니다."),
            ],
        ):
            first = AgentService.chat(
                db=db,
                message="첫 질문",
                session_id="obs-demo",
                observability_sink=sink,
            )
            second = AgentService.chat(
                db=db,
                message="후속 질문",
                session_id="obs-demo",
                observability_sink=sink,
            )

        assert first.trace_id != second.trace_id
        assert len(
            [
                event
                for event in sink.events
                if event.event_type
                == ObservabilityEventType.AGENT_COMPLETED
            ]
        ) == 2
    finally:
        db.close()
