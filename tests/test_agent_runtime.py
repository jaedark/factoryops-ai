from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.app.agents import INCIDENT_ANALYSIS_AGENT
from backend.app.core.database import SessionLocal
from backend.app.observability import (
    InMemoryObservabilitySink,
    ObservabilityEventType,
    ObservabilityRecorder,
)
from backend.app.resilience import (
    AgentExecutionConfig,
    AgentRuntimeContext,
    CircuitBreaker,
    ResilientExecutionService,
    RetryPolicy,
)
from backend.app.services.agent_service import AgentService


client = TestClient(app, headers={"X-API-Key": "test-api-key"})


def test_execution_config_groups_runtime_defaults():
    config = AgentExecutionConfig()

    assert config.max_steps == 5
    assert config.llm_timeout_seconds == 15.0
    assert config.tool_timeout_seconds == 5.0
    assert config.retry_policy.max_attempts == 3


def test_runtime_context_keeps_injected_dependencies():
    sink = InMemoryObservabilitySink()
    recorder = ObservabilityRecorder(
        sink=sink,
        trace_id="trc-runtime",
        agent_name="incident_analysis",
    )
    breaker = CircuitBreaker()
    sleep_calls = []
    sleep_fn = sleep_calls.append
    runtime = AgentRuntimeContext(
        trace_id="trc-runtime",
        recorder=recorder,
        config=AgentExecutionConfig(),
        llm_circuit_breaker=breaker,
        sleep_fn=sleep_fn,
    )

    assert runtime.recorder.sink is sink
    assert runtime.llm_circuit_breaker is breaker
    assert runtime.sleep_fn is sleep_fn


def test_observability_recorder_builds_structured_event():
    sink = InMemoryObservabilitySink()
    recorder = ObservabilityRecorder(
        sink=sink,
        trace_id="trc-recorder",
        agent_name="incident_analysis",
    )

    recorder.emit(
        ObservabilityEventType.AGENT_STARTED,
        success=True,
        status="running",
        metadata={"query_length": 4},
    )

    assert sink.events[0].trace_id == "trc-recorder"
    assert sink.events[0].agent_name == "incident_analysis"
    assert sink.events[0].metadata == {"query_length": 4}


def test_agent_service_delegates_llm_execution_to_resilient_service():
    db = SessionLocal()
    response = SimpleNamespace(
        function_calls=None,
        candidates=[],
        text="안녕하세요.",
    )

    try:
        with patch.object(
            ResilientExecutionService,
            "call_llm",
            return_value=response,
        ) as mock_call_llm:
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="안녕",
                retry_policy=RetryPolicy(max_attempts=2),
                llm_circuit_breaker=CircuitBreaker(),
                sleep_fn=lambda _delay: None,
            )

        runtime = mock_call_llm.call_args.kwargs["runtime"]
        assert result.answer == "안녕하세요."
        assert runtime.config.retry_policy.max_attempts == 2
        assert runtime.config.max_steps == 5
    finally:
        db.close()


def test_resilient_executor_has_no_agent_service_dependency():
    executor_path = Path(
        "backend/app/resilience/executor.py"
    )
    source = executor_path.read_text(encoding="utf-8")

    assert "services.agent_service" not in source


def test_agent_chat_api_response_remains_compatible():
    response_payload = SimpleNamespace(
        function_calls=None,
        candidates=[],
        text="안녕하세요.",
    )

    with patch(
        "backend.app.services.agent_service.LlmService.generate_content",
        return_value=response_payload,
    ):
        response = client.post(
            "/agent/chat",
            json={"message": "안녕"},
        )

    assert response.status_code == 200
    assert set(response.json()) == {
        "trace_id",
        "answer",
        "steps",
        "total_steps",
        "status",
        "termination_reason",
        "approval_request",
    }
