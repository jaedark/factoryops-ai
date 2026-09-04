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
    ObservabilityEventType,
)
from backend.app.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    ErrorCategory,
    RetryPolicy,
    calculate_backoff_delay,
    classify_error,
)
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)
from backend.app.services.guardrail_service import (
    InMemoryApprovalStore,
)
from backend.app.services.tool_calling_service import (
    ToolCallingService,
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


def test_classify_timeout_as_transient():
    result = classify_error(TimeoutError("request timeout"))

    assert result.retryable is True
    assert result.category == ErrorCategory.TIMEOUT


def test_classify_rate_limit_as_transient():
    result = classify_error(RuntimeError("429 rate limit exceeded"))

    assert result.retryable is True
    assert result.category == ErrorCategory.RATE_LIMIT


def test_classify_invalid_arguments_as_permanent():
    result = classify_error(
        ValueError("Invalid arguments for tool 'get_incident'")
    )

    assert result.retryable is False
    assert result.category == ErrorCategory.INVALID_ARGUMENTS


def test_classify_invalid_tool_as_permanent():
    result = classify_error(
        ValueError("Unsupported tool requested: delete_all")
    )

    assert result.retryable is False
    assert result.category == ErrorCategory.INVALID_TOOL


def test_calculate_exponential_backoff_delay():
    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.1,
        max_delay_seconds=0.5,
        backoff_multiplier=2.0,
    )

    assert calculate_backoff_delay(policy, 1) == 0.1
    assert calculate_backoff_delay(policy, 2) == 0.2
    assert calculate_backoff_delay(policy, 4) == 0.5


def test_circuit_breaker_closed_allows_requests():
    breaker = CircuitBreaker()

    assert breaker.allow_request() == "closed"


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()
    state = breaker.record_failure()

    assert state == "open"
    assert breaker.state == "open"


def test_circuit_breaker_rejects_when_open():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()

    try:
        breaker.allow_request()
        assert False, "Expected CircuitOpenError"
    except CircuitOpenError:
        pass


def test_circuit_breaker_enters_half_open_after_recovery_timeout():
    now = [0.0]
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=5.0,
        time_fn=lambda: now[0],
    )
    breaker.record_failure()
    now[0] = 6.0

    assert breaker.allow_request() == "half_open"
    assert breaker.state == "half_open"


def test_circuit_breaker_half_open_success_closes():
    now = [0.0]
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=5.0,
        time_fn=lambda: now[0],
    )
    breaker.record_failure()
    now[0] = 6.0
    breaker.allow_request()
    breaker.record_success()

    assert breaker.state == "closed"


def test_circuit_breaker_half_open_failure_reopens():
    now = [0.0]
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_seconds=5.0,
        time_fn=lambda: now[0],
    )
    breaker.record_failure()
    now[0] = 6.0
    breaker.allow_request()
    breaker.record_failure()

    assert breaker.state == "open"


def test_circuit_breaker_instances_are_isolated():
    first = CircuitBreaker(failure_threshold=1)
    second = CircuitBreaker(failure_threshold=1)

    first.record_failure()

    assert first.state == "open"
    assert second.state == "closed"


def test_transient_llm_error_retries_then_succeeds():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    delays = []

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                TimeoutError("request timeout"),
                _build_text_response("복구 성공"),
            ],
        ) as mock_generate:
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="안녕",
                observability_sink=sink,
                retry_policy=RetryPolicy(max_attempts=3),
                sleep_fn=delays.append,
                llm_circuit_breaker=CircuitBreaker(),
            )

        retry_event = next(
            event
            for event in sink.events
            if event.event_type
            == ObservabilityEventType.RETRY_SCHEDULED
        )
        assert result.answer == "복구 성공"
        assert mock_generate.call_count == 2
        assert delays == [0.1]
        assert retry_event.metadata["component"] == "llm"
        assert retry_event.metadata["error_category"] == "timeout"
        assert sink.events[-1].event_type == (
            ObservabilityEventType.AGENT_COMPLETED
        )
    finally:
        db.close()


def test_permanent_llm_error_does_not_retry():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    breaker = CircuitBreaker()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=RuntimeError("bad request"),
        ) as mock_generate:
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="안녕",
                    observability_sink=sink,
                    retry_policy=RetryPolicy(max_attempts=3),
                    sleep_fn=lambda _delay: None,
                    llm_circuit_breaker=breaker,
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError:
                pass

        assert mock_generate.call_count == 1
        assert not any(
            event.event_type
            == ObservabilityEventType.RETRY_SCHEDULED
            for event in sink.events
        )
    finally:
        db.close()


def test_llm_retry_exhaustion_fails_and_opens_circuit():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    breaker = CircuitBreaker(failure_threshold=1)

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                TimeoutError("timeout-1"),
                TimeoutError("timeout-2"),
                TimeoutError("timeout-3"),
            ],
        ):
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="안녕",
                    observability_sink=sink,
                    retry_policy=RetryPolicy(max_attempts=3),
                    sleep_fn=lambda _delay: None,
                    llm_circuit_breaker=breaker,
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError:
                pass

        assert breaker.state == "open"
        assert sum(
            1
            for event in sink.events
            if event.event_type
            == ObservabilityEventType.RETRY_SCHEDULED
        ) == 2
        assert any(
            event.event_type == ObservabilityEventType.CIRCUIT_OPENED
            for event in sink.events
        )
        assert sink.events[-1].event_type == (
            ObservabilityEventType.AGENT_FAILED
        )
    finally:
        db.close()


def test_open_circuit_rejects_next_llm_request_without_call():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_text_response("should not run"),
        ) as mock_generate:
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="안녕",
                    observability_sink=sink,
                    llm_circuit_breaker=breaker,
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError as exc:
                assert "Circuit breaker is open" in str(exc)

        assert mock_generate.call_count == 0
        assert sink.events[1].event_type == (
            ObservabilityEventType.CIRCUIT_REJECTED
        )
    finally:
        db.close()


def test_transient_read_tool_error_retries_then_succeeds():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    delays = []

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            side_effect=[
                _build_tool_response(
                    "get_equipment_incidents",
                    {"equipment_name": "Robot-01"},
                ),
                _build_text_response("완료"),
            ],
        ), patch(
            "backend.app.services.agent_service.ToolCallingService.execute_tool",
            side_effect=[
                TimeoutError("tool timeout"),
                [{"equipment_name": "Robot-01"}],
            ],
        ) as mock_execute:
            result = AgentService.run(
                db=db,
                agent_definition=INCIDENT_ANALYSIS_AGENT,
                message="Robot-01 장애 이력 알려줘",
                observability_sink=sink,
                retry_policy=RetryPolicy(max_attempts=3),
                sleep_fn=delays.append,
                llm_circuit_breaker=CircuitBreaker(),
            )

        assert result.answer == "완료"
        assert mock_execute.call_count == 2
        assert delays == [0.1]
        assert any(
            event.event_type == ObservabilityEventType.RETRY_SCHEDULED
            and event.metadata["component"] == "tool"
            for event in sink.events
        )
    finally:
        db.close()


def test_permanent_tool_error_does_not_retry():
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
            "backend.app.services.agent_service.ToolCallingService.execute_tool",
            side_effect=RuntimeError("permanent failure"),
        ) as mock_execute:
            try:
                AgentService.run(
                    db=db,
                    agent_definition=INCIDENT_ANALYSIS_AGENT,
                    message="Robot-01 장애 이력 알려줘",
                    observability_sink=sink,
                    retry_policy=RetryPolicy(max_attempts=3),
                )
                assert False, "Expected AgentExecutionError"
            except AgentExecutionError:
                pass

        assert mock_execute.call_count == 1
        assert sum(
            1
            for event in sink.events
            if event.event_type
            == ObservabilityEventType.TOOL_CALL_FAILED
        ) == 1
    finally:
        db.close()


def test_write_tool_automatic_retry_is_disabled():
    db = SessionLocal()
    sink = InMemoryObservabilitySink()
    state = AgentService._create_state(
        trace_id="trc-test",
        message="Robot-01 정비 요청 생성해줘",
        max_steps=1,
    )

    try:
        with patch(
            "backend.app.services.agent_service.ToolCallingService.execute_tool",
            side_effect=TimeoutError("write timeout"),
        ) as mock_execute:
            try:
                AgentService._execute_tool_with_resilience(
                    db=db,
                    trace_id="trc-test",
                    agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
                    tool_name="create_maintenance_request",
                    tool_arguments={
                        "equipment_id": "Robot-01",
                        "reason": "inspection",
                    },
                    step_number=1,
                    state=state,
                    sink=sink,
                    retry_policy=RetryPolicy(max_attempts=3),
                    sleep_fn=lambda _delay: None,
                    timeout_seconds=5.0,
                )
                assert False, "Expected TimeoutError"
            except TimeoutError:
                pass

        assert mock_execute.call_count == 1
        assert not any(
            event.event_type == ObservabilityEventType.RETRY_SCHEDULED
            for event in sink.events
        )
    finally:
        db.close()


def test_approval_required_path_does_not_schedule_retry():
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
                    "reason": "inspection",
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
        assert not any(
            event.event_type == ObservabilityEventType.RETRY_SCHEDULED
            for event in sink.events
        )
    finally:
        db.close()


def test_approval_execute_failure_marks_execution_failed():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()

    try:
        with patch(
            "backend.app.services.agent_service.LlmService.generate_content",
            return_value=_build_tool_response(
                "create_maintenance_request",
                {
                    "equipment_id": "Robot-01",
                    "reason": "inspection",
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

        with patch(
            "backend.app.services.tool_calling_service.ToolCallingService.execute_tool",
            side_effect=RuntimeError("write failed"),
        ):
            try:
                AgentService.approve_tool_execution(
                    db=db,
                    approval_id=approval_id,
                    approval_store=approval_store,
                )
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "Approved tool execution failed" in str(exc)

        assert approval_store.get(approval_id).status == (
            "execution_failed"
        )
    finally:
        db.close()


def test_execution_failed_approval_cannot_be_approved_or_rejected_again():
    db = SessionLocal()
    approval_store = InMemoryApprovalStore()

    try:
        request = approval_store.create(
            tool_name="create_maintenance_request",
            tool_arguments={
                "equipment_id": "Robot-01",
                "reason": "inspection",
            },
            reason="needs approval",
        )
        approval_store.approve(request.approval_id)
        approval_store.mark_execution_failed(request.approval_id)

        try:
            approval_store.approve(request.approval_id)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "execution already failed" in str(exc)

        try:
            approval_store.reject(request.approval_id)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "execution already failed" in str(exc)

        try:
            ToolCallingService.execute_approved_tool(
                db=db,
                approval_id=request.approval_id,
                approval_store=approval_store,
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "execution already failed" in str(exc)
    finally:
        db.close()
