from time import perf_counter

from sqlalchemy.orm import Session

from backend.app.agents.base import AgentDefinition
from backend.app.core.config import get_settings
from backend.app.observability import ObservabilityEventType
from backend.app.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
)
from backend.app.resilience.retry import (
    calculate_backoff_delay,
    classify_error,
)
from backend.app.resilience.runtime import AgentRuntimeContext
from backend.app.schemas.agent import AgentStatus
from backend.app.services.guardrail_service import GuardrailService
from backend.app.services.llm_service import LlmService
from backend.app.services.tool_calling_service import ToolCallingService


class ResilientExecutionService:
    _DEFAULT_LLM_CIRCUIT_BREAKER: CircuitBreaker | None = None

    @classmethod
    def get_default_llm_circuit_breaker(
        cls,
    ) -> CircuitBreaker:
        if cls._DEFAULT_LLM_CIRCUIT_BREAKER is None:
            cls._DEFAULT_LLM_CIRCUIT_BREAKER = (
                CircuitBreaker.from_settings(get_settings())
            )
        return cls._DEFAULT_LLM_CIRCUIT_BREAKER

    @classmethod
    def reset_default_llm_circuit_breaker(cls) -> None:
        cls._DEFAULT_LLM_CIRCUIT_BREAKER = None

    @staticmethod
    def _is_retryable_tool(
        tool_name: str,
    ) -> bool:
        policy = GuardrailService.get_policy(tool_name)
        return not policy.approval_required

    @classmethod
    def call_llm(
        cls,
        *,
        runtime: AgentRuntimeContext,
        agent_definition: AgentDefinition,
        conversation: list,
        step_number: int,
        status: AgentStatus,
    ):
        recorder = runtime.recorder
        retry_policy = runtime.config.retry_policy
        breaker = runtime.llm_circuit_breaker

        try:
            circuit_state = breaker.allow_request()
        except CircuitOpenError as exc:
            classification = classify_error(exc)
            recorder.emit(
                ObservabilityEventType.CIRCUIT_REJECTED,
                step=step_number,
                success=False,
                status=AgentStatus.FAILED.value,
                error=str(exc),
                metadata={
                    "component": "llm",
                    "error_category": classification.category.value,
                    "retryable": classification.retryable,
                    "circuit_state": breaker.state,
                },
            )
            raise

        for attempt in range(1, retry_policy.max_attempts + 1):
            started_at = perf_counter()
            recorder.emit(
                ObservabilityEventType.LLM_CALL_STARTED,
                step=step_number,
                status=status.value,
                metadata={
                    "attempt": attempt,
                    "max_attempts": retry_policy.max_attempts,
                    "conversation_items": len(conversation),
                    "timeout_seconds": (
                        runtime.config.llm_timeout_seconds
                    ),
                    "circuit_state": circuit_state,
                },
            )
            try:
                response = LlmService.generate_content(
                    contents=conversation,
                    config=ToolCallingService.build_generation_config(
                        system_instruction=(
                            agent_definition.system_instruction
                        ),
                        allowed_tools=agent_definition.allowed_tools,
                    ),
                )
                breaker.record_success()
                recorder.emit(
                    ObservabilityEventType.LLM_CALL_COMPLETED,
                    step=step_number,
                    latency_ms=(perf_counter() - started_at) * 1000,
                    success=True,
                    status=status.value,
                    metadata={
                        "attempt": attempt,
                        "max_attempts": retry_policy.max_attempts,
                        "function_call_count": len(
                            response.function_calls or []
                        ),
                        "has_text_response": response.text is not None,
                        "timeout_seconds": (
                            runtime.config.llm_timeout_seconds
                        ),
                        "circuit_state": breaker.state,
                    },
                )
                return response
            except Exception as exc:
                classification = classify_error(exc)
                is_last_attempt = attempt >= retry_policy.max_attempts

                if classification.retryable and not is_last_attempt:
                    delay_seconds = calculate_backoff_delay(
                        retry_policy,
                        attempt,
                    )
                    recorder.emit(
                        ObservabilityEventType.RETRY_SCHEDULED,
                        step=step_number,
                        latency_ms=(perf_counter() - started_at) * 1000,
                        success=False,
                        status=status.value,
                        error=str(exc),
                        metadata={
                            "component": "llm",
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "max_attempts": retry_policy.max_attempts,
                            "delay_ms": delay_seconds * 1000,
                            "error_category": (
                                classification.category.value
                            ),
                            "retryable": classification.retryable,
                            "timeout_seconds": (
                                runtime.config.llm_timeout_seconds
                            ),
                            "circuit_state": breaker.state,
                        },
                    )
                    runtime.sleep_fn(delay_seconds)
                    continue

                if classification.retryable:
                    breaker_state = breaker.record_failure()
                    if breaker_state == "open":
                        recorder.emit(
                            ObservabilityEventType.CIRCUIT_OPENED,
                            step=step_number,
                            success=False,
                            status=AgentStatus.FAILED.value,
                            error=str(exc),
                            metadata={
                                "component": "llm",
                                "attempt": attempt,
                                "max_attempts": (
                                    retry_policy.max_attempts
                                ),
                                "error_category": (
                                    classification.category.value
                                ),
                                "circuit_state": breaker_state,
                            },
                        )

                recorder.emit(
                    ObservabilityEventType.LLM_CALL_FAILED,
                    step=step_number,
                    latency_ms=(perf_counter() - started_at) * 1000,
                    success=False,
                    status=AgentStatus.FAILED.value,
                    error=str(exc),
                    metadata={
                        "attempt": attempt,
                        "max_attempts": retry_policy.max_attempts,
                        "conversation_items": len(conversation),
                        "timeout_seconds": (
                            runtime.config.llm_timeout_seconds
                        ),
                        "error_category": classification.category.value,
                        "retryable": classification.retryable,
                        "circuit_state": breaker.state,
                    },
                )
                raise

        raise RuntimeError("LLM call failed")

    @classmethod
    def execute_tool(
        cls,
        *,
        db: Session,
        runtime: AgentRuntimeContext,
        agent_definition: AgentDefinition,
        tool_name: str,
        tool_arguments: dict,
        step_number: int,
        status: AgentStatus,
    ):
        recorder = runtime.recorder
        retry_policy = runtime.config.retry_policy
        retry_enabled = cls._is_retryable_tool(tool_name)
        max_attempts = (
            retry_policy.max_attempts if retry_enabled else 1
        )

        for attempt in range(1, max_attempts + 1):
            started_at = perf_counter()
            argument_metadata = recorder.tool_argument_metadata(
                tool_arguments
            )
            recorder.emit(
                ObservabilityEventType.TOOL_CALL_STARTED,
                step=step_number,
                tool_name=tool_name,
                status=status.value,
                metadata={
                    **argument_metadata,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "timeout_seconds": (
                        runtime.config.tool_timeout_seconds
                    ),
                    "retry_enabled": retry_enabled,
                },
            )
            try:
                result = ToolCallingService.execute_tool(
                    db=db,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                )
                recorder.emit(
                    ObservabilityEventType.TOOL_CALL_COMPLETED,
                    step=step_number,
                    tool_name=tool_name,
                    latency_ms=(perf_counter() - started_at) * 1000,
                    success=True,
                    status=status.value,
                    metadata={
                        **argument_metadata,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "timeout_seconds": (
                            runtime.config.tool_timeout_seconds
                        ),
                        "result_type": type(result).__name__,
                    },
                )
                return result
            except Exception as exc:
                classification = classify_error(exc)
                is_last_attempt = attempt >= max_attempts
                if (
                    retry_enabled
                    and classification.retryable
                    and not is_last_attempt
                ):
                    delay_seconds = calculate_backoff_delay(
                        retry_policy,
                        attempt,
                    )
                    recorder.emit(
                        ObservabilityEventType.RETRY_SCHEDULED,
                        step=step_number,
                        tool_name=tool_name,
                        latency_ms=(perf_counter() - started_at) * 1000,
                        success=False,
                        status=status.value,
                        error=str(exc),
                        metadata={
                            "component": "tool",
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "delay_ms": delay_seconds * 1000,
                            "error_category": (
                                classification.category.value
                            ),
                            "retryable": classification.retryable,
                            "timeout_seconds": (
                                runtime.config.tool_timeout_seconds
                            ),
                        },
                    )
                    runtime.sleep_fn(delay_seconds)
                    continue

                recorder.emit(
                    ObservabilityEventType.TOOL_CALL_FAILED,
                    step=step_number,
                    tool_name=tool_name,
                    latency_ms=(perf_counter() - started_at) * 1000,
                    success=False,
                    status=AgentStatus.FAILED.value,
                    error=str(exc),
                    metadata={
                        **argument_metadata,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_category": classification.category.value,
                        "retryable": classification.retryable,
                        "timeout_seconds": (
                            runtime.config.tool_timeout_seconds
                        ),
                        "retry_enabled": retry_enabled,
                    },
                )
                raise

        raise RuntimeError("Tool call failed")
