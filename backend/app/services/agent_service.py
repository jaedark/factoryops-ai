from collections.abc import Callable
from time import perf_counter, sleep
from uuid import uuid4

from google.genai import types
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.agents import INCIDENT_ANALYSIS_AGENT
from backend.app.agents.base import AgentDefinition
from backend.app.core.config import get_settings
from backend.app.observability import (
    NoOpObservabilitySink,
    ObservabilityEventType,
    ObservabilityRecorder,
    ObservabilitySink,
)
from backend.app.resilience import (
    AgentExecutionConfig,
    AgentRuntimeContext,
    CircuitBreaker,
    ResilientExecutionService,
    RetryPolicy,
)
from backend.app.schemas.agent import (
    AgentState,
    AgentStatus,
    AgentStep,
    AgentTerminationReason,
    ApprovalActionResponse,
    ApprovalRequest,
    MemoryMessage,
)
from backend.app.services.context_builder import ContextBuilder
from backend.app.services.llm_service import LlmService
from backend.app.services.memory_service import (
    InMemoryMemoryStore,
    MemoryStore,
    MemoryStoreError,
)
from backend.app.services.tool_calling_service import ToolCallingService


class AgentResult(BaseModel):
    trace_id: str
    answer: str
    steps: list[AgentStep]
    total_steps: int
    status: AgentStatus
    termination_reason: AgentTerminationReason
    approval_request: ApprovalRequest | None = None


class AgentExecutionError(ValueError):
    def __init__(
        self,
        message: str,
        state: AgentState,
    ) -> None:
        super().__init__(message)
        self.state = state


class AgentService:
    DEFAULT_MEMORY_MAX_MESSAGES = 6
    _MEMORY_STORE: MemoryStore = InMemoryMemoryStore()
    _OBSERVABILITY_SINK: ObservabilitySink = NoOpObservabilitySink()

    @staticmethod
    def _build_user_content(
        message: str,
    ) -> types.Content:
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )

    @staticmethod
    def _create_state(
        trace_id: str,
        message: str,
        max_steps: int,
    ) -> AgentState:
        return AgentState(
            trace_id=trace_id,
            conversation=[AgentService._build_user_content(message)],
            steps=[],
            current_step=0,
            max_steps=max_steps,
            status=AgentStatus.RUNNING,
        )

    @staticmethod
    def _build_tool_result_content(
        tool_name: str,
        tool_result,
    ) -> types.Content:
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=tool_name,
                    response={"output": tool_result},
                )
            ],
        )

    @staticmethod
    def _build_result(
        state: AgentState,
    ) -> AgentResult:
        return AgentResult(
            trace_id=state.trace_id,
            answer=state.final_answer or "",
            steps=state.steps,
            total_steps=len(state.steps),
            status=state.status,
            termination_reason=(
                state.termination_reason
                or AgentTerminationReason.FINAL_ANSWER
            ),
            approval_request=state.approval_request,
        )

    @classmethod
    def get_memory_store(
        cls,
    ) -> MemoryStore:
        return cls._MEMORY_STORE

    @classmethod
    def get_observability_sink(
        cls,
    ) -> ObservabilitySink:
        return cls._OBSERVABILITY_SINK

    @classmethod
    def get_llm_circuit_breaker(
        cls,
    ) -> CircuitBreaker:
        return (
            ResilientExecutionService
            .get_default_llm_circuit_breaker()
        )

    @staticmethod
    def _generate_trace_id() -> str:
        return f"trc-{uuid4()}"

    @classmethod
    def _build_runtime_context(
        cls,
        *,
        trace_id: str,
        agent_definition: AgentDefinition,
        sink: ObservabilitySink,
        max_steps: int,
        retry_policy: RetryPolicy,
        sleep_fn: Callable[[float], None],
        llm_circuit_breaker: CircuitBreaker,
        llm_timeout_seconds: float,
        tool_timeout_seconds: float,
        approval_store=None,
    ) -> AgentRuntimeContext:
        config = AgentExecutionConfig(
            max_steps=max_steps,
            llm_timeout_seconds=llm_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
            retry_policy=retry_policy,
        )
        recorder = ObservabilityRecorder(
            sink=sink,
            trace_id=trace_id,
            agent_name=agent_definition.name,
        )
        return AgentRuntimeContext(
            trace_id=trace_id,
            recorder=recorder,
            config=config,
            llm_circuit_breaker=llm_circuit_breaker,
            sleep_fn=sleep_fn,
            approval_store=approval_store,
        )

    @classmethod
    def _execute_tool_with_resilience(
        cls,
        *,
        db: Session,
        trace_id: str,
        agent_definition: AgentDefinition,
        tool_name: str,
        tool_arguments: dict,
        step_number: int,
        state: AgentState,
        sink: ObservabilitySink,
        retry_policy: RetryPolicy,
        sleep_fn: Callable[[float], None],
        timeout_seconds: float,
    ):
        runtime = cls._build_runtime_context(
            trace_id=trace_id,
            agent_definition=agent_definition,
            sink=sink,
            max_steps=state.max_steps,
            retry_policy=retry_policy,
            sleep_fn=sleep_fn,
            llm_circuit_breaker=cls.get_llm_circuit_breaker(),
            llm_timeout_seconds=get_settings().llm_timeout_seconds,
            tool_timeout_seconds=timeout_seconds,
        )
        return ResilientExecutionService.execute_tool(
            db=db,
            runtime=runtime,
            agent_definition=agent_definition,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            step_number=step_number,
            status=state.status,
        )

    @classmethod
    def _build_context_message(
        cls,
        message: str,
        session_id: str | None,
        memory_store: MemoryStore | None,
        max_memory_messages: int,
    ) -> str:
        if session_id is None:
            return message

        store = memory_store or cls.get_memory_store()
        try:
            memory_messages = store.get_messages(session_id)
        except Exception as exc:
            raise MemoryStoreError(
                f"Failed to read session memory: {session_id}"
            ) from exc

        return ContextBuilder.build(
            current_message=message,
            memory_messages=memory_messages,
            max_messages=max_memory_messages,
        )

    @classmethod
    def _append_session_memory(
        cls,
        session_id: str | None,
        user_message: str,
        assistant_answer: str,
        memory_store: MemoryStore | None,
    ) -> None:
        if session_id is None:
            return

        store = memory_store or cls.get_memory_store()
        try:
            store.append_message(
                session_id,
                MemoryMessage(role="user", content=user_message),
            )
            store.append_message(
                session_id,
                MemoryMessage(
                    role="assistant",
                    content=assistant_answer,
                ),
            )
        except Exception as exc:
            raise MemoryStoreError(
                f"Failed to write session memory: {session_id}"
            ) from exc

    @staticmethod
    def _get_tool_error_reason(
        error_message: str,
    ) -> AgentTerminationReason:
        if (
            "Unsupported tool requested" in error_message
            or "Tool not allowed for agent" in error_message
        ):
            return AgentTerminationReason.INVALID_TOOL
        if "Invalid arguments for tool" in error_message:
            return AgentTerminationReason.INVALID_ARGUMENTS
        return AgentTerminationReason.TOOL_ERROR

    @staticmethod
    def _validate_allowed_tool(
        agent_definition: AgentDefinition,
        tool_name: str,
    ) -> None:
        if tool_name not in agent_definition.allowed_tools:
            raise ValueError(
                "Tool not allowed for agent "
                f"'{agent_definition.name}': {tool_name}"
            )

    @classmethod
    def run(
        cls,
        db: Session,
        agent_definition: AgentDefinition,
        message: str,
        max_steps: int | None = None,
        session_id: str | None = None,
        approval_store=None,
        observability_sink: ObservabilitySink | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep_fn: Callable[[float], None] = sleep,
        llm_circuit_breaker: CircuitBreaker | None = None,
        llm_timeout_seconds: float | None = None,
        tool_timeout_seconds: float | None = None,
    ) -> AgentResult:
        execution_config = AgentExecutionConfig.from_settings(
            get_settings(),
            max_steps=max_steps,
            llm_timeout_seconds=llm_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
            retry_policy=retry_policy,
        )
        trace_id = cls._generate_trace_id()
        state = cls._create_state(
            trace_id,
            message,
            execution_config.max_steps,
        )
        runtime = cls._build_runtime_context(
            trace_id=trace_id,
            agent_definition=agent_definition,
            sink=observability_sink or cls.get_observability_sink(),
            max_steps=execution_config.max_steps,
            retry_policy=execution_config.retry_policy,
            sleep_fn=sleep_fn,
            llm_circuit_breaker=(
                llm_circuit_breaker
                or cls.get_llm_circuit_breaker()
            ),
            llm_timeout_seconds=execution_config.llm_timeout_seconds,
            tool_timeout_seconds=execution_config.tool_timeout_seconds,
            approval_store=approval_store,
        )
        recorder = runtime.recorder
        run_started_at = perf_counter()
        recorder.emit(
            ObservabilityEventType.AGENT_STARTED,
            success=True,
            status=AgentStatus.RUNNING.value,
            metadata={
                "query_length": len(message),
                "max_steps": execution_config.max_steps,
                "session_id_present": session_id is not None,
            },
        )

        for step_number in range(1, runtime.config.max_steps + 1):
            state.current_step = step_number
            try:
                response = ResilientExecutionService.call_llm(
                    runtime=runtime,
                    agent_definition=agent_definition,
                    conversation=state.conversation,
                    step_number=step_number,
                    status=state.status,
                )
            except Exception as exc:
                state.status = AgentStatus.FAILED
                state.termination_reason = AgentTerminationReason.LLM_ERROR
                state.error = str(exc)
                recorder.agent_failed(state, run_started_at)
                raise AgentExecutionError(str(exc), state) from exc

            function_calls = response.function_calls or []
            if not function_calls:
                state.final_answer = response.text
                state.status = AgentStatus.COMPLETED
                state.termination_reason = (
                    AgentTerminationReason.FINAL_ANSWER
                )
                recorder.emit(
                    ObservabilityEventType.AGENT_COMPLETED,
                    latency_ms=(perf_counter() - run_started_at) * 1000,
                    success=True,
                    status=state.status.value,
                    metadata={
                        "termination_reason": (
                            state.termination_reason.value
                        ),
                        "step_count": len(state.steps),
                        "final_answer_length": len(
                            state.final_answer or ""
                        ),
                    },
                )
                return cls._build_result(state)

            if len(function_calls) > 1:
                error_message = (
                    "Only a single tool call is supported per step"
                )
                state.status = AgentStatus.FAILED
                state.termination_reason = AgentTerminationReason.TOOL_ERROR
                state.error = error_message
                recorder.agent_failed(state, run_started_at)
                raise AgentExecutionError(error_message, state)

            function_call = function_calls[0]
            tool_name = function_call.name
            try:
                tool_arguments = ToolCallingService.validate_tool_call(
                    tool_name=tool_name,
                    tool_arguments=function_call.args,
                )
                cls._validate_allowed_tool(agent_definition, tool_name)
                guardrail_decision = ToolCallingService.evaluate_guardrail(
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                )
                if guardrail_decision.approval_required:
                    approval_request = (
                        ToolCallingService.create_approval_request(
                            tool_name=tool_name,
                            tool_arguments=tool_arguments,
                            reason=guardrail_decision.reason,
                            session_id=session_id,
                            approval_store=runtime.approval_store,
                        )
                    )
                    state.steps.append(
                        AgentStep(
                            step=step_number,
                            tool_called=tool_name,
                            tool_arguments=tool_arguments,
                            tool_result=None,
                            success=False,
                            approval_required=True,
                            approval_id=approval_request.approval_id,
                        )
                    )
                    state.status = AgentStatus.WAITING_APPROVAL
                    state.termination_reason = (
                        AgentTerminationReason.APPROVAL_REQUIRED
                    )
                    state.final_answer = guardrail_decision.reason
                    state.approval_request = approval_request
                    recorder.emit(
                        ObservabilityEventType.APPROVAL_REQUIRED,
                        step=step_number,
                        tool_name=tool_name,
                        success=False,
                        status=state.status.value,
                        metadata={
                            "approval_id": approval_request.approval_id,
                            "risk_level": (
                                guardrail_decision.risk_level.value
                            ),
                            **recorder.tool_argument_metadata(
                                tool_arguments
                            ),
                        },
                    )
                    return cls._build_result(state)

                tool_result = ResilientExecutionService.execute_tool(
                    db=db,
                    runtime=runtime,
                    agent_definition=agent_definition,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    step_number=step_number,
                    status=state.status,
                )
            except Exception as exc:
                error_message = str(exc)
                state.steps.append(
                    AgentStep(
                        step=step_number,
                        tool_called=tool_name,
                        tool_arguments=function_call.args or {},
                        tool_result=None,
                        success=False,
                        error=error_message,
                    )
                )
                state.status = AgentStatus.FAILED
                state.termination_reason = cls._get_tool_error_reason(
                    error_message
                )
                state.error = error_message
                recorder.agent_failed(state, run_started_at)
                raise AgentExecutionError(error_message, state) from exc

            state.steps.append(
                AgentStep(
                    step=step_number,
                    tool_called=tool_name,
                    tool_arguments=tool_arguments,
                    tool_result=tool_result,
                    success=True,
                )
            )
            state.conversation.append(response.candidates[0].content)
            state.conversation.append(
                cls._build_tool_result_content(tool_name, tool_result)
            )

        error_message = "Agent step limit exceeded"
        state.status = AgentStatus.FAILED
        state.termination_reason = (
            AgentTerminationReason.MAX_STEPS_EXCEEDED
        )
        state.error = error_message
        recorder.agent_failed(state, run_started_at)
        raise AgentExecutionError(error_message, state)

    @classmethod
    def chat(
        cls,
        db: Session,
        message: str,
        max_steps: int | None = None,
        session_id: str | None = None,
        memory_store: MemoryStore | None = None,
        max_memory_messages: int = DEFAULT_MEMORY_MAX_MESSAGES,
        observability_sink: ObservabilitySink | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep_fn: Callable[[float], None] = sleep,
        llm_circuit_breaker: CircuitBreaker | None = None,
        llm_timeout_seconds: float | None = None,
        tool_timeout_seconds: float | None = None,
    ) -> AgentResult:
        context_message = cls._build_context_message(
            message,
            session_id,
            memory_store,
            max_memory_messages,
        )
        result = cls.run(
            db=db,
            agent_definition=INCIDENT_ANALYSIS_AGENT,
            message=context_message,
            max_steps=max_steps,
            session_id=session_id,
            approval_store=ToolCallingService.get_approval_store(),
            observability_sink=observability_sink,
            retry_policy=retry_policy,
            sleep_fn=sleep_fn,
            llm_circuit_breaker=llm_circuit_breaker,
            llm_timeout_seconds=llm_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
        )
        if result.status != AgentStatus.WAITING_APPROVAL:
            cls._append_session_memory(
                session_id,
                message,
                result.answer,
                memory_store,
            )
        return result

    @classmethod
    def approve_tool_execution(
        cls,
        db: Session,
        approval_id: str,
        approval_store=None,
    ) -> ApprovalActionResponse:
        return ToolCallingService.approve_tool_execution(
            db=db,
            approval_id=approval_id,
            approval_store=approval_store,
        )

    @classmethod
    def reject_tool_execution(
        cls,
        approval_id: str,
        approval_store=None,
    ) -> ApprovalActionResponse:
        return ToolCallingService.reject_tool_execution(
            approval_id=approval_id,
            approval_store=approval_store,
        )
