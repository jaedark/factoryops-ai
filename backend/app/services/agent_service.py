from google.genai import types
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.agents import INCIDENT_ANALYSIS_AGENT
from backend.app.agents.base import AgentDefinition
from backend.app.schemas.agent import (
    AgentState,
    AgentStatus,
    AgentStep,
    AgentTerminationReason,
)
from backend.app.services.llm_service import LlmService
from backend.app.services.tool_calling_service import (
    ToolCallingService,
)


class AgentResult(BaseModel):
    answer: str
    steps: list[AgentStep]
    total_steps: int
    status: AgentStatus
    termination_reason: AgentTerminationReason


class AgentExecutionError(ValueError):
    def __init__(
        self,
        message: str,
        state: AgentState,
    ) -> None:
        super().__init__(message)
        self.state = state


class AgentService:
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
        message: str,
        max_steps: int,
    ) -> AgentState:
        return AgentState(
            conversation=[
                AgentService._build_user_content(message)
            ],
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
            answer=state.final_answer or "",
            steps=state.steps,
            total_steps=len(state.steps),
            status=state.status,
            termination_reason=state.termination_reason
            or AgentTerminationReason.FINAL_ANSWER,
        )

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
        max_steps: int = 5,
    ) -> AgentResult:
        state = cls._create_state(
            message=message,
            max_steps=max_steps,
        )

        for step_number in range(1, max_steps + 1):
            state.current_step = step_number

            try:
                response = LlmService.generate_content(
                    contents=state.conversation,
                    config=ToolCallingService.build_generation_config(
                        system_instruction=(
                            agent_definition.system_instruction
                        ),
                        allowed_tools=agent_definition.allowed_tools,
                    ),
                )
            except Exception as exc:
                state.status = AgentStatus.FAILED
                state.termination_reason = (
                    AgentTerminationReason.LLM_ERROR
                )
                state.error = str(exc)
                raise AgentExecutionError(
                    str(exc),
                    state,
                ) from exc

            function_calls = response.function_calls or []

            if not function_calls:
                state.final_answer = response.text
                state.status = AgentStatus.COMPLETED
                state.termination_reason = (
                    AgentTerminationReason.FINAL_ANSWER
                )
                return cls._build_result(state)

            if len(function_calls) > 1:
                error_message = (
                    "Only a single tool call is supported per step"
                )
                state.status = AgentStatus.FAILED
                state.termination_reason = (
                    AgentTerminationReason.TOOL_ERROR
                )
                state.error = error_message
                raise AgentExecutionError(
                    error_message,
                    state,
                )

            function_call = function_calls[0]
            tool_name = function_call.name

            try:
                tool_arguments = ToolCallingService.validate_tool_call(
                    tool_name=tool_name,
                    tool_arguments=function_call.args,
                )
                cls._validate_allowed_tool(
                    agent_definition=agent_definition,
                    tool_name=tool_name,
                )
                tool_result = ToolCallingService.execute_tool(
                    db=db,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
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
                raise AgentExecutionError(
                    error_message,
                    state,
                ) from exc

            state.steps.append(
                AgentStep(
                    step=step_number,
                    tool_called=tool_name,
                    tool_arguments=tool_arguments,
                    tool_result=tool_result,
                    success=True,
                )
            )

            state.conversation.append(
                response.candidates[0].content
            )
            state.conversation.append(
                cls._build_tool_result_content(
                    tool_name=tool_name,
                    tool_result=tool_result,
                )
            )

        error_message = "Agent step limit exceeded"
        state.status = AgentStatus.FAILED
        state.termination_reason = (
            AgentTerminationReason.MAX_STEPS_EXCEEDED
        )
        state.error = error_message
        raise AgentExecutionError(
            error_message,
            state,
        )

    @classmethod
    def chat(
        cls,
        db: Session,
        message: str,
        max_steps: int = 5,
    ) -> AgentResult:
        return cls.run(
            db=db,
            agent_definition=INCIDENT_ANALYSIS_AGENT,
            message=message,
            max_steps=max_steps,
        )
