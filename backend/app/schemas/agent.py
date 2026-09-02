from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="Agent Loop로 처리할 사용자 자연어 요청",
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "선택적 세션 ID. 제공되면 이전 대화를 short-term "
            "memory로 불러와 현재 요청 context에 포함한다."
        ),
    )
    max_steps: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Agent가 tool을 실행할 수 있는 최대 횟수",
    )


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTerminationReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TOOL_ERROR = "tool_error"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    LLM_ERROR = "llm_error"


class AgentStep(BaseModel):
    step: int
    tool_called: str
    tool_arguments: dict[str, Any]
    tool_result: Any = None
    success: bool
    error: str | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation: list[Any]
    steps: list[AgentStep]
    current_step: int
    max_steps: int
    status: AgentStatus
    termination_reason: AgentTerminationReason | None = None
    final_answer: str | None = None
    error: str | None = None


class MemoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ConversationMemory(BaseModel):
    session_id: str = Field(min_length=1)
    messages: list[MemoryMessage]


class AgentChatResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    total_steps: int
    status: AgentStatus
    termination_reason: AgentTerminationReason
