from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import get_settings


class AgentChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=8000,
        description="Agent Loop로 처리할 사용자 자연어 요청",
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "선택적 세션 ID. 제공되면 이전 대화를 short-term "
            "memory로 불러와 현재 요청 context에 포함한다."
        ),
    )
    max_steps: int = Field(
        default_factory=lambda: get_settings().agent_max_steps,
        ge=1,
        le=10,
        description="Agent가 tool을 실행할 수 있는 최대 횟수",
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("session_id must not be blank")
        return normalized


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"


class AgentTerminationReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TOOL_ERROR = "tool_error"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    LLM_ERROR = "llm_error"
    APPROVAL_REQUIRED = "approval_required"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    EXECUTION_FAILED = "execution_failed"
    REJECTED = "rejected"


class ToolPolicy(BaseModel):
    tool_name: str = Field(min_length=1)
    risk_level: ToolRiskLevel
    approval_required: bool


class GuardrailDecision(BaseModel):
    allowed_to_execute: bool
    approval_required: bool
    risk_level: ToolRiskLevel
    reason: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    approval_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_arguments: dict[str, Any]
    status: ApprovalStatus
    reason: str = Field(min_length=1)
    session_id: str | None = None


class AgentStep(BaseModel):
    step: int
    tool_called: str
    tool_arguments: dict[str, Any]
    tool_result: Any = None
    success: bool
    error: str | None = None
    approval_required: bool = False
    approval_id: str | None = None


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_id: str = Field(min_length=1)
    conversation: list[Any]
    steps: list[AgentStep]
    current_step: int
    max_steps: int
    status: AgentStatus
    termination_reason: AgentTerminationReason | None = None
    final_answer: str | None = None
    error: str | None = None
    approval_request: ApprovalRequest | None = None


class MemoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ConversationMemory(BaseModel):
    session_id: str = Field(min_length=1)
    messages: list[MemoryMessage]


class AgentChatResponse(BaseModel):
    trace_id: str
    answer: str
    steps: list[AgentStep]
    total_steps: int
    status: AgentStatus
    termination_reason: AgentTerminationReason
    approval_request: ApprovalRequest | None = None


class ApprovalActionResponse(BaseModel):
    approval_request: ApprovalRequest
    tool_result: Any = None
