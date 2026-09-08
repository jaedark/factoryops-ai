from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.agent import (
    AgentStatus,
    AgentTerminationReason,
    ApprovalRequest,
)


class ToolChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=8000,
        description="Tool Calling으로 처리할 사용자 자연어 요청",
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ToolChatResponse(BaseModel):
    answer: str
    tool_called: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: Any | None = None
    status: AgentStatus | None = None
    termination_reason: AgentTerminationReason | None = None
    approval_request: ApprovalRequest | None = None
