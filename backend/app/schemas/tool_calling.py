from typing import Any

from pydantic import BaseModel, Field

from backend.app.schemas.agent import (
    AgentStatus,
    AgentTerminationReason,
    ApprovalRequest,
)


class ToolChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="Tool Calling으로 처리할 사용자 자연어 요청",
    )


class ToolChatResponse(BaseModel):
    answer: str
    tool_called: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: Any | None = None
    status: AgentStatus | None = None
    termination_reason: AgentTerminationReason | None = None
    approval_request: ApprovalRequest | None = None
