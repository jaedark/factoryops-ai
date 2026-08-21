from typing import Any

from pydantic import BaseModel, Field


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
