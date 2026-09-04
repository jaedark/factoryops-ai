from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ObservabilityEventType(str, Enum):
    AGENT_STARTED = "agent_started"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    LLM_CALL_FAILED = "llm_call_failed"
    RETRY_SCHEDULED = "retry_scheduled"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_REJECTED = "circuit_rejected"
    APPROVAL_REQUIRED = "approval_required"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"


class ObservabilityEvent(BaseModel):
    trace_id: str = Field(min_length=1)
    event_type: ObservabilityEventType
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    agent_name: str = Field(min_length=1)
    step: int | None = None
    tool_name: str | None = None
    latency_ms: float | None = None
    success: bool | None = None
    status: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
