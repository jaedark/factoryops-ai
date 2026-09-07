from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from backend.app.observability import (
    ObservabilityRecorder,
)
from backend.app.resilience.circuit_breaker import (
    CircuitBreaker,
)
from backend.app.resilience.models import RetryPolicy


class AgentExecutionConfig(BaseModel):
    max_steps: int = Field(default=5, ge=1)
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0)
    tool_timeout_seconds: float = Field(default=5.0, gt=0.0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class AgentRuntimeContext:
    trace_id: str
    recorder: ObservabilityRecorder
    config: AgentExecutionConfig
    llm_circuit_breaker: CircuitBreaker
    sleep_fn: Callable[[float], None]
    approval_store: object | None = None
