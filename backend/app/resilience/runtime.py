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

    @classmethod
    def from_settings(
        cls,
        settings,
        *,
        max_steps: int | None = None,
        llm_timeout_seconds: float | None = None,
        tool_timeout_seconds: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> "AgentExecutionConfig":
        return cls(
            max_steps=(
                max_steps
                if max_steps is not None
                else settings.agent_max_steps
            ),
            llm_timeout_seconds=(
                llm_timeout_seconds
                if llm_timeout_seconds is not None
                else settings.llm_timeout_seconds
            ),
            tool_timeout_seconds=(
                tool_timeout_seconds
                if tool_timeout_seconds is not None
                else settings.tool_timeout_seconds
            ),
            retry_policy=(
                retry_policy or RetryPolicy.from_settings(settings)
            ),
        )


@dataclass(frozen=True)
class AgentRuntimeContext:
    trace_id: str
    recorder: ObservabilityRecorder
    config: AgentExecutionConfig
    llm_circuit_breaker: CircuitBreaker
    sleep_fn: Callable[[float], None]
    approval_store: object | None = None
