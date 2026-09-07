from time import perf_counter

from backend.app.observability.models import (
    ObservabilityEvent,
    ObservabilityEventType,
)
from backend.app.observability.sink import (
    ObservabilitySink,
)


class ObservabilityRecorder:
    def __init__(
        self,
        sink: ObservabilitySink,
        trace_id: str,
        agent_name: str,
    ) -> None:
        self.sink = sink
        self.trace_id = trace_id
        self.agent_name = agent_name

    def emit(
        self,
        event_type: ObservabilityEventType,
        *,
        step: int | None = None,
        tool_name: str | None = None,
        latency_ms: float | None = None,
        success: bool | None = None,
        status: str | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.sink.emit(
            ObservabilityEvent(
                trace_id=self.trace_id,
                event_type=event_type,
                agent_name=self.agent_name,
                step=step,
                tool_name=tool_name,
                latency_ms=latency_ms,
                success=success,
                status=status,
                error=error,
                metadata=metadata or {},
            )
        )

    def agent_failed(
        self,
        state,
        started_at: float,
    ) -> None:
        self.emit(
            ObservabilityEventType.AGENT_FAILED,
            step=state.current_step or None,
            latency_ms=(perf_counter() - started_at) * 1000,
            success=False,
            status=state.status.value,
            error=state.error,
            metadata={
                "termination_reason": (
                    state.termination_reason.value
                    if state.termination_reason is not None
                    else None
                ),
                "step_count": len(state.steps),
            },
        )

    @staticmethod
    def tool_argument_metadata(
        tool_arguments: dict,
    ) -> dict:
        return {
            "argument_keys": sorted(tool_arguments.keys()),
            "argument_count": len(tool_arguments),
        }
