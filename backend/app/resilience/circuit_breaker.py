from enum import Enum
from time import monotonic


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        time_fn=monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._time_fn = time_fn
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(
        self,
    ) -> str:
        if self._state == CircuitState.OPEN:
            if (
                self._opened_at is not None
                and (
                    self._time_fn() - self._opened_at
                ) >= self.recovery_timeout_seconds
            ):
                self._state = CircuitState.HALF_OPEN
                return self._state.value
            raise CircuitOpenError(
                "Circuit breaker is open"
            )

        return self._state.value

    def record_success(
        self,
    ) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None

    def record_failure(
        self,
    ) -> str:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = self._time_fn()
            self._failure_count = self.failure_threshold
            return self._state.value

        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._time_fn()

        return self._state.value

    def reset(
        self,
    ) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
