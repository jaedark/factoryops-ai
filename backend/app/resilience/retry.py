from backend.app.resilience.models import (
    ErrorCategory,
    ErrorClassification,
    RetryPolicy,
)


def classify_error(
    exc: Exception,
) -> ErrorClassification:
    message = str(exc).lower()

    if isinstance(exc, TimeoutError) or "timeout" in message:
        return ErrorClassification(
            retryable=True,
            category=ErrorCategory.TIMEOUT,
            reason=str(exc) or "Timeout error",
        )

    if isinstance(exc, ConnectionError) or any(
        keyword in message
        for keyword in [
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "network",
            "dns",
        ]
    ):
        return ErrorClassification(
            retryable=True,
            category=ErrorCategory.NETWORK,
            reason=str(exc) or "Network error",
        )

    if any(
        keyword in message
        for keyword in [
            "429",
            "rate limit",
            "quota",
            "resource exhausted",
        ]
    ):
        return ErrorClassification(
            retryable=True,
            category=ErrorCategory.RATE_LIMIT,
            reason=str(exc) or "Rate limit error",
        )

    if any(
        keyword in message
        for keyword in [
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        ]
    ):
        return ErrorClassification(
            retryable=True,
            category=ErrorCategory.SERVER_ERROR,
            reason=str(exc) or "Server error",
        )

    if any(
        keyword in message
        for keyword in [
            "invalid arguments for tool",
            "validation error",
            "field required",
        ]
    ):
        return ErrorClassification(
            retryable=False,
            category=ErrorCategory.INVALID_ARGUMENTS,
            reason=str(exc) or "Invalid arguments",
        )

    if any(
        keyword in message
        for keyword in [
            "unsupported tool requested",
            "tool not allowed for agent",
            "tool policy not found",
        ]
    ):
        return ErrorClassification(
            retryable=False,
            category=ErrorCategory.INVALID_TOOL,
            reason=str(exc) or "Invalid tool",
        )

    if any(
        keyword in message
        for keyword in [
            "permission",
            "forbidden",
            "unauthorized",
        ]
    ):
        return ErrorClassification(
            retryable=False,
            category=ErrorCategory.PERMISSION,
            reason=str(exc) or "Permission error",
        )

    if "human approval required" in message:
        return ErrorClassification(
            retryable=False,
            category=ErrorCategory.APPROVAL_REQUIRED,
            reason=str(exc) or "Approval required",
        )

    if "circuit breaker is open" in message:
        return ErrorClassification(
            retryable=False,
            category=ErrorCategory.CIRCUIT_OPEN,
            reason=str(exc) or "Circuit breaker is open",
        )

    return ErrorClassification(
        retryable=False,
        category=ErrorCategory.UNKNOWN,
        reason=str(exc) or exc.__class__.__name__,
    )


def calculate_backoff_delay(
    policy: RetryPolicy,
    attempt: int,
) -> float:
    delay = policy.base_delay_seconds * (
        policy.backoff_multiplier ** max(attempt - 1, 0)
    )
    return min(delay, policy.max_delay_seconds)
