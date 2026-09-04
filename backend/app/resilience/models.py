from enum import Enum

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    SERVER_ERROR = "server_error"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_TOOL = "invalid_tool"
    PERMISSION = "permission"
    APPROVAL_REQUIRED = "approval_required"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class ErrorClassification(BaseModel):
    retryable: bool
    category: ErrorCategory
    reason: str = Field(min_length=1)


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay_seconds: float = Field(default=0.1, ge=0.0)
    max_delay_seconds: float = Field(default=0.5, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
