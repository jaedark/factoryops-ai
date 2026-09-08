import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import ConfigurationError
from backend.app.schemas.api import ApiError, ApiErrorResponse


logger = logging.getLogger(__name__)


class ApiException(HTTPException):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "req-unavailable")


def create_error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _get_request_id(request)
    response_headers = {"X-Request-ID": request_id}
    if headers:
        response_headers.update(headers)
    body = ApiErrorResponse(
        error=ApiError(
            code=code,
            message=message,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers=response_headers,
    )


def _http_error_code(status_code: int) -> str:
    return {
        400: "INVALID_REQUEST",
        401: "AUTHENTICATION_REQUIRED",
        404: "RESOURCE_NOT_FOUND",
        409: "STATE_CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        422: "VALIDATION_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }.get(status_code, "HTTP_ERROR")


def _safe_http_message(exc: StarletteHTTPException) -> str:
    if exc.status_code >= 500:
        if exc.status_code == 503:
            return "Service is unavailable"
        return "Internal server error"
    if isinstance(exc.detail, str) and exc.detail:
        return exc.detail
    return "Request failed"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        code = exc.code if isinstance(exc, ApiException) else _http_error_code(
            exc.status_code
        )
        message = (
            str(exc.detail)
            if isinstance(exc, ApiException)
            else _safe_http_message(exc)
        )
        return create_error_response(
            request,
            status_code=exc.status_code,
            code=code,
            message=message,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return create_error_response(
            request,
            status_code=422,
            code="VALIDATION_ERROR",
            message="Request validation failed",
        )

    @app.exception_handler(ConfigurationError)
    async def handle_configuration_error(
        request: Request,
        exc: ConfigurationError,
    ) -> JSONResponse:
        return create_error_response(
            request,
            status_code=503,
            code="SERVICE_CONFIGURATION_ERROR",
            message="Service configuration is invalid",
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.error(
            "Unhandled API error",
            extra={
                "request_id": _get_request_id(request),
                "error_type": type(exc).__name__,
            },
        )
        return create_error_response(
            request,
            status_code=500,
            code="INTERNAL_ERROR",
            message="Internal server error",
        )


PROTECTED_ROUTE_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ApiErrorResponse, "description": "API key authentication failed"},
    413: {"model": ApiErrorResponse, "description": "Request body is too large"},
    422: {"model": ApiErrorResponse, "description": "Request validation failed"},
    503: {"model": ApiErrorResponse, "description": "Service configuration is unavailable"},
}
