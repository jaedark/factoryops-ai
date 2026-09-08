import logging
import re
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from backend.app.api.errors import create_error_response
from backend.app.core.config import get_settings


logger = logging.getLogger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _resolve_request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID")
    if supplied and _REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return f"req-{uuid4()}"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = _resolve_request_id(request)
        request.state.request_id = request_id
        started_at = time.perf_counter()
        status_code = 500

        try:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    request_bytes = int(content_length)
                except ValueError:
                    request_bytes = 0
                if request_bytes > get_settings().api_max_request_bytes:
                    status_code = 413
                    return create_error_response(
                        request,
                        status_code=413,
                        code="PAYLOAD_TOO_LARGE",
                        message="Request body is too large",
                    )

            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        finally:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "API request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": round(latency_ms, 3),
                },
            )
