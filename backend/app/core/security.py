import secrets

from fastapi import Security
from fastapi.security import APIKeyHeader

from backend.app.api.errors import ApiException
from backend.app.core.config import get_settings


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="FactoryAgentApiKey",
    description="Factory Agent protected API access key",
    auto_error=False,
)


def require_api_key(
    supplied_api_key: str | None = Security(api_key_header),
) -> None:
    configured_api_key = get_settings().get_factory_agent_api_key()
    if configured_api_key is None:
        raise ApiException(
            status_code=503,
            code="API_AUTH_NOT_CONFIGURED",
            message="API authentication is not configured",
        )
    if supplied_api_key is None or not supplied_api_key.strip():
        raise ApiException(
            status_code=401,
            code="AUTHENTICATION_REQUIRED",
            message="X-API-Key header is required",
        )
    if not secrets.compare_digest(supplied_api_key, configured_api_key):
        raise ApiException(
            status_code=401,
            code="INVALID_API_KEY",
            message="API key is invalid",
        )
