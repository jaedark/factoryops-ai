from fastapi import FastAPI
from sqlalchemy import text

from backend.app.api.agent import router as agent_router
from backend.app.api import rag
from backend.app.api.admin import router as admin_router
from backend.app.api.incidents import router as incidents_router
from backend.app.api.tools import router as tools_router
from backend.app.api.errors import ApiException, register_exception_handlers
from backend.app.api.middleware import RequestContextMiddleware
from backend.app.core.config import get_settings, validate_startup_settings
from backend.app.core.database import Base, engine
from backend.app.models import incident
from backend.app.schemas.api import ApiErrorResponse


validate_startup_settings()
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="FactoryOps AI API",
    description="제조 장애 대응 및 지식 검색 API",
    version="0.1.0",
)
app.add_middleware(RequestContextMiddleware)
register_exception_handlers(app)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/ready",
    responses={503: {"model": ApiErrorResponse}},
)
def readiness_check() -> dict[str, str]:
    settings = get_settings()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise ApiException(
            status_code=503,
            code="SERVICE_NOT_READY",
            message="Service is not ready",
        ) from exc

    if settings.app_env.lower() in {"production", "prod"}:
        has_gemini_key = (
            settings.gemini_api_key is not None
            and bool(settings.gemini_api_key.get_secret_value().strip())
        )
        if not has_gemini_key or settings.get_factory_agent_api_key() is None:
            raise ApiException(
                status_code=503,
                code="SERVICE_NOT_READY",
                message="Service is not ready",
            )
    return {"status": "ready"}


app.include_router(admin_router)
app.include_router(incidents_router)
app.include_router(rag.router)
app.include_router(tools_router)
app.include_router(agent_router)
