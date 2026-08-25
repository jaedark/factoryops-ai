from fastapi import FastAPI

from backend.app.api.agent import router as agent_router
from backend.app.api import rag
from backend.app.api.admin import router as admin_router
from backend.app.api.incidents import router as incidents_router
from backend.app.api.tools import router as tools_router
from backend.app.core.database import Base, engine
from backend.app.models import incident


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="FactoryOps AI API",
    description="제조 장애 대응 및 지식 검색 API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(admin_router)
app.include_router(incidents_router)
app.include_router(rag.router)
app.include_router(tools_router)
app.include_router(agent_router)
