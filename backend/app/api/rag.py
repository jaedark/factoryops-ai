from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import require_api_key
from backend.app.api.errors import PROTECTED_ROUTE_RESPONSES
from backend.app.schemas.rag import (
    RagAnalyzeRequest,
    RagAnalyzeResponse,
)
from backend.app.services.rag_service import RagService


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
    dependencies=[Depends(require_api_key)],
    responses=PROTECTED_ROUTE_RESPONSES,
)


@router.post(
    "/analyze",
    response_model=RagAnalyzeResponse,
)
def analyze_incident(
    request: RagAnalyzeRequest,
    db: Session = Depends(get_db),
) -> dict:

    return RagService.analyze(
        db=db,
        query=request.query,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
    )
