from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.rag import (
    RagAnalyzeRequest,
    RagAnalyzeResponse,
)
from backend.app.services.rag_service import RagService


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
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
