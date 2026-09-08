from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import require_api_key
from backend.app.api.errors import PROTECTED_ROUTE_RESPONSES
from backend.app.models.incident import Incident
from backend.app.schemas.incident import (
    IncidentCreate,
    IncidentResponse,
)
from backend.app.schemas.vector_search import VectorSearchResult
from backend.app.services.incident_service import IncidentService
from backend.app.services.vector_search_service import VectorSearchService


router = APIRouter(
    prefix="/incidents",
    tags=["incidents"],
    dependencies=[Depends(require_api_key)],
    responses=PROTECTED_ROUTE_RESPONSES,
)


@router.post(
    "",
    response_model=IncidentResponse,
)
def create_incident(
    incident: IncidentCreate,
    db: Session = Depends(get_db),
) -> Incident:
    return IncidentService.create_incident(
        db,
        incident,
    )


@router.get(
    "",
    response_model=list[IncidentResponse],
)
def get_incidents(
    limit: int = Query(default=100, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[Incident]:
    return IncidentService.get_incidents(db, limit=limit)


@router.get(
    "/search",
    response_model=list[IncidentResponse],
)
def search_incidents(
    keyword: str = Query(min_length=1, max_length=1000),
    limit: int = Query(default=100, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[Incident]:
    return IncidentService.search_incidents(
        db,
        keyword,
        limit=limit,
    )


@router.get(
    "/vector-search",
    response_model=list[VectorSearchResult],
)
def vector_search_incidents(
    query: str = Query(min_length=1, max_length=1000),
    top_k: int = Query(
        default=3,
        ge=1,
        le=10,
    ),
    db: Session = Depends(get_db),
) -> list[dict]:
    return VectorSearchService.search(
        db,
        query,
        top_k,
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentResponse,
)
def get_incident(
    incident_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> Incident:
    incident = IncidentService.get_incident(
        db,
        incident_id,
    )

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return incident
