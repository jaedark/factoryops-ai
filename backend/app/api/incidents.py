from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.incident import Incident
from backend.app.schemas.incident import (
    IncidentCreate,
    IncidentResponse,
)
from backend.app.services.incident_service import IncidentService


router = APIRouter(
    prefix="/incidents",
    tags=["incidents"],
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
    db: Session = Depends(get_db),
) -> list[Incident]:
    return IncidentService.get_incidents(db)


@router.get(
    "/search",
    response_model=list[IncidentResponse],
)
def search_incidents(
    keyword: str,
    db: Session = Depends(get_db),
) -> list[Incident]:
    return IncidentService.search_incidents(
        db,
        keyword,
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentResponse,
)
def get_incident(
    incident_id: int,
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
