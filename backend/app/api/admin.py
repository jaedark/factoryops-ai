from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.services.incident_service import IncidentService


router = APIRouter(
    prefix="/admin",
    tags=["admin"],
)


@router.post("/seed")
def seed_incidents(
    db: Session = Depends(get_db),
) -> dict[str, int]:
    return IncidentService.seed_incidents(db)
