from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import require_api_key
from backend.app.services.incident_service import IncidentService
from backend.app.api.errors import PROTECTED_ROUTE_RESPONSES


router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_api_key)],
    responses=PROTECTED_ROUTE_RESPONSES,
)


@router.post("/seed")
def seed_incidents(
    db: Session = Depends(get_db),
) -> dict[str, int]:
    return IncidentService.seed_incidents(db)
