from sqlalchemy.orm import Session

from backend.app.models.incident import Incident
from backend.app.repositories.incident_repository import IncidentRepository
from backend.app.schemas.incident import IncidentCreate


class IncidentService:
    @staticmethod
    def create_incident(
        db: Session,
        incident_data: IncidentCreate,
    ) -> Incident:
        incident = Incident(
            **incident_data.model_dump()
        )

        return IncidentRepository.create(
            db,
            incident,
        )

    @staticmethod
    def get_incidents(
        db: Session,
    ) -> list[Incident]:
        return IncidentRepository.get_all(db)

    @staticmethod
    def get_incident(
        db: Session,
        incident_id: int,
    ) -> Incident | None:
        return IncidentRepository.get_by_id(
            db,
            incident_id,
        )