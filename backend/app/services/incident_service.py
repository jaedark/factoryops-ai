from sqlalchemy.orm import Session

from backend.app.data.sample_incidents import SAMPLE_INCIDENTS
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

    @staticmethod
    def get_equipment_incidents(
        db: Session,
        equipment_name: str,
    ) -> list[Incident]:
        return IncidentRepository.get_by_equipment_name(
            db,
            equipment_name,
        )

    @staticmethod
    def search_incidents(
        db: Session,
        keyword: str,
    ) -> list[Incident]:
        return IncidentRepository.search_by_keyword(
            db,
            keyword,
        )

    @staticmethod
    def seed_incidents(db: Session) -> dict[str, int]:
        created = 0
        skipped = 0

        for item in SAMPLE_INCIDENTS:
            validated = IncidentCreate.model_validate(item)

            existing = IncidentRepository.get_by_equipment_and_occurred_at(
                db,
                validated.equipment_name,
                validated.occurred_at,
            )

            if existing:
                skipped += 1
                continue

            incident = Incident(
                **validated.model_dump()
            )

            IncidentRepository.create(
                db,
                incident,
            )

            created += 1

        return {
            "created": created,
            "skipped": skipped,
        }
