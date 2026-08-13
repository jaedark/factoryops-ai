from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.app.models.incident import Incident


class IncidentRepository:
    @staticmethod
    def create(
        db: Session,
        incident: Incident,
    ) -> Incident:
        db.add(incident)
        db.commit()
        db.refresh(incident)

        return incident

    @staticmethod
    def get_all(
        db: Session,
    ) -> list[Incident]:
        return db.query(Incident).all()

    @staticmethod
    def get_by_id(
        db: Session,
        incident_id: int,
    ) -> Incident | None:
        return (
            db.query(Incident)
            .filter(
                Incident.incident_id == incident_id
            )
            .first()
        )

    @staticmethod
    def get_by_equipment_and_occurred_at(
        db: Session,
        equipment_name: str,
        occurred_at,
    ) -> Incident | None:
        return (
            db.query(Incident)
            .filter(
                Incident.equipment_name == equipment_name,
                Incident.occurred_at == occurred_at,
            )
            .first()
        )

    @staticmethod
    def search_by_keyword(
        db: Session,
        keyword: str,
    ) -> list[Incident]:
        search_pattern = f"%{keyword}%"

        return (
            db.query(Incident)
            .filter(
                or_(
                    Incident.equipment_name.ilike(search_pattern),
                    Incident.process_name.ilike(search_pattern),
                    Incident.symptom.ilike(search_pattern),
                    Incident.cause.ilike(search_pattern),
                    Incident.action_taken.ilike(search_pattern),
                    Incident.result.ilike(search_pattern),
                )
            )
            .all()
        )
