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