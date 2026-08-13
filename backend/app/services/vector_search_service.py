from sqlalchemy.orm import Session

from backend.app.models.incident import Incident
from backend.app.repositories.incident_repository import IncidentRepository
from backend.app.services.embedding_service import EmbeddingService


class VectorSearchService:
    @staticmethod
    def _build_incident_text(
        incident: Incident,
    ) -> str:
        return (
            f"Equipment: {incident.equipment_name}\n"
            f"Process: {incident.process_name}\n"
            f"Symptom: {incident.symptom}\n"
            f"Cause: {incident.cause or ''}\n"
            f"Action: {incident.action_taken or ''}\n"
            f"Result: {incident.result or ''}"
        )

    @classmethod
    def search(
        cls,
        db: Session,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        incidents = IncidentRepository.get_all(db)

        if not incidents:
            return []

        query_vector = EmbeddingService.embed_text(query)

        results = []

        for incident in incidents:
            incident_text = cls._build_incident_text(
                incident
            )

            incident_vector = EmbeddingService.embed_text(
                incident_text
            )

            similarity = (
                EmbeddingService.calculate_similarity(
                    query_vector,
                    incident_vector,
                )
            )

            results.append(
                {
                    "incident": incident,
                    "similarity": similarity,
                }
            )

        results.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return results[:top_k]
