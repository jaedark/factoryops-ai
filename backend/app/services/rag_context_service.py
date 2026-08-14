from backend.app.models.incident import Incident


class RagContextService:

    @staticmethod
    def build_context(search_results: list[dict]) -> str:
        context_parts = []

        for index, result in enumerate(search_results, start=1):
            incident: Incident = result["incident"]
            similarity = result["similarity"]

            incident_context = (
                f"[Incident {index}]\n"
                f"Equipment: {incident.equipment_name}\n"
                f"Process: {incident.process_name}\n"
                f"Symptom: {incident.symptom}\n"
                f"Cause: {incident.cause or 'Unknown'}\n"
                f"Action: {incident.action_taken or 'Unknown'}\n"
                f"Result: {incident.result or 'Unknown'}\n"
                f"Similarity: {similarity:.4f}"
            )

            context_parts.append(incident_context)

        return "\n\n".join(context_parts)
