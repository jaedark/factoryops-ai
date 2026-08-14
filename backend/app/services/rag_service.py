from sqlalchemy.orm import Session

from backend.app.services.llm_service import LlmService
from backend.app.services.rag_context_service import RagContextService
from backend.app.services.rag_prompt_service import RagPromptService
from backend.app.services.vector_search_service import VectorSearchService


class RagService:

    @classmethod
    def analyze(
        cls,
        db: Session,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0,
    ) -> dict:

        search_results = VectorSearchService.search(
            db=db,
            query=query,
            top_k=top_k,
        )

        filtered_results = [
            result
            for result in search_results
            if result["similarity"] >= similarity_threshold
        ]

        if not filtered_results:
            return {
                "query": query,
                "answer": "관련 장애 이력을 찾지 못했습니다.",
                "sources": [],
            }

        context = RagContextService.build_context(
            filtered_results
        )

        prompt = RagPromptService.build_prompt(
            query=query,
            context=context,
        )

        answer = LlmService.generate(prompt)

        sources = []

        for result in filtered_results:
            incident = result["incident"]

            sources.append(
                {
                    "incident_id": incident.incident_id,
                    "equipment_name": incident.equipment_name,
                    "symptom": incident.symptom,
                    "similarity": result["similarity"],
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
        }
