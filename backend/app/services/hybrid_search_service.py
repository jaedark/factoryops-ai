import re

from sqlalchemy.orm import Session

from backend.app.repositories.incident_repository import (
    IncidentRepository,
)
from backend.app.services.embedding_service import (
    EmbeddingService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


class HybridSearchService:
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Keep tokenization simple but compatible with both
        # equipment ids and Korean natural-language phrases.
        return re.findall(
            r"[a-zA-Z0-9_-]+|[媛-??+",
            text.lower(),
        )

    @classmethod
    def _calculate_keyword_score(
        cls,
        query: str,
        incident,
    ) -> float:
        query = query.strip().lower()

        if not query:
            return 0.0

        equipment_name = incident.equipment_name.lower()

        if query == equipment_name:
            return 1.0

        query_tokens = cls._tokenize(query)

        # If the equipment id appears as a token, treat it as
        # the strongest possible keyword signal.
        if equipment_name in query_tokens:
            return 1.0

        fields = [
            incident.equipment_name,
            incident.process_name,
            incident.symptom,
            incident.cause,
            incident.action_taken,
            incident.result,
        ]

        searchable_text = " ".join(
            field.lower()
            for field in fields
            if field
        )

        matched_tokens = 0

        for token in query_tokens:
            if token in searchable_text:
                matched_tokens += 1

        if matched_tokens == 0:
            return 0.0

        # Use a matched-token ratio so the keyword score stays
        # easy to reason about in experiments.
        return matched_tokens / len(query_tokens)

    @classmethod
    def search(
        cls,
        db: Session,
        query: str,
        top_k: int = 3,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> list[dict]:

        # Hybrid retrieval combines vector similarity with a
        # lightweight lexical score from keyword matching.
        vector_results = VectorSearchService.search(
            db=db,
            query=query,
            top_k=100,
        )

        incidents = IncidentRepository.get_all(db)

        vector_score_map = {
            result["incident"].incident_id: result["similarity"]
            for result in vector_results
        }

        results = []

        for incident in incidents:

            keyword_score = cls._calculate_keyword_score(
                query=query,
                incident=incident,
            )

            vector_score = vector_score_map.get(
                incident.incident_id,
                0.0,
            )

            hybrid_score = (
                keyword_score * keyword_weight
                + vector_score * vector_weight
            )

            results.append(
                {
                    "incident": incident,
                    "keyword_score": keyword_score,
                    "vector_score": vector_score,
                    "hybrid_score": hybrid_score,
                }
            )

        results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        return results[:top_k]
