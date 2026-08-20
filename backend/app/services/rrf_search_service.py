from sqlalchemy.orm import Session

from backend.app.repositories.incident_repository import (
    IncidentRepository,
)
from backend.app.services.hybrid_search_service import (
    HybridSearchService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


class RrfSearchService:
    @classmethod
    def search(
        cls,
        db: Session,
        query: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[dict]:

        incidents = IncidentRepository.get_all(db)

        if not incidents:
            return []

        # RRF needs full ranked lists from each retriever, so the
        # vector side is expanded to all incidents first.
        vector_results = VectorSearchService.search(
            db=db,
            query=query,
            top_k=len(incidents),
        )

        keyword_results = []

        for incident in incidents:
            # Reuse the keyword scoring logic from hybrid search
            # to build the lexical ranking list for fusion.
            keyword_score = (
                HybridSearchService._calculate_keyword_score(
                    query=query,
                    incident=incident,
                )
            )

            if keyword_score > 0:
                keyword_results.append(
                    {
                        "incident": incident,
                        "keyword_score": keyword_score,
                    }
                )

        keyword_results.sort(
            key=lambda item: item["keyword_score"],
            reverse=True,
        )

        scores = {}

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            incident = result["incident"]

            scores.setdefault(
                incident.incident_id,
                {
                    "incident": incident,
                    "rrf_score": 0.0,
                    "keyword_rank": None,
                    "vector_rank": None,
                },
            )

            scores[incident.incident_id]["keyword_rank"] = rank
            # Reciprocal Rank Fusion gives higher reward to items
            # appearing near the top of each ranking list.
            scores[incident.incident_id]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):
            incident = result["incident"]

            scores.setdefault(
                incident.incident_id,
                {
                    "incident": incident,
                    "rrf_score": 0.0,
                    "keyword_rank": None,
                    "vector_rank": None,
                },
            )

            scores[incident.incident_id]["vector_rank"] = rank
            scores[incident.incident_id]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

        results = list(scores.values())

        results.sort(
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return results[:top_k]
