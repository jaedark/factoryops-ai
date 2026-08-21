from backend.app.core.database import SessionLocal
from backend.app.services.rrf_search_service import (
    RrfSearchService,
)


def test_rrf_search_exact_equipment_name():
    db = SessionLocal()

    try:
        # Exact equipment-name queries should be dominated by
        # the keyword side of the fused ranking.
        results = RrfSearchService.search(
            db=db,
            query="Conveyor-01",
            top_k=3,
        )

        assert len(results) > 0

        assert (
            results[0]["incident"].equipment_name
            == "Conveyor-01"
        )

        assert results[0]["keyword_rank"] == 1

    finally:
        db.close()


def test_rrf_search_combines_keyword_and_vector_rank():
    db = SessionLocal()

    try:
        # A natural-language issue query should still rank the
        # correct incident first after fusion.
        results = RrfSearchService.search(
            db=db,
            query="Robot-01 ?꾩튂 臾몄젣",
            top_k=3,
        )

        assert len(results) > 0

        first = results[0]

        assert (
            first["incident"].equipment_name
            == "Robot-01"
        )

        assert first["keyword_rank"] == 1
        assert first["vector_rank"] is not None
        assert first["rrf_score"] > 0

    finally:
        db.close()
