from backend.app.core.database import SessionLocal
from backend.app.services.hybrid_search_service import (
    HybridSearchService,
)


def test_hybrid_search_exact_equipment_name():

    db = SessionLocal()

    try:
        results = HybridSearchService.search(
            db=db,
            query="Conveyor-01",
            top_k=3,
        )

        assert len(results) > 0

        assert (
            results[0]["incident"].equipment_name
            == "Conveyor-01"
        )

        assert results[0]["keyword_score"] == 1.0

    finally:
        db.close()


def test_hybrid_search_equipment_with_natural_language():

    db = SessionLocal()

    try:
        results = HybridSearchService.search(
            db=db,
            query="Robot-01 위치 문제",
            top_k=3,
        )

        assert len(results) > 0

        assert (
            results[0]["incident"].equipment_name
            == "Robot-01"
        )

        assert results[0]["keyword_score"] == 1.0

    finally:
        db.close()
