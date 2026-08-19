from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal
from backend.app.services.hybrid_search_service import (
    HybridSearchService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


def main():
    db: Session = SessionLocal()

    queries = [
        "장비가 너무 뜨거워졌어",
        "Conveyor-01",
        "Robot-01 위치 문제",
        "Sensor-05",
    ]

    try:
        for query in queries:

            print("=" * 70)
            print(f"[Query] {query}")
            print()

            vector_results = VectorSearchService.search(
                db=db,
                query=query,
                top_k=3,
            )

            print("[Vector Search]")

            for index, result in enumerate(
                vector_results,
                start=1,
            ):
                incident = result["incident"]

                print(
                    f"{index}. "
                    f"{incident.equipment_name} / "
                    f"{incident.symptom} / "
                    f"{result['similarity']:.4f}"
                )

            print()

            hybrid_results = HybridSearchService.search(
                db=db,
                query=query,
                top_k=3,
            )

            print("[Hybrid Search]")

            for index, result in enumerate(
                hybrid_results,
                start=1,
            ):
                incident = result["incident"]

                print(
                    f"{index}. "
                    f"{incident.equipment_name} / "
                    f"{incident.symptom} / "
                    f"keyword={result['keyword_score']:.4f} / "
                    f"vector={result['vector_score']:.4f} / "
                    f"hybrid={result['hybrid_score']:.4f}"
                )

            print()

    finally:
        db.close()


if __name__ == "__main__":
    main()
