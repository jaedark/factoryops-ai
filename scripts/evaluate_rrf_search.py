from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal
from backend.app.services.hybrid_search_service import (
    HybridSearchService,
)
from backend.app.services.rrf_search_service import (
    RrfSearchService,
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

            hybrid_results = HybridSearchService.search(
                db=db,
                query=query,
                top_k=3,
            )

            print("[Weighted Hybrid]")

            for index, result in enumerate(
                hybrid_results,
                start=1,
            ):
                incident = result["incident"]

                print(
                    f"{index}. "
                    f"{incident.equipment_name} / "
                    f"hybrid={result['hybrid_score']:.4f}"
                )

            print()

            rrf_results = RrfSearchService.search(
                db=db,
                query=query,
                top_k=3,
            )

            print("[RRF]")

            for index, result in enumerate(
                rrf_results,
                start=1,
            ):
                incident = result["incident"]

                print(
                    f"{index}. "
                    f"{incident.equipment_name} / "
                    f"keyword_rank={result['keyword_rank']} / "
                    f"vector_rank={result['vector_rank']} / "
                    f"rrf={result['rrf_score']:.6f}"
                )

            print()

    finally:
        db.close()


if __name__ == "__main__":
    main()
