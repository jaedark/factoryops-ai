from backend.app.core.database import SessionLocal
from backend.app.services.rrf_search_service import (
    RrfSearchService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


def print_vector_results(results):
    for index, result in enumerate(results, start=1):
        incident = result["incident"]

        print(
            f"{index}. "
            f"{incident.equipment_name} / "
            f"{incident.symptom} / "
            f"similarity={result['similarity']:.4f}"
        )


def print_rrf_results(results):
    for index, result in enumerate(results, start=1):
        incident = result["incident"]

        print(
            f"{index}. "
            f"{incident.equipment_name} / "
            f"{incident.symptom} / "
            f"keyword_rank={result['keyword_rank']} / "
            f"vector_rank={result['vector_rank']} / "
            f"rrf={result['rrf_score']:.6f}"
        )


def main():
    db = SessionLocal()

    queries = [
        "장비가 너무 뜨거워졌어",
        "Conveyor-01 냉각 문제",
        "Robot-01 위치가 이상해",
        "Sensor-05 신호 문제",
        "카메라 영상이 안 들어와",
    ]

    try:
        for query in queries:
            print("=" * 80)
            print(f"[Query] {query}")
            print()

            vector_results = VectorSearchService.search(
                db=db,
                query=query,
                top_k=3,
            )

            print("[Vector Retrieval]")
            print_vector_results(vector_results)

            print()

            rrf_results = RrfSearchService.search(
                db=db,
                query=query,
                top_k=3,
            )

            print("[RRF Retrieval]")
            print_rrf_results(rrf_results)

            print()

    finally:
        db.close()


if __name__ == "__main__":
    main()
