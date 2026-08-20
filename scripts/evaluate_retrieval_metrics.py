from backend.app.core.database import SessionLocal
from backend.app.services.rrf_search_service import (
    RrfSearchService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


EVALUATION_CASES = [
    {
        "query": "장비가 너무 뜨거워졌어",
        "expected_equipment": "Conveyor-01",
    },
    {
        "query": "Conveyor-01 냉각 문제",
        "expected_equipment": "Conveyor-01",
    },
    {
        "query": "Robot-01 위치가 이상해",
        "expected_equipment": "Robot-01",
    },
    {
        "query": "Sensor-05 신호 문제",
        "expected_equipment": "Sensor-05",
    },
    {
        "query": "센서가 가끔 감지를 못해",
        "expected_equipment": "Sensor-05",
    },
]


HARD_EVALUATION_CASES = [
    {
        "query": "모터 쪽 열이 계속 올라가는데 냉각 계통부터 봐야 할까?",
        "expected_equipment": "Conveyor-01",
    },
    {
        "query": "로봇이 명령한 위치랑 실제 움직이는 위치가 계속 달라",
        "expected_equipment": "Robot-01",
    },
    {
        "query": "계속 고장난 건 아닌데 가끔 센서 입력이 빠져",
        "expected_equipment": "Sensor-05",
    },
    {
        "query": "카메라 전원은 살아있는 것 같은데 영상이 안 들어와",
        "expected_equipment": "Vision-02",
    },
    {
        "query": "컨베이어가 갑자기 멈추고 다시 켜도 반응이 없어",
        "expected_equipment": "Conveyor-02",
    },
    {
        "query": "검사 결과가 자꾸 틀리는데 비전 쪽 정확도가 떨어진 것 같아",
        "expected_equipment": "Vision-02",
    },
    {
        "query": "PLC랑 통신이 자꾸 끊겨서 설비 상태를 못 읽어와",
        "expected_equipment": "PLC-01",
    },
    {
        "query": "서보 알람이 떠서 로봇 축이 제대로 안 돈다",
        "expected_equipment": "Robot-02",
    },
    {
        "query": "제품이 지나가도 검출이 들쭉날쭉해서 누락이 생겨",
        "expected_equipment": "Sensor-03",
    },
    {
        "query": "영상 검사 결과가 예전보다 부정확하고 판정이 흔들려",
        "expected_equipment": "Vision-02",
    },
]


def find_rank(results, expected_equipment: str) -> int | None:
    for rank, result in enumerate(results, start=1):
        incident = result["incident"]

        if incident.equipment_name == expected_equipment:
            return rank

    return None


def calculate_metrics(ranks: list[int | None]) -> dict:
    total = len(ranks)

    hit_at_1 = sum(
        1 for rank in ranks
        if rank == 1
    ) / total

    hit_at_3 = sum(
        1 for rank in ranks
        if rank is not None and rank <= 3
    ) / total

    reciprocal_ranks = [
        1 / rank if rank is not None else 0
        for rank in ranks
    ]

    mrr = sum(reciprocal_ranks) / total

    return {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": mrr,
    }


def evaluate_cases(db, cases, label: str):
    vector_ranks = []
    rrf_ranks = []

    print("=" * 80)
    print(f"[{label}]")

    for case in cases:
        query = case["query"]
        expected = case["expected_equipment"]

        # This baseline script compares retrievers directly
        # without any reranking stage.
        vector_results = VectorSearchService.search(
            db=db,
            query=query,
            top_k=3,
        )

        rrf_results = RrfSearchService.search(
            db=db,
            query=query,
            top_k=3,
        )

        vector_rank = find_rank(
            vector_results,
            expected,
        )

        rrf_rank = find_rank(
            rrf_results,
            expected,
        )

        vector_ranks.append(vector_rank)
        rrf_ranks.append(rrf_rank)

        print("-" * 80)
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Vector Rank: {vector_rank}")
        print(f"RRF Rank: {rrf_rank}")
        print()

    vector_metrics = calculate_metrics(vector_ranks)
    rrf_metrics = calculate_metrics(rrf_ranks)

    print(f"[{label} Vector Search Metrics]")
    print(
        f"Hit@1: {vector_metrics['hit_at_1']:.2%}"
    )
    print(
        f"Hit@3: {vector_metrics['hit_at_3']:.2%}"
    )
    print(
        f"MRR: {vector_metrics['mrr']:.4f}"
    )
    print()
    print(f"[{label} RRF Metrics]")
    print(
        f"Hit@1: {rrf_metrics['hit_at_1']:.2%}"
    )
    print(
        f"Hit@3: {rrf_metrics['hit_at_3']:.2%}"
    )
    print(
        f"MRR: {rrf_metrics['mrr']:.4f}"
    )
    print()


def main():
    db = SessionLocal()

    try:
        evaluate_cases(
            db=db,
            cases=EVALUATION_CASES,
            label="Base Evaluation Cases",
        )
        evaluate_cases(
            db=db,
            cases=HARD_EVALUATION_CASES,
            label="Hard Evaluation Cases",
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
