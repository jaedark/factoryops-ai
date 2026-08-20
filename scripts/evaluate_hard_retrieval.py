from time import perf_counter

from backend.app.core.config import (
    RERANKER_MODEL_NAME,
    RERANK_FINAL_TOP_K,
    RERANK_RETRIEVER_TOP_N,
)
from backend.app.core.database import SessionLocal
from backend.app.services.reranker_service import (
    RerankerService,
)
from backend.app.services.rrf_search_service import (
    RrfSearchService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


HARD_EVALUATION_CASES = [
    {
        "query": "모터 쪽 열이 계속 올라가는데 냉각 계통부터 봐야 할까?",
        "expected_equipment": "Conveyor-01",
    },
    {
        "query": "카메라 전원은 살아있는 것 같은데 화면이 한참 안 들어와",
        "expected_equipment": "Vision-01",
    },
    {
        "query": "로봇이 무거운 작업하다가 축 쪽 경보 띄우고 멈췄어",
        "expected_equipment": "Robot-02",
    },
    {
        "query": "포장 라인 PLC 연결이 자꾸 끊겨서 상태를 못 읽어와",
        "expected_equipment": "PLC-01",
    },
    {
        "query": "계속 고장난 건 아닌데 가끔 센서 입력이 아예 안 잡혀",
        "expected_equipment": "Sensor-05",
    },
    {
        "query": "라인이 갑자기 서버렸는데 누가 급정지 눌렀던 것처럼 보여",
        "expected_equipment": "Conveyor-02",
    },
    {
        "query": "검사 결과가 자꾸 빗나가서 불량 판정 정확도가 떨어졌어",
        "expected_equipment": "Vision-02",
    },
    {
        "query": "로봇이 가르쳐 둔 좌표랑 실제 가는 위치가 계속 어긋나",
        "expected_equipment": "Robot-01",
    },
    {
        "query": "PLC 응답이 전보다 굼뜨고 주기 시간이 계속 늘어지는 느낌이야",
        "expected_equipment": "PLC-02",
    },
    {
        "query": "제품은 지나가는데 감지가 들쭉날쭉해서 카운트가 자꾸 빠져",
        "expected_equipment": "Sensor-03",
    },
]


FOCUS_FAILURE_QUERIES = [
    "계속 고장난 건 아닌데 가끔 센서 입력이 아예 안 잡혀",
    "검사 결과가 자꾸 빗나가서 불량 판정 정확도가 떨어졌어",
    "제품은 지나가는데 감지가 들쭉날쭉해서 카운트가 자꾸 빠져",
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


def print_top_results(
    label: str,
    results: list[dict],
) -> None:
    print(f"[{label} Top 3]")

    for rank, result in enumerate(results[:3], start=1):
        incident = result["incident"]
        detail = (
            f"{incident.equipment_name} / "
            f"{incident.symptom}"
        )

        if "similarity" in result:
            detail += (
                f" / similarity={result['similarity']:.4f}"
            )

        if "rrf_score" in result:
            detail += (
                f" / keyword_rank={result['keyword_rank']}"
                f" / vector_rank={result['vector_rank']}"
                f" / rrf={result['rrf_score']:.6f}"
            )

        if "rerank_score" in result:
            detail += (
                f" / rerank={result['rerank_score']:.4f}"
            )

        print(f"{rank}. {detail}")

    print()


def get_vector_results(db, query: str) -> list[dict]:
    return VectorSearchService.search(
        db=db,
        query=query,
        top_k=3,
    )


def get_rrf_results(db, query: str) -> list[dict]:
    return RrfSearchService.search(
        db=db,
        query=query,
        top_k=3,
    )


def get_vector_rerank_results(
    db,
    query: str,
) -> list[dict]:
    vector_candidates = VectorSearchService.search(
        db=db,
        query=query,
        top_k=RERANK_RETRIEVER_TOP_N,
    )

    return RerankerService.rerank_candidates(
        query=query,
        candidates=vector_candidates,
        top_k=RERANK_FINAL_TOP_K,
    )


def get_rrf_rerank_results(
    db,
    query: str,
) -> list[dict]:
    rrf_candidates = RrfSearchService.search(
        db=db,
        query=query,
        top_k=RERANK_RETRIEVER_TOP_N,
    )

    return RerankerService.rerank_candidates(
        query=query,
        candidates=rrf_candidates,
        top_k=RERANK_FINAL_TOP_K,
    )


def summarize_method(
    db,
    cases,
    label: str,
    search_fn,
) -> dict:
    ranks = []
    failures = []
    per_query = {}

    print("=" * 80)
    print(f"[{label}]")

    for index, case in enumerate(cases, start=1):
        query = case["query"]
        expected = case["expected_equipment"]
        results = search_fn(db, query)
        rank = find_rank(results, expected)

        ranks.append(rank)
        per_query[query] = {
            "expected": expected,
            "rank": rank,
            "results": results,
        }

        print("-" * 80)
        print(f"Case {index}")
        print(f"Query: {query}")
        print(f"Ground Truth: {expected}")
        print(f"Rank: {rank}")
        print()

        if rank != 1:
            failures.append(
                {
                    "query": query,
                    "expected": expected,
                    "rank": rank,
                    "results": results,
                }
            )

    metrics = calculate_metrics(ranks)

    print(f"[{label} Metrics]")
    print(f"Hit@1: {metrics['hit_at_1']:.2%}")
    print(f"Hit@3: {metrics['hit_at_3']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print()

    print(f"[{label} Failures]")

    if not failures:
        print("None")
        print()
    else:
        for failure in failures:
            print(f"Query: {failure['query']}")
            print(
                f"Ground Truth: {failure['expected']}"
            )
            print(
                f"Returned Rank: {failure['rank']}"
            )
            print_top_results(
                label=label,
                results=failure["results"],
            )

    return {
        "label": label,
        "metrics": metrics,
        "per_query": per_query,
    }


def print_before_after(summary_map: dict) -> None:
    print("=" * 80)
    print("[Before / After For Existing Failure Queries]")

    for query in FOCUS_FAILURE_QUERIES:
        vector_before = summary_map["Vector"]["per_query"][
            query
        ]["rank"]
        vector_after = summary_map[
            "Vector + Rerank"
        ]["per_query"][query]["rank"]
        rrf_before = summary_map["RRF"]["per_query"][
            query
        ]["rank"]
        rrf_after = summary_map["RRF + Rerank"][
            "per_query"
        ][query]["rank"]

        print("-" * 80)
        print(f"Query: {query}")
        print(
            f"Vector: {vector_before} -> {vector_after}"
        )
        print(f"RRF: {rrf_before} -> {rrf_after}")
        print()


def print_degraded_queries(summary_map: dict) -> None:
    degraded = []

    for case in HARD_EVALUATION_CASES:
        query = case["query"]

        vector_before = summary_map["Vector"]["per_query"][
            query
        ]["rank"]
        vector_after = summary_map[
            "Vector + Rerank"
        ]["per_query"][query]["rank"]

        if (
            vector_before is not None
            and vector_after is not None
            and vector_after > vector_before
        ):
            degraded.append(
                {
                    "query": query,
                    "pipeline": "Vector -> Vector + Rerank",
                    "before": vector_before,
                    "after": vector_after,
                }
            )

        rrf_before = summary_map["RRF"]["per_query"][
            query
        ]["rank"]
        rrf_after = summary_map["RRF + Rerank"][
            "per_query"
        ][query]["rank"]

        if (
            rrf_before is not None
            and rrf_after is not None
            and rrf_after > rrf_before
        ):
            degraded.append(
                {
                    "query": query,
                    "pipeline": "RRF -> RRF + Rerank",
                    "before": rrf_before,
                    "after": rrf_after,
                }
            )

    print("=" * 80)
    print("[Degraded Queries After Reranking]")

    if not degraded:
        print("None")
        print()
        return

    for item in degraded:
        print("-" * 80)
        print(f"Pipeline: {item['pipeline']}")
        print(f"Query: {item['query']}")
        print(
            f"Rank: {item['before']} -> {item['after']}"
        )
        print()


def main():
    db = SessionLocal()

    try:
        model_started_at = perf_counter()
        RerankerService.get_model()
        model_total_elapsed = (
            perf_counter() - model_started_at
        )

        evaluation_started_at = perf_counter()

        summary_map = {}

        summary_map["Vector"] = summarize_method(
            db=db,
            cases=HARD_EVALUATION_CASES,
            label="Vector",
            search_fn=get_vector_results,
        )

        summary_map["RRF"] = summarize_method(
            db=db,
            cases=HARD_EVALUATION_CASES,
            label="RRF",
            search_fn=get_rrf_results,
        )

        summary_map["Vector + Rerank"] = summarize_method(
            db=db,
            cases=HARD_EVALUATION_CASES,
            label="Vector + Rerank",
            search_fn=get_vector_rerank_results,
        )

        summary_map["RRF + Rerank"] = summarize_method(
            db=db,
            cases=HARD_EVALUATION_CASES,
            label="RRF + Rerank",
            search_fn=get_rrf_rerank_results,
        )

        evaluation_elapsed = (
            perf_counter() - evaluation_started_at
        )

        print_before_after(summary_map)
        print_degraded_queries(summary_map)

        print("=" * 80)
        print("[Reranker Settings]")
        print(
            f"Model: {RERANKER_MODEL_NAME}"
        )
        print(
            f"Retriever Top N: {RERANK_RETRIEVER_TOP_N}"
        )
        print(
            f"Final Top K: {RERANK_FINAL_TOP_K}"
        )
        print(
            f"Model Load Seconds: {RerankerService.get_model_load_seconds():.4f}"
        )
        print(
            f"Model Load Seconds Measured In Script: {model_total_elapsed:.4f}"
        )
        print(
            f"Evaluation Seconds: {evaluation_elapsed:.4f}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
