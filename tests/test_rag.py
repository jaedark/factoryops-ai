from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_rag_analyze_returns_answer_and_sources():
    client.post("/admin/seed")

    mock_answer = """
1. 예상 원인
냉각 팬 오작동

2. 우선 확인할 항목
냉각 팬 작동 상태

3. 권장 조치
냉각 팬 점검 또는 교체

4. 참고한 장애 이력
Conveyor-01
""".strip()

    with patch(
        "backend.app.services.rag_service.LlmService.generate",
        return_value=mock_answer,
    ):
        response = client.post(
            "/rag/analyze",
            json={
                "query": "장비가 너무 뜨거워졌어. 무엇을 확인해야 해?",
                "top_k": 3,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "장비가 너무 뜨거워졌어. 무엇을 확인해야 해?"
    assert data["answer"] == mock_answer

    assert len(data["sources"]) == 3

    assert data["sources"][0]["equipment_name"] == "Conveyor-01"
    assert (
        data["sources"][0]["symptom"]
        == "Motor temperature exceeded threshold"
    )


@patch(
    "backend.app.services.rag_service.VectorSearchService.search",
    return_value=[],
)
@patch(
    "backend.app.services.rag_service.LlmService.generate",
)
def test_rag_analyze_without_search_results(
    mock_generate,
    mock_search,
):
    response = client.post(
        "/rag/analyze",
        json={
            "query": "존재하지 않는 장애 상황",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "관련 장애 이력을 찾지 못했습니다."
    assert data["sources"] == []

    mock_generate.assert_not_called()


def test_rag_analyze_rejects_empty_query():
    response = client.post(
        "/rag/analyze",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_rag_analyze_applies_similarity_threshold():

    with patch(
        "backend.app.services.rag_service.LlmService.generate",
        return_value="mock answer",
    ):
        response = client.post(
            "/rag/analyze",
            json={
                "query": "장비가 너무 뜨거워졌어. 무엇을 확인해야 해?",
                "top_k": 3,
                "similarity_threshold": 0.4,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["sources"]) == 1
    assert data["sources"][0]["equipment_name"] == "Conveyor-01"
