from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_seed_incidents():
    response = client.post("/admin/seed")

    assert response.status_code == 200

    data = response.json()

    assert "created" in data
    assert "skipped" in data
    assert data["created"] >= 0
    assert data["skipped"] >= 0


def test_seed_is_idempotent():
    client.post("/admin/seed")

    second_response = client.post("/admin/seed")

    assert second_response.status_code == 200

    data = second_response.json()

    assert data["created"] == 0
    assert data["skipped"] >= 10


def test_search_temperature():
    client.post("/admin/seed")

    response = client.get(
        "/incidents/search",
        params={"keyword": "temperature"},
    )

    assert response.status_code == 200

    results = response.json()

    assert len(results) >= 1
    assert any(
        item["equipment_name"] == "Conveyor-01"
        for item in results
    )


def test_search_vision():
    client.post("/admin/seed")

    response = client.get(
        "/incidents/search",
        params={"keyword": "Vision"},
    )

    assert response.status_code == 200

    results = response.json()

    equipment_names = {
        item["equipment_name"]
        for item in results
    }

    assert "Vision-01" in equipment_names
    assert "Vision-02" in equipment_names


def test_search_unknown_keyword_returns_empty_list():
    client.post("/admin/seed")

    response = client.get(
        "/incidents/search",
        params={
            "keyword": "THIS_KEYWORD_DOES_NOT_EXIST_999"
        },
    )

    assert response.status_code == 200
    assert response.json() == []


def test_keyword_search_cannot_understand_semantics():
    client.post("/admin/seed")

    response = client.get(
        "/incidents/search",
        params={"keyword": "장비가 뜨거워짐"},
    )

    assert response.status_code == 200
    assert response.json() == []
