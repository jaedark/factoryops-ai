from pathlib import Path
import sys

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


client = TestClient(app)


TEST_QUERIES = [
    "장비가 너무 뜨거워졌어",
    "카메라 영상이 안 들어와",
    "로봇 위치가 이상해",
    "센서가 가끔 감지를 못해",
]


def main():
    client.post("/admin/seed")

    print("=== Keyword vs Vector Search Evaluation ===")
    print()

    for query in TEST_QUERIES:
        keyword_response = client.get(
            "/incidents/search",
            params={"keyword": query},
        )

        vector_response = client.get(
            "/incidents/vector-search",
            params={
                "query": query,
                "top_k": 3,
            },
        )

        keyword_results = keyword_response.json()
        vector_results = vector_response.json()

        print(f"[Query] {query}")

        print("Keyword Search:")
        if keyword_results:
            for item in keyword_results:
                print(
                    f"  - {item['equipment_name']} / "
                    f"{item['symptom']}"
                )
        else:
            print("  - 결과 없음")

        print("Vector Search:")
        for index, item in enumerate(
            vector_results,
            start=1,
        ):
            incident = item["incident"]
            similarity = item["similarity"]

            print(
                f"  {index}. "
                f"{incident['equipment_name']} / "
                f"{incident['symptom']} / "
                f"{similarity:.4f}"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()
