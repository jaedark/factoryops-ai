from pathlib import Path
import sys

from sqlalchemy.orm import Session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal
from backend.app.services.rag_context_service import RagContextService
from backend.app.services.vector_search_service import VectorSearchService


def main():
    db: Session = SessionLocal()

    try:
        query = "장비가 너무 뜨거워졌어"

        search_results = VectorSearchService.search(
            db=db,
            query=query,
            top_k=3,
        )

        context = RagContextService.build_context(search_results)

        print("=== RAG Context Test ===")
        print()
        print("[User Query]")
        print(query)
        print()
        print("[Retrieved Context]")
        print(context)

    finally:
        db.close()


if __name__ == "__main__":
    main()
