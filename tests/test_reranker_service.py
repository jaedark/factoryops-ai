from backend.app.core.database import SessionLocal
from backend.app.services.reranker_service import (
    RerankerService,
)
from backend.app.services.vector_search_service import (
    VectorSearchService,
)


class FakeRerankerModel:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        # Mirror the real CrossEncoder interface while keeping
        # tests deterministic and offline.
        assert len(pairs) == len(self.scores)

        return self.scores


def test_build_rerank_text_includes_all_fields():
    db = SessionLocal()

    try:
        candidate = VectorSearchService.search(
            db=db,
            query="temperature",
            top_k=1,
        )[0]

        text = RerankerService.build_rerank_text(
            candidate
        )

        assert "Equipment:" in text
        assert "Process:" in text
        assert "Symptom:" in text
        assert "Cause:" in text
        assert "Action:" in text
        assert "Result:" in text

    finally:
        db.close()


def test_rerank_sorts_by_cross_encoder_score():
    db = SessionLocal()

    try:
        candidates = VectorSearchService.search(
            db=db,
            query="sensor",
            top_k=3,
        )

        service = RerankerService(
            model=FakeRerankerModel(
                scores=[0.1, 0.9, 0.3]
            )
        )

        ranked = service.rerank(
            query="sensor",
            candidates=candidates,
            top_k=2,
        )

        assert len(ranked) == 2
        assert ranked[0][1] == 0.9
        assert ranked[1][1] == 0.3

    finally:
        db.close()


def test_rerank_candidates_adds_rerank_score():
    db = SessionLocal()

    try:
        candidates = VectorSearchService.search(
            db=db,
            query="robot",
            top_k=3,
        )

        original_get_model = RerankerService.get_model

        try:
            # Patch model loading so this test verifies the
            # result-shaping logic without touching Hugging Face.
            RerankerService.get_model = classmethod(
                lambda cls: FakeRerankerModel(
                    scores=[0.2, 0.8, 0.5]
                )
            )

            reranked = RerankerService.rerank_candidates(
                query="robot",
                candidates=candidates,
                top_k=2,
            )

            assert len(reranked) == 2
            assert "rerank_score" in reranked[0]
            assert reranked[0]["rerank_score"] == 0.8

        finally:
            RerankerService.get_model = original_get_model

    finally:
        db.close()
