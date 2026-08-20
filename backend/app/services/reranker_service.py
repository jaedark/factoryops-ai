from time import perf_counter

from sentence_transformers import CrossEncoder

from backend.app.core.config import (
    RERANKER_MODEL_NAME,
)


class RerankerService:
    _shared_model = None
    _model_load_seconds = None

    def __init__(self, model=None):
        self.model = model or self.get_model()

    @classmethod
    def get_model(cls):
        if cls._shared_model is None:
            started_at = perf_counter()
            cls._shared_model = CrossEncoder(
                RERANKER_MODEL_NAME
            )
            cls._model_load_seconds = (
                perf_counter() - started_at
            )

        return cls._shared_model

    @classmethod
    def get_model_load_seconds(cls) -> float:
        cls.get_model()

        return float(cls._model_load_seconds)

    @staticmethod
    def build_rerank_text(candidate) -> str:
        incident = candidate["incident"]

        return (
            f"Equipment: {incident.equipment_name}\n"
            f"Process: {incident.process_name}\n"
            f"Symptom: {incident.symptom}\n"
            f"Cause: {incident.cause or ''}\n"
            f"Action: {incident.action_taken or ''}\n"
            f"Result: {incident.result or ''}"
        )

    def rerank(
        self,
        query,
        candidates,
        top_k=3,
    ):
        pairs = []

        for candidate in candidates:
            document = self.build_rerank_text(candidate)

            pairs.append(
                [query, document]
            )

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]

    @classmethod
    def rerank_candidates(
        cls,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        service = cls()
        ranked = service.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        reranked_results = []

        for candidate, score in ranked:
            result = dict(candidate)
            result["rerank_score"] = float(score)
            reranked_results.append(result)

        return reranked_results
