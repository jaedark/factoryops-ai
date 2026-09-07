from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.core.config import get_settings


class EmbeddingService:
    # Reuse one multilingual encoder for every vector-search call
    # so the app does not pay model load cost repeatedly.
    _model = SentenceTransformer(
        get_settings().embedding_model
    )

    @classmethod
    def embed_text(cls, text: str):
        return cls._model.encode(
            text,
            normalize_embeddings=True,
        )

    @staticmethod
    def calculate_similarity(
        vector_a,
        vector_b,
    ) -> float:
        # Retrieval ranking is driven by cosine similarity between
        # the query embedding and each incident embedding.
        similarity = cosine_similarity(
            [vector_a],
            [vector_b],
        )[0][0]

        return float(similarity)
