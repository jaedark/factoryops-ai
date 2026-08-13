from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingService:
    _model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
        similarity = cosine_similarity(
            [vector_a],
            [vector_b],
        )[0][0]

        return float(similarity)
