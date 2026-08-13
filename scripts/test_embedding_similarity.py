from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.services.embedding_service import EmbeddingService


def main():
    base_text = "Motor temperature exceeded threshold"
    similar_text = "장비가 너무 뜨거워졌어"
    different_text = "Camera network communication timeout"

    base_vector = EmbeddingService.embed_text(base_text)
    similar_vector = EmbeddingService.embed_text(similar_text)
    different_vector = EmbeddingService.embed_text(different_text)

    similar_score = EmbeddingService.calculate_similarity(
        base_vector,
        similar_vector,
    )

    different_score = EmbeddingService.calculate_similarity(
        base_vector,
        different_vector,
    )

    print("=== Embedding Similarity Test ===")
    print()
    print(f"[기준 문장] {base_text}")
    print()

    print(f"[비교 1] {similar_text}")
    print(f"Similarity: {similar_score:.4f}")
    print()

    print(f"[비교 2] {different_text}")
    print(f"Similarity: {different_score:.4f}")
    print()

    if similar_score > different_score:
        print("결과: 의미가 비슷한 문장의 유사도가 더 높습니다.")
    else:
        print("결과: 예상과 다른 결과입니다. Embedding 결과를 확인해보세요.")


if __name__ == "__main__":
    main()
