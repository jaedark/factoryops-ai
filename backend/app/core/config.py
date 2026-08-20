import os


# Keep reranker knobs in one place so experiments can change
# model and cutoffs without touching search code.
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

RERANK_RETRIEVER_TOP_N = int(
    os.getenv("RERANK_RETRIEVER_TOP_N", "5")
)

RERANK_FINAL_TOP_K = int(
    os.getenv("RERANK_FINAL_TOP_K", "3")
)
