from fastapi import FastAPI

app = FastAPI(
    title="FactoryOps AI API",
    description="제조 장애 대응 및 지식 검색 API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
