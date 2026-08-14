from pydantic import BaseModel, Field


class RagAnalyzeRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="분석할 장애 상황 또는 질문",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )

    similarity_threshold: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
    )


class RagSource(BaseModel):
    incident_id: int
    equipment_name: str
    symptom: str
    similarity: float


class RagAnalyzeResponse(BaseModel):
    query: str
    answer: str
    sources: list[RagSource]
