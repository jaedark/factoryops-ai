from backend.app.schemas.incident import IncidentResponse

from pydantic import BaseModel


class VectorSearchResult(BaseModel):
    incident: IncidentResponse
    similarity: float
