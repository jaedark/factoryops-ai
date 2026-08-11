from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IncidentCreate(BaseModel):
    equipment_name: str = Field(min_length=1, max_length=100)
    process_name: str = Field(min_length=1, max_length=100)
    occurred_at: datetime
    symptom: str = Field(min_length=1)
    cause: str | None = None
    action_taken: str | None = None
    result: str | None = None


class IncidentResponse(IncidentCreate):
    model_config = ConfigDict(from_attributes=True)

    incident_id: int
    created_at: datetime
