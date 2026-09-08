from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    request_id: str = Field(min_length=1)


class ApiErrorResponse(BaseModel):
    error: ApiError
