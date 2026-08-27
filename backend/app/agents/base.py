from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    system_instruction: str = Field(min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
