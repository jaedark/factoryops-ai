from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
)
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


@router.post(
    "/chat",
    response_model=AgentChatResponse,
)
def agent_chat(
    request: AgentChatRequest,
    db: Session = Depends(get_db),
) -> AgentChatResponse:
    try:
        result = AgentService.chat(
            db=db,
            message=request.message,
            max_steps=request.max_steps,
        )
    except AgentExecutionError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return AgentChatResponse.model_validate(
        result.model_dump()
    )
