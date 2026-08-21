from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.tool_calling import (
    ToolChatRequest,
    ToolChatResponse,
)
from backend.app.services.tool_calling_service import (
    ToolCallingService,
)


router = APIRouter(
    prefix="/tools",
    tags=["tools"],
)


@router.post(
    "/chat",
    response_model=ToolChatResponse,
)
def tool_chat(
    request: ToolChatRequest,
    db: Session = Depends(get_db),
) -> ToolChatResponse:
    try:
        result = ToolCallingService.chat(
            db=db,
            message=request.message,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return ToolChatResponse.model_validate(
        result.model_dump()
    )
