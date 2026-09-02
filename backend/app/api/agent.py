from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.agent import (
    ApprovalActionResponse,
    AgentChatRequest,
    AgentChatResponse,
)
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)
from backend.app.services.memory_service import MemoryStoreError


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
            session_id=request.session_id,
        )
    except AgentExecutionError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except MemoryStoreError as exc:
        raise HTTPException(
            status_code=500,
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


@router.post(
    "/approvals/{approval_id}/approve",
    response_model=ApprovalActionResponse,
)
def approve_tool_execution(
    approval_id: str,
    db: Session = Depends(get_db),
) -> ApprovalActionResponse:
    try:
        result = AgentService.approve_tool_execution(
            db=db,
            approval_id=approval_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return result


@router.post(
    "/approvals/{approval_id}/reject",
    response_model=ApprovalActionResponse,
)
def reject_tool_execution(
    approval_id: str,
) -> ApprovalActionResponse:
    try:
        result = AgentService.reject_tool_execution(
            approval_id=approval_id,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return result
