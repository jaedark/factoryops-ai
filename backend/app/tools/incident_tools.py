from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.schemas.incident import IncidentResponse
from backend.app.services.incident_service import IncidentService
from backend.app.services.rrf_search_service import (
    RrfSearchService,
)


def _serialize_incident(incident) -> dict:
    return IncidentResponse.model_validate(
        incident
    ).model_dump(mode="json")


def search_incidents(
    db: Session,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    자연어 query로 관련 incident를 검색한다.

    Parameters:
    - db: SQLAlchemy 세션
    - query: 검색에 사용할 자연어 질의
    - top_k: 반환할 최대 incident 개수

    Returns:
    - LLM이 바로 사용할 수 있는 JSON serializable incident 목록
    """
    results = RrfSearchService.search(
        db=db,
        query=query,
        top_k=top_k,
    )

    return [
        _serialize_incident(result["incident"])
        for result in results
    ]


def get_incident(
    db: Session,
    incident_id: int,
) -> dict:
    """
    incident_id로 특정 장애 상세 정보를 조회한다.

    Parameters:
    - db: SQLAlchemy 세션
    - incident_id: 조회할 incident의 고유 ID

    Returns:
    - JSON serializable incident 상세 정보
    """
    incident = IncidentService.get_incident(
        db=db,
        incident_id=incident_id,
    )

    if incident is None:
        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return _serialize_incident(incident)


def get_equipment_incidents(
    db: Session,
    equipment_name: str,
) -> list[dict]:
    """
    equipment_name으로 해당 장비의 장애 이력을 조회한다.

    Parameters:
    - db: SQLAlchemy 세션
    - equipment_name: 조회할 장비 이름

    Returns:
    - 해당 장비의 JSON serializable incident 목록
    """
    incidents = IncidentService.get_equipment_incidents(
        db=db,
        equipment_name=equipment_name,
    )

    return [
        _serialize_incident(incident)
        for incident in incidents
    ]
