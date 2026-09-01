from sqlalchemy.orm import Session

from backend.app.services.industrial_data_service import (
    IndustrialDataService,
)


def _serialize_model(
    model,
) -> dict:
    return model.model_dump(mode="json")


def get_equipment_status(
    db: Session,
    equipment_id: str,
) -> dict:
    """
    설비의 현재 상태 요약을 조회한다.

    Parameters:
    - db: 기존 tool 인터페이스와 호환성을 위한 SQLAlchemy 세션
    - equipment_id: 조회할 설비 ID

    Returns:
    - 설비 정보, 최신 telemetry, 고위험 여부, 위험 사유를 담은 상태 요약
    """
    _ = db
    service = IndustrialDataService()
    equipment = service.get_equipment(equipment_id)
    telemetry = service.get_equipment_telemetry(equipment_id)
    high_risk_map = {
        item.equipment.equipment_id: item
        for item in service.get_high_risk_equipment()
    }
    high_risk_item = high_risk_map.get(equipment_id)

    return {
        "equipment": _serialize_model(equipment),
        "latest_telemetry": (
            _serialize_model(telemetry[-1])
            if telemetry
            else None
        ),
        "is_high_risk": high_risk_item is not None,
        "risk_reasons": (
            list(high_risk_item.risk_reasons)
            if high_risk_item is not None
            else []
        ),
    }


def get_equipment_telemetry(
    db: Session,
    equipment_id: str,
) -> list[dict]:
    """
    설비 telemetry 이력을 조회한다.

    Parameters:
    - db: 기존 tool 인터페이스와 호환성을 위한 SQLAlchemy 세션
    - equipment_id: 조회할 설비 ID

    Returns:
    - JSON serializable telemetry 목록
    """
    _ = db
    service = IndustrialDataService()
    telemetry = service.get_equipment_telemetry(equipment_id)

    return [
        _serialize_model(item)
        for item in telemetry
    ]


def get_high_risk_equipment(
    db: Session,
) -> list[dict]:
    """
    현재 고위험 설비 목록을 조회한다.

    Parameters:
    - db: 기존 tool 인터페이스와 호환성을 위한 SQLAlchemy 세션

    Returns:
    - 설비, 최신 telemetry, 위험 사유를 포함한 고위험 설비 목록
    """
    _ = db
    service = IndustrialDataService()
    high_risk_items = service.get_high_risk_equipment()

    return [
        {
            "equipment": _serialize_model(item.equipment),
            "latest_telemetry": _serialize_model(
                item.latest_telemetry
            ),
            "risk_reasons": list(item.risk_reasons),
        }
        for item in high_risk_items
    ]
