import pytest

from backend.app.core.database import SessionLocal
from backend.app.tools.industrial_tools import (
    create_maintenance_request,
    get_equipment_status,
    get_equipment_telemetry,
    get_high_risk_equipment,
)


def test_get_equipment_status_returns_current_status_summary():
    db = SessionLocal()

    try:
        result = get_equipment_status(
            db=db,
            equipment_id="Robot-01",
        )
    finally:
        db.close()

    assert result["equipment"]["equipment_id"] == "Robot-01"
    assert result["latest_telemetry"]["equipment_id"] == "Robot-01"
    assert result["is_high_risk"] is True
    assert "process_temperature_gap" in result["risk_reasons"]


def test_get_equipment_telemetry_returns_serializable_history():
    db = SessionLocal()

    try:
        result = get_equipment_telemetry(
            db=db,
            equipment_id="Vision-02",
        )
    finally:
        db.close()

    assert len(result) == 2
    assert result[0]["equipment_id"] == "Vision-02"
    assert result[0]["sequence"] == 7


def test_get_high_risk_equipment_returns_current_risk_list():
    db = SessionLocal()

    try:
        result = get_high_risk_equipment(db=db)
    finally:
        db.close()

    assert [item["equipment"]["equipment_id"] for item in result] == [
        "Conveyor-01",
        "Press-01",
        "Robot-01",
    ]
    assert result[0]["risk_reasons"] == [
        "machine_failure",
        "tool_wear_threshold",
    ]


def test_get_equipment_status_raises_for_missing_equipment():
    db = SessionLocal()

    try:
        with pytest.raises(LookupError) as exc:
            get_equipment_status(
                db=db,
                equipment_id="UNKNOWN-EQUIPMENT",
            )
    finally:
        db.close()

    assert str(exc.value) == "Equipment not found: UNKNOWN-EQUIPMENT"


def test_create_maintenance_request_returns_dummy_write_result():
    db = SessionLocal()

    try:
        result = create_maintenance_request(
            db=db,
            equipment_id="Robot-01",
            reason="servo drift requires inspection",
        )
    finally:
        db.close()

    assert result["equipment_id"] == "Robot-01"
    assert result["reason"] == "servo drift requires inspection"
    assert result["status"] == "created"
    assert result["request_id"].startswith("MR-")
