from fastapi import HTTPException

from backend.app.core.database import SessionLocal
from backend.app.services.incident_service import IncidentService
from backend.app.tools.incident_tools import (
    get_equipment_incidents,
    get_incident,
    search_incidents,
)


def test_search_incidents_returns_serializable_results():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)

        results = search_incidents(
            db=db,
            query="motor temperature issue",
            top_k=3,
        )

        assert len(results) > 0
        assert isinstance(results[0], dict)
        assert results[0]["equipment_name"] == "Conveyor-01"
        assert isinstance(results[0]["incident_id"], int)
        assert isinstance(results[0]["occurred_at"], str)

    finally:
        db.close()


def test_get_incident_returns_serializable_detail():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)

        equipment_incidents = get_equipment_incidents(
            db=db,
            equipment_name="Conveyor-01",
        )

        incident = get_incident(
            db=db,
            incident_id=equipment_incidents[0]["incident_id"],
        )

        assert incident["equipment_name"] == "Conveyor-01"
        assert isinstance(incident["created_at"], str)

    finally:
        db.close()


def test_get_incident_raises_not_found_for_missing_id():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)

        try:
            get_incident(
                db=db,
                incident_id=999999,
            )
            assert False, "Expected HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 404
            assert exc.detail == "Incident not found"

    finally:
        db.close()


def test_get_equipment_incidents_returns_matching_history():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)

        incidents = get_equipment_incidents(
            db=db,
            equipment_name="Vision-02",
        )

        assert len(incidents) == 1
        assert incidents[0]["equipment_name"] == "Vision-02"
        assert isinstance(incidents[0]["incident_id"], int)

    finally:
        db.close()


def test_get_equipment_incidents_returns_empty_list_for_unknown_equipment():
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)

        incidents = get_equipment_incidents(
            db=db,
            equipment_name="UNKNOWN-EQUIPMENT-999",
        )

        assert incidents == []

    finally:
        db.close()
