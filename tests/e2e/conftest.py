from dataclasses import dataclass
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base, get_db
from backend.app.observability import InMemoryObservabilitySink
from backend.app.resilience import ResilientExecutionService
from backend.app.schemas.incident import IncidentCreate
from backend.app.services.agent_service import AgentService
from backend.app.services.guardrail_service import InMemoryApprovalStore
from backend.app.services.incident_service import IncidentService
from backend.app.services.memory_service import InMemoryMemoryStore
from backend.app.services.tool_calling_service import ToolCallingService
from backend.main import app


API_KEY = "test-api-key"


@dataclass
class E2EEnvironment:
    client: TestClient
    public_client: TestClient
    session_factory: sessionmaker
    memory_store: InMemoryMemoryStore
    approval_store: InMemoryApprovalStore
    observability_sink: InMemoryObservabilitySink
    incident_ids: list[int]


@pytest.fixture
def e2e_environment(tmp_path, monkeypatch):
    database_path = tmp_path / "e2e.db"
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    Base.metadata.create_all(bind=engine)

    incidents = [
        IncidentCreate(
            equipment_name="Robot-01",
            process_name="Assembly",
            occurred_at=datetime(2026, 1, 10, 9, 0),
            symptom="Bearing vibration and motor temperature increase",
            cause="Bearing wear",
            action_taken="Replace bearing and align shaft",
            result="Vibration returned to normal",
        ),
        IncidentCreate(
            equipment_name="Robot-01",
            process_name="Assembly",
            occurred_at=datetime(2026, 2, 11, 14, 30),
            symptom="Servo position deviation and intermittent alarm",
            cause="Encoder calibration drift",
            action_taken="Recalibrate encoder and inspect connector",
            result="Position accuracy restored",
        ),
    ]
    with testing_session() as db:
        incident_ids = [
            IncidentService.create_incident(db, incident).incident_id
            for incident in incidents
        ]

    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    memory_store = InMemoryMemoryStore()
    approval_store = InMemoryApprovalStore()
    observability_sink = InMemoryObservabilitySink()
    monkeypatch.setattr(AgentService, "_MEMORY_STORE", memory_store)
    monkeypatch.setattr(
        AgentService,
        "_OBSERVABILITY_SINK",
        observability_sink,
    )
    monkeypatch.setattr(
        ToolCallingService,
        "_APPROVAL_STORE",
        approval_store,
    )
    ResilientExecutionService.reset_default_llm_circuit_breaker()
    app.dependency_overrides[get_db] = override_get_db

    environment = E2EEnvironment(
        client=TestClient(
            app,
            headers={"X-API-Key": API_KEY},
        ),
        public_client=TestClient(app),
        session_factory=testing_session,
        memory_store=memory_store,
        approval_store=approval_store,
        observability_sink=observability_sink,
        incident_ids=incident_ids,
    )
    yield environment

    app.dependency_overrides.clear()
    ResilientExecutionService.reset_default_llm_circuit_breaker()
    engine.dispose()
