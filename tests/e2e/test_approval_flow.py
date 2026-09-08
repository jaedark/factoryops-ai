from unittest.mock import Mock, patch

import pytest

from backend.app.agents import MAINTENANCE_RECOMMENDATION_AGENT
from backend.app.observability import ObservabilityEventType
from backend.app.schemas.agent import ApprovalStatus
from backend.app.services.agent_service import AgentService
from backend.app.services.llm_service import LlmService
from backend.app.services.tool_calling_service import ToolCallingService
from tests.e2e.helpers import build_tool_response


pytestmark = pytest.mark.e2e


def _maintenance_tool_response():
    return build_tool_response(
        "create_maintenance_request",
        {
            "equipment_id": "Robot-01",
            "reason": "Bearing vibration requires inspection",
        },
    )


def test_e2e_02_03_04_maintenance_approve_is_one_shot(
    e2e_environment,
    monkeypatch,
):
    raw_tool = ToolCallingService._TOOL_REGISTRY[
        "create_maintenance_request"
    ]
    execution_spy = Mock(wraps=raw_tool)
    monkeypatch.setitem(
        ToolCallingService._TOOL_REGISTRY,
        "create_maintenance_request",
        execution_spy,
    )

    with patch.object(
        LlmService,
        "generate_content",
        return_value=_maintenance_tool_response(),
    ):
        pending = e2e_environment.client.post(
            "/tools/chat",
            json={"message": "Robot-01 정비 요청을 만들어줘."},
        )

    assert pending.status_code == 200
    pending_data = pending.json()
    approval_id = pending_data["approval_request"]["approval_id"]
    assert pending_data["status"] == "waiting_approval"
    assert pending_data["termination_reason"] == "approval_required"
    assert pending_data["tool_called"] == "create_maintenance_request"
    assert pending_data["tool_result"] is None
    assert execution_spy.call_count == 0
    assert e2e_environment.approval_store.get(approval_id).status == (
        ApprovalStatus.PENDING
    )

    approved = e2e_environment.client.post(
        f"/agent/approvals/{approval_id}/approve"
    )
    assert approved.status_code == 200
    approved_data = approved.json()
    assert approved_data["approval_request"]["approval_id"] == approval_id
    assert approved_data["approval_request"]["status"] == "executed"
    assert approved_data["tool_result"]["status"] == "created"
    assert approved_data["tool_result"]["equipment_id"] == "Robot-01"
    assert execution_spy.call_count == 1

    duplicate = e2e_environment.client.post(
        f"/agent/approvals/{approval_id}/approve"
    )
    assert duplicate.status_code == 400
    assert duplicate.json()["error"]["code"] == "INVALID_REQUEST"
    assert execution_spy.call_count == 1


def test_e2e_05_rejected_approval_cannot_execute(
    e2e_environment,
    monkeypatch,
):
    execution_spy = Mock(
        wraps=ToolCallingService._TOOL_REGISTRY[
            "create_maintenance_request"
        ]
    )
    monkeypatch.setitem(
        ToolCallingService._TOOL_REGISTRY,
        "create_maintenance_request",
        execution_spy,
    )
    with patch.object(
        LlmService,
        "generate_content",
        return_value=_maintenance_tool_response(),
    ):
        pending = e2e_environment.client.post(
            "/tools/chat",
            json={"message": "Robot-01 정비 요청을 만들어줘."},
        )

    approval_id = pending.json()["approval_request"]["approval_id"]
    rejected = e2e_environment.client.post(
        f"/agent/approvals/{approval_id}/reject"
    )
    assert rejected.status_code == 200
    assert rejected.json()["approval_request"]["status"] == "rejected"
    assert execution_spy.call_count == 0

    approve_after_reject = e2e_environment.client.post(
        f"/agent/approvals/{approval_id}/approve"
    )
    assert approve_after_reject.status_code == 400
    assert e2e_environment.approval_store.get(approval_id).status == (
        ApprovalStatus.REJECTED
    )
    assert execution_spy.call_count == 0


def test_approval_guardrail_emits_event_before_any_tool_execution(
    e2e_environment,
    monkeypatch,
):
    execution_spy = Mock(
        wraps=ToolCallingService._TOOL_REGISTRY[
            "create_maintenance_request"
        ]
    )
    monkeypatch.setitem(
        ToolCallingService._TOOL_REGISTRY,
        "create_maintenance_request",
        execution_spy,
    )
    with e2e_environment.session_factory() as db, patch.object(
        LlmService,
        "generate_content",
        return_value=_maintenance_tool_response(),
    ):
        result = AgentService.run(
            db=db,
            agent_definition=MAINTENANCE_RECOMMENDATION_AGENT,
            message="Robot-01 정비 요청을 만들어줘.",
            approval_store=e2e_environment.approval_store,
            observability_sink=e2e_environment.observability_sink,
        )

    event_types = [
        event.event_type
        for event in e2e_environment.observability_sink.events
    ]
    assert result.status.value == "waiting_approval"
    assert ObservabilityEventType.APPROVAL_REQUIRED in event_types
    assert ObservabilityEventType.TOOL_CALL_STARTED not in event_types
    assert ObservabilityEventType.TOOL_CALL_COMPLETED not in event_types
    assert execution_spy.call_count == 0
