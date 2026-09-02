from backend.app.services.guardrail_service import (
    GuardrailService,
    InMemoryApprovalStore,
)


def test_guardrail_service_returns_low_risk_policy_for_read_tool():
    policy = GuardrailService.get_policy("search_incidents")
    decision = GuardrailService.evaluate(
        "search_incidents",
        {"query": "motor overheating"},
    )

    assert policy.risk_level == "low"
    assert policy.approval_required is False
    assert decision.allowed_to_execute is True
    assert decision.approval_required is False


def test_guardrail_service_returns_high_risk_policy_for_write_tool():
    policy = GuardrailService.get_policy(
        "create_maintenance_request"
    )
    decision = GuardrailService.evaluate(
        "create_maintenance_request",
        {
            "equipment_id": "Robot-01",
            "reason": "servo drift requires inspection",
        },
    )

    assert policy.risk_level == "high"
    assert policy.approval_required is True
    assert decision.allowed_to_execute is False
    assert decision.approval_required is True


def test_approval_store_creates_and_reads_request():
    store = InMemoryApprovalStore()
    request = store.create(
        tool_name="create_maintenance_request",
        tool_arguments={
            "equipment_id": "Robot-01",
            "reason": "servo drift requires inspection",
        },
        reason="Human approval required before executing tool.",
        session_id="demo-001",
    )

    loaded = store.get(request.approval_id)

    assert loaded.approval_id == request.approval_id
    assert loaded.status == "pending"
    assert loaded.session_id == "demo-001"


def test_approval_store_approve_and_reject_transitions():
    approve_store = InMemoryApprovalStore()
    approve_request = approve_store.create(
        tool_name="create_maintenance_request",
        tool_arguments={
            "equipment_id": "Robot-01",
            "reason": "servo drift requires inspection",
        },
        reason="needs approval",
    )

    approved = approve_store.approve(
        approve_request.approval_id
    )

    assert approved.status == "approved"

    reject_store = InMemoryApprovalStore()
    reject_request = reject_store.create(
        tool_name="create_maintenance_request",
        tool_arguments={
            "equipment_id": "Robot-01",
            "reason": "servo drift requires inspection",
        },
        reason="needs approval",
    )

    rejected = reject_store.reject(
        reject_request.approval_id
    )

    assert rejected.status == "rejected"
