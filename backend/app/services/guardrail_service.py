from itertools import count
from typing import Any, Protocol

from backend.app.schemas.agent import (
    ApprovalRequest,
    ApprovalStatus,
    GuardrailDecision,
    ToolPolicy,
    ToolRiskLevel,
)


class ApprovalStore(Protocol):
    def create(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        reason: str,
        session_id: str | None = None,
    ) -> ApprovalRequest: ...

    def get(
        self,
        approval_id: str,
    ) -> ApprovalRequest: ...

    def approve(
        self,
        approval_id: str,
    ) -> ApprovalRequest: ...

    def reject(
        self,
        approval_id: str,
    ) -> ApprovalRequest: ...


class InMemoryApprovalStore:
    def __init__(
        self,
    ) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._id_counter = count(1)

    def create(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        reason: str,
        session_id: str | None = None,
    ) -> ApprovalRequest:
        approval_id = f"APR-{next(self._id_counter):04d}"
        request = ApprovalRequest(
            approval_id=approval_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            status=ApprovalStatus.PENDING,
            reason=reason,
            session_id=session_id,
        )
        self._requests[approval_id] = request
        return request

    def get(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        request = self._requests.get(approval_id)
        if request is None:
            raise LookupError(
                f"Approval request not found: {approval_id}"
            )
        return request

    def approve(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        request = self.get(approval_id)
        if request.status == ApprovalStatus.REJECTED:
            raise ValueError(
                f"Approval request was rejected: {approval_id}"
            )
        request.status = ApprovalStatus.APPROVED
        return request

    def reject(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        request = self.get(approval_id)
        if request.status == ApprovalStatus.APPROVED:
            raise ValueError(
                f"Approval request was already approved: {approval_id}"
            )
        request.status = ApprovalStatus.REJECTED
        return request


class GuardrailService:
    _TOOL_POLICIES = {
        "search_incidents": ToolPolicy(
            tool_name="search_incidents",
            risk_level=ToolRiskLevel.LOW,
            approval_required=False,
        ),
        "get_incident": ToolPolicy(
            tool_name="get_incident",
            risk_level=ToolRiskLevel.LOW,
            approval_required=False,
        ),
        "get_equipment_incidents": ToolPolicy(
            tool_name="get_equipment_incidents",
            risk_level=ToolRiskLevel.LOW,
            approval_required=False,
        ),
        "get_equipment_status": ToolPolicy(
            tool_name="get_equipment_status",
            risk_level=ToolRiskLevel.LOW,
            approval_required=False,
        ),
        "get_equipment_telemetry": ToolPolicy(
            tool_name="get_equipment_telemetry",
            risk_level=ToolRiskLevel.LOW,
            approval_required=False,
        ),
        "get_high_risk_equipment": ToolPolicy(
            tool_name="get_high_risk_equipment",
            risk_level=ToolRiskLevel.LOW,
            approval_required=False,
        ),
        "create_maintenance_request": ToolPolicy(
            tool_name="create_maintenance_request",
            risk_level=ToolRiskLevel.HIGH,
            approval_required=True,
        ),
    }

    @classmethod
    def get_policy(
        cls,
        tool_name: str,
    ) -> ToolPolicy:
        policy = cls._TOOL_POLICIES.get(tool_name)
        if policy is None:
            raise ValueError(
                f"Tool policy not found: {tool_name}"
            )
        return policy

    @classmethod
    def evaluate(
        cls,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> GuardrailDecision:
        _ = tool_arguments
        policy = cls.get_policy(tool_name)

        if policy.approval_required:
            return GuardrailDecision(
                allowed_to_execute=False,
                approval_required=True,
                risk_level=policy.risk_level,
                reason=(
                    "Human approval required before executing "
                    f"tool '{tool_name}'."
                ),
            )

        return GuardrailDecision(
            allowed_to_execute=True,
            approval_required=False,
            risk_level=policy.risk_level,
            reason=f"Tool '{tool_name}' can be executed automatically.",
        )
