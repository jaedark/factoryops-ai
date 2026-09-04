from backend.app.evaluation.agent_evaluator import (
    AgentEvaluationCase,
)
from backend.app.schemas.agent import (
    AgentStatus,
)


DEFAULT_AGENT_EVALUATION_CASES = [
    AgentEvaluationCase(
        case_id="eval-001",
        query="Robot-01 현재 상태 알려줘",
        agent_name="incident_analysis",
        expected_tools=["get_equipment_status"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["industrial", "single-tool"],
    ),
    AgentEvaluationCase(
        case_id="eval-002",
        query="Robot-01 telemetry 이력 보여줘",
        agent_name="incident_analysis",
        expected_tools=["get_equipment_telemetry"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["industrial", "telemetry"],
    ),
    AgentEvaluationCase(
        case_id="eval-003",
        query="현재 고위험 설비 알려줘",
        agent_name="incident_analysis",
        expected_tools=["get_high_risk_equipment"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["industrial", "risk"],
    ),
    AgentEvaluationCase(
        case_id="eval-004",
        query="Robot-01 과거 장애 알려줘",
        agent_name="incident_analysis",
        expected_tools=["get_equipment_incidents"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["incident", "history"],
    ),
    AgentEvaluationCase(
        case_id="eval-005",
        query="모터 과열과 비슷한 장애 찾아줘",
        agent_name="incident_analysis",
        expected_tools=["search_incidents"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["incident", "rag"],
    ),
    AgentEvaluationCase(
        case_id="eval-006",
        query="Incident 1번 상세 내용 알려줘",
        agent_name="incident_analysis",
        expected_tools=["get_incident"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["incident", "detail"],
    ),
    AgentEvaluationCase(
        case_id="eval-007",
        query="Robot-01 현재 상태 확인하고 과거 비슷한 장애 찾아줘",
        agent_name="incident_analysis",
        expected_tools=[
            "get_equipment_status",
            "search_incidents",
        ],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["agentic-rag", "multi-step"],
    ),
    AgentEvaluationCase(
        case_id="eval-008",
        query=(
            "Robot-01의 현재 상태를 확인하고 과거 비슷한 장애이력을 "
            "찾아서 상세 원인까지 알려줘"
        ),
        agent_name="incident_analysis",
        expected_tools=[
            "get_equipment_status",
            "search_incidents",
            "get_incident",
        ],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["agentic-rag", "multi-step", "detail"],
    ),
    AgentEvaluationCase(
        case_id="eval-009",
        query="Robot-01 정비 요청 생성해줘",
        agent_name="maintenance_recommendation",
        expected_tools=["create_maintenance_request"],
        expected_status=AgentStatus.WAITING_APPROVAL,
        expected_approval_required=True,
        tags=["approval", "write-tool"],
    ),
    AgentEvaluationCase(
        case_id="eval-010",
        query="Sensor-05 정비 요청 만들어줘",
        agent_name="maintenance_recommendation",
        expected_tools=["create_maintenance_request"],
        expected_status=AgentStatus.WAITING_APPROVAL,
        expected_approval_required=True,
        tags=["approval", "write-tool"],
    ),
    AgentEvaluationCase(
        case_id="eval-011",
        query="안녕",
        agent_name="incident_analysis",
        expected_tools=[],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["no-tool", "chat"],
    ),
    AgentEvaluationCase(
        case_id="eval-012",
        query="Conveyor 상태 확인해줘",
        agent_name="incident_analysis",
        expected_tools=["get_equipment_status"],
        expected_status=AgentStatus.COMPLETED,
        expected_approval_required=False,
        tags=["industrial", "default"],
    ),
]
