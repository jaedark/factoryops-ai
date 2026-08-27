import asyncio
from types import SimpleNamespace

from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT,
)
from backend.app.services.agent_orchestrator import (
    AgentOrchestrator,
)


class FakeSession:
    def close(self) -> None:
        self.closed = True


class FakeAgentService:
    def __init__(
        self,
        outputs: dict[str, str] | None = None,
    ) -> None:
        self.calls = []
        self.outputs = outputs or {
            "incident_analysis": "incident-output",
            "knowledge_search": "knowledge-output",
            "maintenance_recommendation": "maintenance-output",
            "report": "report-output",
        }

    def run(
        self,
        db,
        agent_definition,
        message: str,
        max_steps: int = 5,
    ):
        self.calls.append(
            {
                "db": db,
                "agent_definition": agent_definition,
                "message": message,
                "max_steps": max_steps,
            }
        )
        return SimpleNamespace(
            answer=self.outputs[agent_definition.name],
            agent_name=agent_definition.name,
            message=message,
            max_steps=max_steps,
        )


def test_select_agent_routes_to_incident_analysis():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert (
        orchestrator.select_agent(
            "Conveyor-01에서 모터 과열 알람이 발생했어"
        )
        == "incident_analysis"
    )


def test_select_agent_routes_to_knowledge_search():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert orchestrator.select_agent("PLC 점검 SOP 찾아줘") == (
        "knowledge_search"
    )


def test_select_agent_routes_to_maintenance_recommendation():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert orchestrator.select_agent(
        "Robot-01 예방 정비 방법 추천해줘"
    ) == "maintenance_recommendation"


def test_select_agent_routes_to_report():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert orchestrator.select_agent(
        "분석 결과를 보고서 형식으로 정리해줘"
    ) == "report"


def test_select_agent_uses_incident_analysis_as_default():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert orchestrator.select_agent("Conveyor 상태 확인해줘") == (
        "incident_analysis"
    )


def test_select_agent_applies_priority_to_maintenance():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert orchestrator.select_agent(
        "컨베이어 장애 원인을 분석하고 유지보수 방법을 추천해줘"
    ) == "maintenance_recommendation"


def test_select_agent_applies_priority_to_report():
    orchestrator = AgentOrchestrator(FakeAgentService())

    assert orchestrator.select_agent(
        "컨베이어 장애 원인과 유지보수 방법을 보고서로 정리해줘"
    ) == "report"


def test_run_passes_selected_agent_to_agent_service():
    fake_agent_service = FakeAgentService()
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run(
            "Robot-01 예방 정비 방법 추천해줘",
            max_steps=4,
        )
    )

    assert result.agent_name == "maintenance_recommendation"
    assert len(fake_agent_service.calls) == 1
    assert (
        fake_agent_service.calls[0]["agent_definition"]
        == MAINTENANCE_RECOMMENDATION_AGENT
    )
    assert fake_agent_service.calls[0]["message"] == (
        "Robot-01 예방 정비 방법 추천해줘"
    )
    assert fake_agent_service.calls[0]["max_steps"] == 4


def test_run_uses_default_incident_analysis_agent():
    fake_agent_service = FakeAgentService()
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run("Conveyor 상태 확인해줘")
    )

    assert result.agent_name == "incident_analysis"
    assert (
        fake_agent_service.calls[0]["agent_definition"]
        == INCIDENT_ANALYSIS_AGENT
    )


def test_build_execution_plan_uses_default_incident_analysis():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan("Conveyor 상태 확인해줘")

    assert plan.agents == ["incident_analysis"]
    assert plan.primary_agent == "incident_analysis"
    assert plan.total_steps == 1
    assert plan.steps[0].agent_name == "incident_analysis"
    assert plan.steps[0].depends_on == []


def test_build_execution_plan_incident_only():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "Conveyor-01 과열 알람 원인을 분석해줘"
    )

    assert plan.agents == ["incident_analysis"]


def test_build_execution_plan_knowledge_only():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan("PLC 점검 SOP 찾아줘")

    assert plan.agents == ["knowledge_search"]


def test_build_execution_plan_maintenance_only():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "Robot-01 예방 정비 방법 추천해줘"
    )

    assert plan.agents == ["maintenance_recommendation"]


def test_build_execution_plan_report_only():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan("이 내용을 보고서로 정리해줘")

    assert plan.agents == ["report"]


def test_build_execution_plan_incident_and_maintenance():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "Robot-01 장애 원인을 분석하고 예방 정비 방법 추천해줘"
    )

    assert plan.agents == [
        "incident_analysis",
        "maintenance_recommendation",
    ]


def test_build_execution_plan_incident_and_knowledge():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "PLC 장애 원인을 분석하고 관련 SOP도 찾아줘"
    )

    assert plan.agents == [
        "incident_analysis",
        "knowledge_search",
    ]


def test_build_execution_plan_incident_maintenance_report():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "컨베이어 장애 원인을 분석하고 정비 방법을 추천한 뒤 보고서로 정리해줘"
    )

    assert plan.agents == [
        "incident_analysis",
        "maintenance_recommendation",
        "report",
    ]


def test_build_execution_plan_knowledge_maintenance_report():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "관련 SOP를 찾고 유지보수 방법을 추천해서 보고서로 정리해줘"
    )

    assert plan.agents == [
        "knowledge_search",
        "maintenance_recommendation",
        "report",
    ]


def test_build_execution_plan_all_agents():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "장애 원인을 분석하고 관련 매뉴얼을 찾아서 예방 정비 방안을 보고서로 정리해줘"
    )

    assert plan.agents == [
        "incident_analysis",
        "knowledge_search",
        "maintenance_recommendation",
        "report",
    ]


def test_build_execution_plan_ignores_input_order():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "보고서로 정리하고 장애 원인도 분석해줘"
    )

    assert plan.agents == [
        "incident_analysis",
        "report",
    ]


def test_build_execution_plan_removes_duplicate_intent():
    orchestrator = AgentOrchestrator(FakeAgentService())

    plan = orchestrator.build_execution_plan(
        "장애 원인을 분석하고 고장 원인도 다시 분석해줘"
    )

    assert plan.agents == ["incident_analysis"]


def test_select_agent_and_execution_plan_have_different_roles():
    orchestrator = AgentOrchestrator(FakeAgentService())

    message = "컨베이어 장애 원인을 분석하고 유지보수 방법을 추천해줘"

    assert orchestrator.select_agent(message) == (
        "maintenance_recommendation"
    )
    assert orchestrator.build_execution_plan(message).agents == [
        "incident_analysis",
        "maintenance_recommendation",
    ]


def test_run_execution_plan_supports_single_agent_plan():
    fake_agent_service = FakeAgentService(
        outputs={"knowledge_search": "knowledge-output"}
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run_execution_plan("PLC 점검 SOP 찾아줘")
    )

    assert result.plan.agents == ["knowledge_search"]
    assert len(result.steps) == 1
    assert result.steps[0].agent == "knowledge_search"
    assert result.steps[0].output == "knowledge-output"
    assert result.final_output == "knowledge-output"
    assert len(fake_agent_service.calls) == 1


def test_run_execution_plan_runs_agents_sequentially():
    fake_agent_service = FakeAgentService()
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run_execution_plan(
            "Robot-01 장애 원인을 분석하고 예방 정비 방법 추천해줘"
        )
    )

    assert result.plan.agents == [
        "incident_analysis",
        "maintenance_recommendation",
    ]
    assert [step.agent for step in result.steps] == [
        "incident_analysis",
        "maintenance_recommendation",
    ]
    assert [
        call["agent_definition"].name
        for call in fake_agent_service.calls
    ] == [
        "incident_analysis",
        "maintenance_recommendation",
    ]


def test_run_execution_plan_propagates_previous_result():
    fake_agent_service = FakeAgentService(
        outputs={
            "incident_analysis": "bearing wear detected",
            "maintenance_recommendation": "replace bearing soon",
        }
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )
    message = "Robot-01 장애 원인을 분석하고 예방 정비 방법 추천해줘"

    asyncio.run(orchestrator.run_execution_plan(message))

    assert "bearing wear detected" in fake_agent_service.calls[1]["message"]


def test_run_execution_plan_preserves_original_message():
    fake_agent_service = FakeAgentService(
        outputs={
            "incident_analysis": "bearing wear detected",
            "maintenance_recommendation": "replace bearing soon",
        }
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )
    message = "Robot-01 장애 원인을 분석하고 예방 정비 방법 추천해줘"

    asyncio.run(orchestrator.run_execution_plan(message))

    assert message in fake_agent_service.calls[1]["message"]


def test_run_execution_plan_accumulates_three_agent_results():
    fake_agent_service = FakeAgentService(
        outputs={
            "incident_analysis": "incident-result",
            "knowledge_search": "knowledge-result",
            "report": "report-output",
        }
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    asyncio.run(
        orchestrator.run_execution_plan(
            "PLC 장애 원인을 분석하고 SOP를 찾아서 보고서로 정리해줘"
        )
    )

    report_input = fake_agent_service.calls[2]["message"]

    assert "incident-result" in report_input
    assert "knowledge-result" in report_input


def test_run_execution_plan_supports_full_four_agent_execution():
    fake_agent_service = FakeAgentService()
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run_execution_plan(
            "장애 원인을 분석하고 관련 매뉴얼을 찾아서 예방 정비 방안을 보고서로 정리해줘"
        )
    )

    assert result.plan.agents == [
        "incident_analysis",
        "knowledge_search",
        "maintenance_recommendation",
        "report",
    ]
    assert [step.agent for step in result.steps] == result.plan.agents


def test_run_execution_plan_uses_last_agent_output_as_final_output():
    fake_agent_service = FakeAgentService(
        outputs={
            "incident_analysis": "incident-output",
            "knowledge_search": "knowledge-output",
            "maintenance_recommendation": "maintenance-output",
            "report": "final-report-output",
        }
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run_execution_plan(
            "장애 원인을 분석하고 관련 매뉴얼을 찾아서 예방 정비 방안을 보고서로 정리해줘"
        )
    )

    assert result.final_output == "final-report-output"


def test_run_execution_plan_preserves_step_results():
    fake_agent_service = FakeAgentService(
        outputs={
            "incident_analysis": "incident-output",
            "maintenance_recommendation": "maintenance-output",
        }
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    result = asyncio.run(
        orchestrator.run_execution_plan(
            "Robot-01 장애 원인을 분석하고 예방 정비 방법 추천해줘"
        )
    )

    assert [
        (step.agent, step.output)
        for step in result.steps
    ] == [
        ("incident_analysis", "incident-output"),
        (
            "maintenance_recommendation",
            "maintenance-output",
        ),
    ]


def test_run_execution_plan_uses_build_execution_plan():
    fake_agent_service = FakeAgentService()
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )

    custom_plan = orchestrator.build_execution_plan(
        "placeholder"
    ).model_copy(
        update={
            "agents": [
                "knowledge_search",
                "report",
            ]
        }
    )

    original_build_execution_plan = (
        orchestrator.build_execution_plan
    )

    try:
        orchestrator.build_execution_plan = lambda _message: custom_plan
        result = asyncio.run(
            orchestrator.run_execution_plan(
                "Robot-01 장애 원인을 분석하고 예방 정비 방법 추천해줘"
            )
        )
    finally:
        orchestrator.build_execution_plan = original_build_execution_plan

    assert result.plan.agents == ["knowledge_search", "report"]
    assert [
        call["agent_definition"].name
        for call in fake_agent_service.calls
    ] == ["knowledge_search", "report"]


def test_run_and_run_execution_plan_have_different_behavior():
    fake_agent_service = FakeAgentService(
        outputs={
            "maintenance_recommendation": "maintenance-output",
            "incident_analysis": "incident-output",
        }
    )
    fake_session = FakeSession()
    orchestrator = AgentOrchestrator(
        fake_agent_service,
        session_factory=lambda: fake_session,
    )
    message = "컨베이어 장애 원인을 분석하고 유지보수 방법 추천해줘"

    single_result = asyncio.run(orchestrator.run(message))
    multi_result = asyncio.run(orchestrator.run_execution_plan(message))

    assert single_result.agent_name == "maintenance_recommendation"
    assert multi_result.plan.agents == [
        "incident_analysis",
        "maintenance_recommendation",
    ]
    assert len(multi_result.steps) == 2
