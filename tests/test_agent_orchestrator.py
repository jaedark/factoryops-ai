import asyncio

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
    def __init__(self) -> None:
        self.calls = []

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
        return {
            "agent_name": agent_definition.name,
            "message": message,
            "max_steps": max_steps,
        }


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

    assert result["agent_name"] == "maintenance_recommendation"
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

    assert result["agent_name"] == "incident_analysis"
    assert (
        fake_agent_service.calls[0]["agent_definition"]
        == INCIDENT_ANALYSIS_AGENT
    )
