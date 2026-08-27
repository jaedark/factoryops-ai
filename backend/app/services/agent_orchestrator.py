from collections.abc import Callable

from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    KNOWLEDGE_SEARCH_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT,
    REPORT_AGENT,
)
from backend.app.agents.base import AgentDefinition
from backend.app.core.database import SessionLocal
from backend.app.services.agent_service import AgentService


ROUTING_RULES = [
    (
        "report",
        [
            "보고서",
            "리포트",
            "보고 형식",
            "보고서 형식",
            "요약해줘",
            "정리해줘",
        ],
    ),
    (
        "maintenance_recommendation",
        [
            "유지보수",
            "정비",
            "예방 정비",
            "예방정비",
            "교체",
            "예방조치",
            "점검 방법",
            "정비 방법",
        ],
    ),
    (
        "knowledge_search",
        [
            "매뉴얼",
            "manual",
            "sop",
            "작업절차",
            "작업 절차",
            "가이드",
            "문서 찾아",
            "절차 찾아",
        ],
    ),
]


AGENT_REGISTRY: dict[str, AgentDefinition] = {
    INCIDENT_ANALYSIS_AGENT.name: INCIDENT_ANALYSIS_AGENT,
    KNOWLEDGE_SEARCH_AGENT.name: KNOWLEDGE_SEARCH_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT.name: (
        MAINTENANCE_RECOMMENDATION_AGENT
    ),
    REPORT_AGENT.name: REPORT_AGENT,
}


class AgentOrchestrator:
    def __init__(
        self,
        agent_service,
        session_factory: Callable = SessionLocal,
    ) -> None:
        self.agent_service = agent_service
        self.session_factory = session_factory

    def select_agent(
        self,
        message: str,
    ) -> str:
        normalized_message = message.lower()

        # TODO:
        # Current routing selects a single agent only.
        # Multi-agent execution plans will be introduced
        # in the orchestration workflow layer.
        for agent_name, keywords in ROUTING_RULES:
            if any(
                keyword.lower() in normalized_message
                for keyword in keywords
            ):
                return agent_name

        return INCIDENT_ANALYSIS_AGENT.name

    async def run(
        self,
        message: str,
        max_steps: int = 5,
    ):
        agent_name = self.select_agent(message)
        agent_definition = AGENT_REGISTRY[agent_name]
        db = self.session_factory()

        try:
            return self.agent_service.run(
                db=db,
                agent_definition=agent_definition,
                message=message,
                max_steps=max_steps,
            )
        finally:
            db.close()


DEFAULT_AGENT_ORCHESTRATOR = AgentOrchestrator(AgentService)
