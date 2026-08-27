from collections.abc import Callable

from pydantic import BaseModel

from backend.app.agents import (
    INCIDENT_ANALYSIS_AGENT,
    KNOWLEDGE_SEARCH_AGENT,
    MAINTENANCE_RECOMMENDATION_AGENT,
    REPORT_AGENT,
)
from backend.app.agents.base import AgentDefinition
from backend.app.core.database import SessionLocal
from backend.app.services.agent_service import AgentService


class AgentPlanStep(BaseModel):
    step: int
    agent_name: str
    depends_on: list[str]


class AgentExecutionPlan(BaseModel):
    agents: list[str]
    primary_agent: str
    steps: list[AgentPlanStep]
    total_steps: int


class AgentExecutionResult(BaseModel):
    step: int
    agent: str
    output: str


class MultiAgentExecutionResult(BaseModel):
    plan: AgentExecutionPlan
    steps: list[AgentExecutionResult]
    final_output: str


AGENT_INTENT_KEYWORDS = {
    INCIDENT_ANALYSIS_AGENT.name: [
        "장애",
        "알람",
        "에러",
        "오류",
        "이상",
        "원인",
        "분석",
        "고장",
        "문제",
        "incident",
    ],
    KNOWLEDGE_SEARCH_AGENT.name: [
        "매뉴얼",
        "manual",
        "sop",
        "작업절차",
        "작업 절차",
        "가이드",
        "문서 찾아",
        "절차 찾아",
        "문서 검색",
        "지식 검색",
    ],
    MAINTENANCE_RECOMMENDATION_AGENT.name: [
        "유지보수",
        "정비",
        "예방 정비",
        "예방정비",
        "교체",
        "예방조치",
        "점검 방법",
        "정비 방법",
        "유지보수 방법",
        "예방 정비 방법",
        "예방정비 방법",
    ],
    REPORT_AGENT.name: [
        "보고서",
        "리포트",
        "보고 형식",
        "보고서 형식",
        "요약해줘",
        "정리해줘",
        "보고용",
    ],
}

SINGLE_AGENT_ROUTING_PRIORITY = [
    REPORT_AGENT.name,
    MAINTENANCE_RECOMMENDATION_AGENT.name,
    KNOWLEDGE_SEARCH_AGENT.name,
]

EXECUTION_ORDER = [
    INCIDENT_ANALYSIS_AGENT.name,
    KNOWLEDGE_SEARCH_AGENT.name,
    MAINTENANCE_RECOMMENDATION_AGENT.name,
    REPORT_AGENT.name,
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

    @staticmethod
    def _contains_any_keyword(
        message: str,
        keywords: list[str],
    ) -> bool:
        return any(
            keyword.lower() in message
            for keyword in keywords
        )

    def select_agent(
        self,
        message: str,
    ) -> str:
        normalized_message = message.lower()

        # TODO:
        # Current routing selects a single agent only.
        # Multi-agent execution plans will be introduced
        # in the orchestration workflow layer.
        for agent_name in SINGLE_AGENT_ROUTING_PRIORITY:
            if self._contains_any_keyword(
                normalized_message,
                AGENT_INTENT_KEYWORDS[agent_name],
            ):
                return agent_name

        return INCIDENT_ANALYSIS_AGENT.name

    def build_execution_plan(
        self,
        message: str,
    ) -> AgentExecutionPlan:
        normalized_message = message.lower()
        matched_agents: set[str] = set()

        for agent_name, keywords in AGENT_INTENT_KEYWORDS.items():
            if self._contains_any_keyword(
                normalized_message,
                keywords,
            ):
                matched_agents.add(agent_name)

        if not matched_agents:
            matched_agents.add(INCIDENT_ANALYSIS_AGENT.name)

        ordered_agents = [
            agent_name
            for agent_name in EXECUTION_ORDER
            if agent_name in matched_agents
        ]

        steps = []

        for index, agent_name in enumerate(
            ordered_agents,
            start=1,
        ):
            steps.append(
                AgentPlanStep(
                    step=index,
                    agent_name=agent_name,
                    depends_on=[] if index == 1 else [ordered_agents[index - 2]],
                )
            )

        return AgentExecutionPlan(
            agents=ordered_agents,
            primary_agent=self.select_agent(message),
            steps=steps,
            total_steps=len(steps),
        )

    @staticmethod
    def _extract_agent_output(
        result,
    ) -> str:
        return result.answer or ""

    @staticmethod
    def _build_agent_input(
        original_message: str,
        previous_results: list[AgentExecutionResult],
    ) -> str:
        if not previous_results:
            return original_message

        previous_result_lines = [
            "Original User Request:",
            original_message,
            "",
            "Previous Agent Results (reference context):",
        ]

        for previous_result in previous_results:
            previous_result_lines.extend(
                [
                    "",
                    f"[{previous_result.agent}]",
                    previous_result.output,
                ]
            )

        return "\n".join(previous_result_lines)

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

    async def run_execution_plan(
        self,
        message: str,
        max_steps: int = 5,
    ) -> MultiAgentExecutionResult:
        plan = self.build_execution_plan(message)
        db = self.session_factory()
        execution_steps: list[AgentExecutionResult] = []

        try:
            for step_index, agent_name in enumerate(
                plan.agents,
                start=1,
            ):
                agent_definition = AGENT_REGISTRY[agent_name]
                agent_input = self._build_agent_input(
                    original_message=message,
                    previous_results=execution_steps,
                )
                result = self.agent_service.run(
                    db=db,
                    agent_definition=agent_definition,
                    message=agent_input,
                    max_steps=max_steps,
                )
                execution_steps.append(
                    AgentExecutionResult(
                        step=step_index,
                        agent=agent_name,
                        output=self._extract_agent_output(result),
                    )
                )
        finally:
            db.close()

        return MultiAgentExecutionResult(
            plan=plan,
            steps=execution_steps,
            final_output=execution_steps[-1].output
            if execution_steps
            else "",
        )


DEFAULT_AGENT_ORCHESTRATOR = AgentOrchestrator(AgentService)
