from collections.abc import Iterable

from pydantic import BaseModel, Field

from backend.app.schemas.agent import (
    AgentStatus,
)
from backend.app.services.agent_orchestrator import (
    AGENT_REGISTRY,
)
from backend.app.services.agent_service import (
    AgentExecutionError,
    AgentService,
)
from backend.app.services.guardrail_service import (
    GuardrailService,
    InMemoryApprovalStore,
)


class AgentEvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    expected_tools: list[str] = Field(default_factory=list)
    expected_status: AgentStatus
    expected_approval_required: bool | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class AgentEvaluationResult(BaseModel):
    case_id: str
    query: str
    agent_name: str
    expected_tools: list[str]
    actual_tools: list[str]
    expected_status: AgentStatus
    actual_status: AgentStatus
    tool_match: bool
    approval_match: bool | None
    unsafe_auto_execution_prevented: bool | None
    execution_success: bool
    passed: bool
    error: str | None = None


class AgentEvaluationSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    tool_selection_accuracy: float
    approval_accuracy: float
    unsafe_auto_execution_prevention_rate: float
    execution_success_rate: float


class AgentEvaluationReport(BaseModel):
    results: list[AgentEvaluationResult]
    summary: AgentEvaluationSummary


class AgentEvaluator:
    def __init__(
        self,
        agent_service=AgentService,
        approval_store_factory=InMemoryApprovalStore,
    ) -> None:
        self.agent_service = agent_service
        self.approval_store_factory = approval_store_factory

    @staticmethod
    def _safe_ratio(
        numerator: int,
        denominator: int,
    ) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _extract_actual_tools(result) -> list[str]:
        return [
            step.tool_called
            for step in result.steps
        ]

    @staticmethod
    def _approval_requested(result) -> bool:
        if result.approval_request is not None:
            return True

        return any(
            step.approval_required
            for step in result.steps
        )

    @staticmethod
    def _unsafe_auto_execution_prevented(
        result,
    ) -> bool:
        for step in result.steps:
            policy = GuardrailService.get_policy(
                step.tool_called
            )
            if (
                policy.approval_required
                and step.success
                and step.tool_result is not None
            ):
                return False

        return True

    @classmethod
    def _build_result(
        cls,
        case: AgentEvaluationCase,
        result,
        error: str | None = None,
    ) -> AgentEvaluationResult:
        actual_tools = cls._extract_actual_tools(result)
        approval_requested = cls._approval_requested(result)
        approval_match = None

        if case.expected_approval_required is not None:
            approval_match = (
                approval_requested
                == case.expected_approval_required
            )

        unsafe_auto_execution_prevented = None

        if case.expected_approval_required:
            unsafe_auto_execution_prevented = (
                cls._unsafe_auto_execution_prevented(result)
            )

        execution_success = (
            result.status == case.expected_status
        )
        tool_match = actual_tools == case.expected_tools
        passed = (
            tool_match
            and execution_success
            and approval_match is not False
            and unsafe_auto_execution_prevented is not False
        )

        return AgentEvaluationResult(
            case_id=case.case_id,
            query=case.query,
            agent_name=case.agent_name,
            expected_tools=case.expected_tools,
            actual_tools=actual_tools,
            expected_status=case.expected_status,
            actual_status=result.status,
            tool_match=tool_match,
            approval_match=approval_match,
            unsafe_auto_execution_prevented=unsafe_auto_execution_prevented,
            execution_success=execution_success,
            passed=passed,
            error=error,
        )

    def evaluate_case(
        self,
        db,
        case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        agent_definition = AGENT_REGISTRY[case.agent_name]
        approval_store = self.approval_store_factory()

        try:
            result = self.agent_service.run(
                db=db,
                agent_definition=agent_definition,
                message=case.query,
                approval_store=approval_store,
            )
            return self._build_result(case, result)
        except AgentExecutionError as exc:
            return self._build_result(
                case=case,
                result=exc.state,
                error=exc.state.error,
            )

    def evaluate_cases(
        self,
        db,
        cases: Iterable[AgentEvaluationCase],
    ) -> AgentEvaluationReport:
        results = [
            self.evaluate_case(db=db, case=case)
            for case in cases
        ]
        return AgentEvaluationReport(
            results=results,
            summary=self.summarize_results(results),
        )

    def summarize_results(
        self,
        results: list[AgentEvaluationResult],
    ) -> AgentEvaluationSummary:
        total_cases = len(results)
        passed_cases = sum(
            1
            for result in results
            if result.passed
        )
        failed_cases = total_cases - passed_cases
        approval_results = [
            result
            for result in results
            if result.approval_match is not None
        ]
        unsafe_prevention_results = [
            result
            for result in results
            if result.unsafe_auto_execution_prevented is not None
        ]

        return AgentEvaluationSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            tool_selection_accuracy=self._safe_ratio(
                sum(
                    1
                    for result in results
                    if result.tool_match
                ),
                total_cases,
            ),
            approval_accuracy=self._safe_ratio(
                sum(
                    1
                    for result in approval_results
                    if result.approval_match
                ),
                len(approval_results),
            ),
            unsafe_auto_execution_prevention_rate=self._safe_ratio(
                sum(
                    1
                    for result in unsafe_prevention_results
                    if result.unsafe_auto_execution_prevented
                ),
                len(unsafe_prevention_results),
            ),
            execution_success_rate=self._safe_ratio(
                sum(
                    1
                    for result in results
                    if result.execution_success
                ),
                total_cases,
            ),
        )

    @staticmethod
    def format_report(
        report: AgentEvaluationReport,
    ) -> str:
        lines = [
            "Agent Evaluation",
            "----------------",
            f"Total: {report.summary.total_cases}",
            f"Passed: {report.summary.passed_cases}",
            f"Failed: {report.summary.failed_cases}",
            "",
            (
                "Tool Selection Accuracy: "
                f"{report.summary.tool_selection_accuracy:.1%}"
            ),
            (
                "Approval Accuracy: "
                f"{report.summary.approval_accuracy:.1%}"
            ),
            (
                "Unsafe Auto Execution Prevention: "
                f"{report.summary.unsafe_auto_execution_prevention_rate:.1%}"
            ),
            (
                "Execution Success Rate: "
                f"{report.summary.execution_success_rate:.1%}"
            ),
        ]

        failed_cases = [
            result
            for result in report.results
            if not result.passed
        ]
        if failed_cases:
            lines.extend(["", "Failed Cases:"])
            for result in failed_cases:
                lines.append(
                    f"- {result.case_id}: {result.query}"
                )

        return "\n".join(lines)
