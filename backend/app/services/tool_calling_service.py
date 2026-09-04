from typing import Any

from google.genai import types
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from backend.app.schemas.agent import (
    AgentStatus,
    AgentTerminationReason,
    ApprovalActionResponse,
    ApprovalRequest,
)
from backend.app.services.guardrail_service import (
    ApprovalStore,
    GuardrailService,
    InMemoryApprovalStore,
)
from backend.app.services.llm_service import LlmService
from backend.app.tools.industrial_tools import (
    create_maintenance_request,
    get_equipment_status,
    get_equipment_telemetry,
    get_high_risk_equipment,
)
from backend.app.tools.incident_tools import (
    get_equipment_incidents,
    get_incident,
    search_incidents,
)


class SearchIncidentsArgs(BaseModel):
    query: str
    top_k: int = 3


class GetIncidentArgs(BaseModel):
    incident_id: int


class GetEquipmentIncidentsArgs(BaseModel):
    equipment_name: str


class GetEquipmentStatusArgs(BaseModel):
    equipment_id: str


class GetEquipmentTelemetryArgs(BaseModel):
    equipment_id: str


class GetHighRiskEquipmentArgs(BaseModel):
    pass


class CreateMaintenanceRequestArgs(BaseModel):
    equipment_id: str
    reason: str


class ToolCallResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str
    tool_called: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: Any | None = None
    status: AgentStatus | None = None
    termination_reason: AgentTerminationReason | None = None
    approval_request: ApprovalRequest | None = None


class ToolCallingService:
    _APPROVAL_STORE: ApprovalStore = InMemoryApprovalStore()
    _SYSTEM_INSTRUCTION = """
You are FactoryOps AI.
Choose at most one tool for each user request.
Use get_equipment_incidents for equipment-specific incident history requests.
Use search_incidents for symptom-based or similar-incident search.
Use get_incident for a specific incident ID or when you need to inspect one chosen incident in more detail after search results.
Use get_equipment_status to inspect current equipment condition, latest telemetry, and current risk reasons.
Use get_equipment_telemetry when you need telemetry history for one equipment.
Use get_high_risk_equipment when the user asks which equipment is currently risky.
Use create_maintenance_request only when the user explicitly asks to create a maintenance request.
If no tool is needed, answer directly.
After receiving a tool result, write a concise Korean answer grounded in the tool output.
""".strip()

    _TOOL_ARGUMENT_MODELS = {
        "search_incidents": SearchIncidentsArgs,
        "get_incident": GetIncidentArgs,
        "get_equipment_incidents": GetEquipmentIncidentsArgs,
        "get_equipment_status": GetEquipmentStatusArgs,
        "get_equipment_telemetry": GetEquipmentTelemetryArgs,
        "get_high_risk_equipment": GetHighRiskEquipmentArgs,
        "create_maintenance_request": CreateMaintenanceRequestArgs,
    }

    _TOOL_REGISTRY = {
        "search_incidents": search_incidents,
        "get_incident": get_incident,
        "get_equipment_incidents": get_equipment_incidents,
        "get_equipment_status": get_equipment_status,
        "get_equipment_telemetry": get_equipment_telemetry,
        "get_high_risk_equipment": get_high_risk_equipment,
        "create_maintenance_request": create_maintenance_request,
    }

    @classmethod
    def get_tool_registry(cls) -> dict[str, Any]:
        return cls._TOOL_REGISTRY

    @classmethod
    def get_approval_store(
        cls,
    ) -> ApprovalStore:
        return cls._APPROVAL_STORE

    @classmethod
    def build_generation_config(
        cls,
        system_instruction: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction
            or cls._SYSTEM_INSTRUCTION,
            tools=cls.build_tool_schemas(
                allowed_tools=allowed_tools
            ),
        )

    @classmethod
    def build_tool_schemas(
        cls,
        allowed_tools: list[str] | None = None,
    ) -> list[types.Tool]:
        if allowed_tools is None:
            selected_tools = set(cls._TOOL_REGISTRY.keys())
        else:
            selected_tools = set(allowed_tools)
        declarations = []

        if "search_incidents" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="search_incidents",
                    description=(
                        "자연어 증상이나 장애 상황을 기반으로 "
                        "관련 incident 후보를 검색한다. "
                        "반환 결과에는 incident_id가 포함되므로, "
                        "가장 관련 있는 항목을 고른 뒤 "
                        "추가 상세 확인이 필요하면 get_incident를 호출할 수 있다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "검색에 사용할 자연어 증상 또는 장애 설명"
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "반환할 incident 개수",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                )
            )

        if "get_incident" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="get_incident",
                    description=(
                        "Incident ID로 특정 장애의 상세 정보를 조회한다. "
                        "검색 결과 중 하나를 선택한 뒤 원인, 조치, 결과를 "
                        "다시 확인하거나 최종 답변 전에 단일 incident를 "
                        "명확히 검증할 때 사용한다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "incident_id": {
                                "type": "integer",
                                "description": "조회할 incident의 고유 ID",
                            },
                        },
                        "required": ["incident_id"],
                    },
                )
            )

        if "get_equipment_incidents" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="get_equipment_incidents",
                    description=(
                        "특정 equipment_name의 장애 이력을 조회한다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "equipment_name": {
                                "type": "string",
                                "description": "조회할 장비 이름",
                            },
                        },
                        "required": ["equipment_name"],
                    },
                )
            )

        if "get_equipment_status" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="get_equipment_status",
                    description=(
                        "현재 설비 상태를 조회한다. 설비 정보, 최신 "
                        "telemetry, 현재 고위험 여부와 위험 사유를 함께 "
                        "확인할 수 있다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "equipment_id": {
                                "type": "string",
                                "description": "조회할 설비 ID",
                            },
                        },
                        "required": ["equipment_id"],
                    },
                )
            )

        if "get_equipment_telemetry" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="get_equipment_telemetry",
                    description=(
                        "특정 설비의 telemetry 이력을 조회한다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "equipment_id": {
                                "type": "string",
                                "description": "조회할 설비 ID",
                            },
                        },
                        "required": ["equipment_id"],
                    },
                )
            )

        if "get_high_risk_equipment" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="get_high_risk_equipment",
                    description=(
                        "현재 고위험 설비 목록을 조회한다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {},
                    },
                )
            )

        if "create_maintenance_request" in selected_tools:
            declarations.append(
                types.FunctionDeclaration(
                    name="create_maintenance_request",
                    description=(
                        "정비 요청을 생성한다. 실제 자동 실행 전에 "
                        "반드시 사람 승인 절차가 필요하다."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "equipment_id": {
                                "type": "string",
                                "description": "정비 요청 대상 설비 ID",
                            },
                            "reason": {
                                "type": "string",
                                "description": "정비 요청 사유",
                            },
                        },
                        "required": [
                            "equipment_id",
                            "reason",
                        ],
                    },
                )
            )

        if not declarations:
            return []

        return [
            types.Tool(
                function_declarations=declarations
            )
        ]

    @classmethod
    def _validate_tool_call(
        cls,
        tool_name: str,
        tool_arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if tool_name not in cls._TOOL_REGISTRY:
            raise ValueError(f"Unsupported tool requested: {tool_name}")

        argument_model = cls._TOOL_ARGUMENT_MODELS[tool_name]

        try:
            validated = argument_model.model_validate(
                tool_arguments or {}
            )
        except ValidationError as exc:
            raise ValueError(
                f"Invalid arguments for tool '{tool_name}'"
            ) from exc

        return validated.model_dump()

    @classmethod
    def validate_tool_call(
        cls,
        tool_name: str,
        tool_arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return cls._validate_tool_call(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

    @classmethod
    def _execute_tool(
        cls,
        db: Session,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> Any:
        tool_function = cls._TOOL_REGISTRY[tool_name]
        return tool_function(
            db=db,
            **tool_arguments,
        )

    @classmethod
    def execute_tool(
        cls,
        db: Session,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> Any:
        return cls._execute_tool(
            db=db,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

    @classmethod
    def evaluate_guardrail(
        cls,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ):
        return GuardrailService.evaluate(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

    @classmethod
    def create_approval_request(
        cls,
        tool_name: str,
        tool_arguments: dict[str, Any],
        reason: str,
        session_id: str | None = None,
        approval_store: ApprovalStore | None = None,
    ) -> ApprovalRequest:
        store = approval_store or cls.get_approval_store()
        return store.create(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            reason=reason,
            session_id=session_id,
        )

    @classmethod
    def get_approval_request(
        cls,
        approval_id: str,
        approval_store: ApprovalStore | None = None,
    ) -> ApprovalRequest:
        store = approval_store or cls.get_approval_store()
        return store.get(approval_id)

    @classmethod
    def approve_tool_execution(
        cls,
        db: Session,
        approval_id: str,
        approval_store: ApprovalStore | None = None,
    ) -> ApprovalActionResponse:
        store = approval_store or cls.get_approval_store()
        approval_request = store.approve(approval_id)
        try:
            tool_result = cls.execute_tool(
                db=db,
                tool_name=approval_request.tool_name,
                tool_arguments=approval_request.tool_arguments,
            )
        except Exception as exc:
            store.mark_execution_failed(approval_id)
            raise RuntimeError(
                "Approved tool execution failed: "
                f"{approval_id}: {exc}"
            ) from exc
        approval_request = store.mark_executed(approval_id)

        return ApprovalActionResponse(
            approval_request=approval_request,
            tool_result=tool_result,
        )

    @classmethod
    def reject_tool_execution(
        cls,
        approval_id: str,
        approval_store: ApprovalStore | None = None,
    ) -> ApprovalActionResponse:
        store = approval_store or cls.get_approval_store()
        approval_request = store.reject(approval_id)

        return ApprovalActionResponse(
            approval_request=approval_request,
            tool_result=None,
        )

    @classmethod
    def execute_approved_tool(
        cls,
        db: Session,
        approval_id: str,
        approval_store: ApprovalStore | None = None,
    ) -> ApprovalActionResponse:
        approval_request = cls.get_approval_request(
            approval_id=approval_id,
            approval_store=approval_store,
        )

        if approval_request.status == "pending":
            raise ValueError(
                f"Approval request is still pending: {approval_id}"
            )
        if approval_request.status == "rejected":
            raise ValueError(
                f"Approval request was rejected: {approval_id}"
            )
        if approval_request.status == "executed":
            raise ValueError(
                f"Approval request was already executed: {approval_id}"
            )
        if approval_request.status == "execution_failed":
            raise ValueError(
                "Approval request execution already failed: "
                f"{approval_id}"
            )

        tool_result = cls.execute_tool(
            db=db,
            tool_name=approval_request.tool_name,
            tool_arguments=approval_request.tool_arguments,
        )
        store = approval_store or cls.get_approval_store()
        approval_request = store.mark_executed(approval_id)

        return ApprovalActionResponse(
            approval_request=approval_request,
            tool_result=tool_result,
        )

    @classmethod
    def _build_final_answer(
        cls,
        user_message: str,
        tool_response,
        tool_name: str,
        tool_result: Any,
    ) -> str:
        function_response_part = types.Part.from_function_response(
            name=tool_name,
            response={"output": tool_result},
        )

        final_response = LlmService.generate_content(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_message)],
                ),
                tool_response.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[function_response_part],
                ),
            ],
            config=cls.build_generation_config(),
        )

        return final_response.text

    @classmethod
    def chat(
        cls,
        db: Session,
        message: str,
    ) -> ToolCallResult:
        tool_response = LlmService.generate_content(
            contents=message,
            config=cls.build_generation_config(),
        )

        function_calls = tool_response.function_calls or []

        if not function_calls:
            return ToolCallResult(answer=tool_response.text)

        if len(function_calls) > 1:
            raise ValueError(
                "Only a single tool call is supported per request"
            )

        function_call = function_calls[0]
        tool_name = function_call.name
        tool_arguments = cls.validate_tool_call(
            tool_name=tool_name,
            tool_arguments=function_call.args,
        )
        guardrail_decision = cls.evaluate_guardrail(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

        if guardrail_decision.approval_required:
            approval_request = cls.create_approval_request(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                reason=guardrail_decision.reason,
            )
            return ToolCallResult(
                answer=guardrail_decision.reason,
                tool_called=tool_name,
                tool_arguments=tool_arguments,
                tool_result=None,
                status=AgentStatus.WAITING_APPROVAL,
                termination_reason=(
                    AgentTerminationReason.APPROVAL_REQUIRED
                ),
                approval_request=approval_request,
            )

        tool_result = cls.execute_tool(
            db=db,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )
        answer = cls._build_final_answer(
            user_message=message,
            tool_response=tool_response,
            tool_name=tool_name,
            tool_result=tool_result,
        )

        return ToolCallResult(
            answer=answer,
            tool_called=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            status=AgentStatus.COMPLETED,
            termination_reason=AgentTerminationReason.FINAL_ANSWER,
        )
