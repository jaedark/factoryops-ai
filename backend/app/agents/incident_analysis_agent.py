from backend.app.agents.base import AgentDefinition


INCIDENT_ANALYSIS_AGENT = AgentDefinition(
    name="incident_analysis",
    description=(
        "Analyze incident symptoms, search similar incidents, "
        "inspect incident details, and review equipment history."
    ),
    system_instruction="""
You are the Incident Analysis Agent for FactoryOps AI.
Focus on understanding equipment symptoms and incident history.
Use search_incidents for symptom-based retrieval.
Use get_incident to inspect one chosen incident in detail.
Use get_equipment_incidents for equipment-specific incident history.
Answer in concise Korean grounded in the tool results.
""".strip(),
    allowed_tools=[
        "search_incidents",
        "get_incident",
        "get_equipment_incidents",
    ],
)
