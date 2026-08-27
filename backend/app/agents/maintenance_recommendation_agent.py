from backend.app.agents.base import AgentDefinition


MAINTENANCE_RECOMMENDATION_AGENT = AgentDefinition(
    name="maintenance_recommendation",
    description=(
        "Recommend maintenance actions from incident causes, "
        "actions, and results."
    ),
    system_instruction="""
You are the Maintenance Recommendation Agent for FactoryOps AI.
Use historical incident evidence to recommend maintenance actions.
Use search_incidents to find relevant past cases.
Use get_incident to inspect a selected incident more closely.
Answer in concise Korean grounded in the tool results.
Do not invent unavailable maintenance-specific tools.
""".strip(),
    allowed_tools=[
        "search_incidents",
        "get_incident",
    ],
)
