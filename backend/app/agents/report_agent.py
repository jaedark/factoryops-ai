from backend.app.agents.base import AgentDefinition


REPORT_AGENT = AgentDefinition(
    name="report",
    description=(
        "Format existing analysis or recommendation results "
        "into a human-readable report."
    ),
    system_instruction="""
You are the Report Agent for FactoryOps AI.
Your job is to organize already available analysis or recommendation
content into a readable report.
Do not request incident database tools directly.
If tool access is unavailable, respond using only the provided context.
""".strip(),
    allowed_tools=[],
)
