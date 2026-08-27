from backend.app.agents.base import AgentDefinition


KNOWLEDGE_SEARCH_AGENT = AgentDefinition(
    name="knowledge_search",
    description=(
        "Reserved agent for future manual, SOP, and knowledge search."
    ),
    system_instruction="""
You are the Knowledge Search Agent for FactoryOps AI.
Your responsibility is future manual, SOP, and knowledge lookup.
If no tools are available, respond based only on the conversation.
Do not invent unavailable tools.
""".strip(),
    allowed_tools=[],
)
