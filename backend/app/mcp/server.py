from collections.abc import Callable
from typing import TypeVar

from mcp.server import MCPServer

from backend.app.core.database import SessionLocal
from backend.app.tools.incident_tools import (
    get_equipment_incidents as get_equipment_incidents_tool,
)
from backend.app.tools.incident_tools import (
    get_incident as get_incident_tool,
)
from backend.app.tools.incident_tools import (
    search_incidents as search_incidents_tool,
)


ToolResultT = TypeVar("ToolResultT")

mcp_server = MCPServer(
    name="Factory Agent MCP Server",
    instructions=(
        "Factory incident search and incident history tools for local "
        "development and integration testing."
    ),
)


def _run_with_db(
    tool_func: Callable[..., ToolResultT],
    **kwargs,
) -> ToolResultT:
    """기존 incident tool 호출에 필요한 DB 세션 수명주기를 감싼다."""
    db = SessionLocal()

    try:
        return tool_func(
            db=db,
            **kwargs,
        )
    finally:
        db.close()


@mcp_server.tool()
def search_incidents(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    자연어 query로 관련 incident를 검색한다.

    Parameters:
    - query: 검색에 사용할 자연어 질의
    - top_k: 반환할 최대 incident 개수

    Returns:
    - JSON serializable incident 목록
    """
    return _run_with_db(
        search_incidents_tool,
        query=query,
        top_k=top_k,
    )


@mcp_server.tool()
def get_incident(
    incident_id: int,
) -> dict:
    """
    incident_id로 특정 장애 상세 정보를 조회한다.

    Parameters:
    - incident_id: 조회할 incident의 고유 ID

    Returns:
    - JSON serializable incident 상세 정보
    """
    return _run_with_db(
        get_incident_tool,
        incident_id=incident_id,
    )


@mcp_server.tool()
def get_equipment_incidents(
    equipment_name: str,
) -> list[dict]:
    """
    equipment_name으로 해당 장비의 장애 이력을 조회한다.

    Parameters:
    - equipment_name: 조회할 장비 이름

    Returns:
    - 해당 장비의 JSON serializable incident 목록
    """
    return _run_with_db(
        get_equipment_incidents_tool,
        equipment_name=equipment_name,
    )


if __name__ == "__main__":
    mcp_server.run("stdio")
