import json

import pytest
from mcp import Client

from backend.app.core.database import SessionLocal
from backend.app.mcp.server import mcp_server
from backend.app.services.incident_service import IncidentService


def _seed_incidents() -> None:
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)
    finally:
        db.close()


def _extract_json_payload(result) -> object:
    assert result.is_error is False

    structured_content = getattr(
        result,
        "structured_content",
        None,
    )

    if structured_content is not None:
        if (
            isinstance(structured_content, dict)
            and "result" in structured_content
        ):
            return structured_content["result"]

        return structured_content

    payloads = [
        json.loads(content.text)
        for content in result.content
    ]

    if len(payloads) == 1:
        return payloads[0]

    return payloads


@pytest.mark.anyio
async def test_mcp_server_can_be_created():
    assert mcp_server.name == "Factory Agent MCP Server"


@pytest.mark.anyio
async def test_mcp_server_lists_three_incident_tools():
    async with Client(mcp_server) as client:
        result = await client.list_tools()

    names = [tool.name for tool in result.tools]

    assert names == [
        "search_incidents",
        "get_incident",
        "get_equipment_incidents",
    ]


@pytest.mark.anyio
async def test_mcp_search_incidents_call_succeeds():
    _seed_incidents()

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "search_incidents",
            {
                "query": "motor temperature issue",
                "top_k": 3,
            },
        )

    payload = _extract_json_payload(result)

    assert len(payload) > 0
    assert payload[0]["equipment_name"] == "Conveyor-01"
    assert isinstance(payload[0]["incident_id"], int)


@pytest.mark.anyio
async def test_mcp_get_incident_call_succeeds():
    _seed_incidents()

    db = SessionLocal()

    try:
        incident = IncidentService.get_equipment_incidents(
            db=db,
            equipment_name="Robot-01",
        )[0]
    finally:
        db.close()

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_incident",
            {"incident_id": incident.incident_id},
        )

    payload = _extract_json_payload(result)

    assert payload["incident_id"] == incident.incident_id
    assert payload["equipment_name"] == "Robot-01"


@pytest.mark.anyio
async def test_mcp_get_equipment_incidents_call_succeeds():
    _seed_incidents()

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_equipment_incidents",
            {"equipment_name": "Vision-02"},
        )

    payload = _extract_json_payload(result)

    assert len(payload) == 1
    assert payload[0]["equipment_name"] == "Vision-02"


@pytest.mark.anyio
async def test_mcp_tool_rejects_invalid_arguments():
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "get_incident",
            {"incident_id": "wrong-type"},
        )

    assert result.is_error is True
    assert "Input should be a valid integer" in result.content[0].text
