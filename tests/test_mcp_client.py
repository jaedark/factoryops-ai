import pytest

from backend.app.core.database import SessionLocal
from backend.app.mcp.client import (
    MCPClientError,
    MCPClientService,
)
from backend.app.services.incident_service import IncidentService


def _seed_incidents() -> None:
    db = SessionLocal()

    try:
        IncidentService.seed_incidents(db)
    finally:
        db.close()


@pytest.mark.anyio
async def test_mcp_client_can_be_created():
    client = MCPClientService()

    assert client.is_connected is False


@pytest.mark.anyio
async def test_mcp_client_connects_over_stdio():
    async with MCPClientService() as client:
        assert client.is_connected is True


@pytest.mark.anyio
async def test_mcp_client_discovers_three_tools_dynamically():
    async with MCPClientService() as client:
        tools = await client.list_tools()

    names = [tool.name for tool in tools]

    assert names == [
        "search_incidents",
        "get_incident",
        "get_equipment_incidents",
    ]


@pytest.mark.anyio
async def test_mcp_client_tool_definitions_include_schema_metadata():
    async with MCPClientService() as client:
        tools = await client.list_tools()

    for tool in tools:
        assert tool.name
        assert isinstance(tool.description, str)
        assert isinstance(tool.input_schema, dict)
        assert tool.input_schema["type"] == "object"


@pytest.mark.anyio
async def test_mcp_client_calls_search_incidents():
    _seed_incidents()

    async with MCPClientService() as client:
        result = await client.call_tool(
            "search_incidents",
            {
                "query": "motor temperature issue",
                "top_k": 3,
            },
        )

    assert result.is_error is False
    assert len(result.data) > 0
    assert result.data[0]["equipment_name"] == "Conveyor-01"


@pytest.mark.anyio
async def test_mcp_client_calls_get_incident():
    _seed_incidents()

    db = SessionLocal()

    try:
        incident = IncidentService.get_equipment_incidents(
            db=db,
            equipment_name="Robot-01",
        )[0]
    finally:
        db.close()

    async with MCPClientService() as client:
        result = await client.call_tool(
            "get_incident",
            {"incident_id": incident.incident_id},
        )

    assert result.is_error is False
    assert result.data["incident_id"] == incident.incident_id
    assert result.data["equipment_name"] == "Robot-01"


@pytest.mark.anyio
async def test_mcp_client_calls_get_equipment_incidents():
    _seed_incidents()

    async with MCPClientService() as client:
        result = await client.call_tool(
            "get_equipment_incidents",
            {"equipment_name": "Vision-02"},
        )

    assert result.is_error is False
    assert len(result.data) == 1
    assert result.data[0]["equipment_name"] == "Vision-02"


@pytest.mark.anyio
async def test_mcp_client_returns_invalid_argument_error():
    async with MCPClientService() as client:
        result = await client.call_tool(
            "get_incident",
            {"incident_id": "wrong-type"},
        )

    assert result.is_error is True
    assert result.error is not None
    assert "Input should be a valid integer" in result.error


@pytest.mark.anyio
async def test_mcp_client_rejects_unknown_tool():
    async with MCPClientService() as client:
        with pytest.raises(MCPClientError) as exc:
            await client.call_tool(
                "delete_all_incidents",
                {},
            )

    assert "Tool not found" in str(exc.value)


@pytest.mark.anyio
async def test_mcp_client_close_clears_connection_state():
    client = MCPClientService()

    await client.connect()
    assert client.is_connected is True

    await client.close()

    assert client.is_connected is False

    with pytest.raises(MCPClientError) as exc:
        await client.list_tools()

    assert "not connected" in str(exc.value)


@pytest.mark.anyio
async def test_mcp_client_propagates_previous_connection_reuse():
    client = MCPClientService()

    await client.connect()
    first_connection_state = client.is_connected
    await client.connect()

    assert first_connection_state is True
    assert client.is_connected is True

    await client.close()
