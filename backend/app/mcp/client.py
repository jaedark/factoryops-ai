import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel, ConfigDict


class MCPClientError(RuntimeError):
    """MCP client 연결 및 호출 과정의 애플리케이션 예외."""


class MCPToolDefinition(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class MCPToolCallResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    data: Any | None = None
    is_error: bool = False
    error: str | None = None


class MCPClientService:
    def __init__(
        self,
        python_executable: str | None = None,
        project_root: str | Path | None = None,
        server_module: str = "backend.app.mcp.server",
    ) -> None:
        self._project_root = Path(
            project_root
            or Path(__file__).resolve().parents[3]
        )
        self._python_executable = python_executable or sys.executable
        self._server_module = server_module
        self._client_context = None
        self._client = None
        self._tool_cache: list[MCPToolDefinition] | None = None

    @property
    def is_connected(
        self,
    ) -> bool:
        return self._client is not None

    def _build_server_env(
        self,
    ) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH")
        project_root = str(self._project_root)

        if pythonpath:
            env["PYTHONPATH"] = (
                f"{project_root}{os.pathsep}{pythonpath}"
            )
        else:
            env["PYTHONPATH"] = project_root

        return env

    def _build_server_parameters(
        self,
    ) -> StdioServerParameters:
        return StdioServerParameters(
            command=self._python_executable,
            args=["-m", self._server_module],
            env=self._build_server_env(),
            cwd=self._project_root,
        )

    async def connect(
        self,
    ) -> "MCPClientService":
        if self.is_connected:
            return self

        try:
            self._client_context = Client(
                self._build_server_parameters()
            )
            self._client = await self._client_context.__aenter__()
        except Exception as exc:
            await self.close()
            raise MCPClientError(
                "Failed to connect to MCP server"
            ) from exc

        return self

    def _require_connected(
        self,
    ):
        if self._client is None:
            raise MCPClientError(
                "MCP client is not connected"
            )

        return self._client

    @staticmethod
    def _normalize_tool_result_data(
        result,
    ) -> Any:
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

    @staticmethod
    def _normalize_tool_error(
        result,
    ) -> str:
        return "\n".join(
            content.text
            for content in getattr(result, "content", [])
        ).strip()

    async def list_tools(
        self,
    ) -> list[MCPToolDefinition]:
        client = self._require_connected()

        try:
            result = await client.list_tools()
        except Exception as exc:
            raise MCPClientError(
                "Failed to list MCP tools"
            ) from exc

        tools = [
            MCPToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in result.tools
        ]
        self._tool_cache = tools
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolCallResult:
        client = self._require_connected()
        tools = self._tool_cache or await self.list_tools()

        if tool_name not in {
            tool.name
            for tool in tools
        }:
            raise MCPClientError(
                f"Tool not found: {tool_name}"
            )

        try:
            result = await client.call_tool(
                tool_name,
                arguments or {},
            )
        except Exception as exc:
            raise MCPClientError(
                f"Failed to call MCP tool: {tool_name}"
            ) from exc

        if result.is_error:
            return MCPToolCallResult(
                tool_name=tool_name,
                is_error=True,
                error=self._normalize_tool_error(result),
            )

        return MCPToolCallResult(
            tool_name=tool_name,
            data=self._normalize_tool_result_data(result),
            is_error=False,
        )

    async def close(
        self,
    ) -> None:
        if self._client_context is not None:
            try:
                await self._client_context.__aexit__(
                    None,
                    None,
                    None,
                )
            finally:
                self._client_context = None
                self._client = None
                self._tool_cache = None
        else:
            self._client = None
            self._tool_cache = None

    async def __aenter__(
        self,
    ) -> "MCPClientService":
        return await self.connect()

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        await self.close()
