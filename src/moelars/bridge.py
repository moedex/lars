"""`moelars mcp-bridge`: stdio MCP for clients that cannot speak Streamable HTTP.

Holds no model. Every tool call goes to a running `moelars serve` over its HTTP API, so any
number of agent sessions can start a bridge without loading the model again. When the
server is down or refuses the key, the tool call fails with a message that says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from moelars.mcp_server import build_mcp


def bridge_api_key(api_key_file: str | None = None) -> str | None:
    """The key to send: --api-key-file, then MOELARS_API_KEY / MOELARS_API_KEY_FILE, then the service's key."""
    from moelars.server import configured_api_key
    from moelars.service import API_KEY_FILE

    if api_key_file:
        return Path(api_key_file).read_text().strip() or None
    key = configured_api_key()
    if key:
        return key
    return API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists() else None


def make_bridge(url: str, api_key: str | None, transport: httpx.AsyncBaseTransport | None = None) -> Any:
    from mcp.server.mcpserver.exceptions import ToolError

    base = url.rstrip("/")
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    # The first request after an idle spell loads the model; allow for a cold start from disk.
    client = httpx.AsyncClient(base_url=base, headers=headers, timeout=180.0, transport=transport)

    async def call(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.TransportError as error:
            raise ToolError(f"moe-LARS is not reachable at {base} ({error.__class__.__name__}); "
                            "is the service running? `moelars service status`") from error
        if response.status_code == 401:
            raise ToolError("moe-LARS refused the API key; pass --api-key-file or set MOELARS_API_KEY")
        body = response.json()
        if response.status_code >= 400:
            raise ToolError(body.get("message", response.text))
        return body

    async def decide(request: dict[str, Any]) -> dict[str, Any]:
        return await call("POST", "/v1/systemone", json=request)

    async def status() -> dict[str, Any]:
        return await call("GET", "/v1/status")

    return build_mcp(decide, status)


def run_bridge(url: str, api_key_file: str | None = None) -> None:
    make_bridge(url, bridge_api_key(api_key_file)).run("stdio")

