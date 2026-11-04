"""Manual SDK verification client; reads token privately from the environment."""

import asyncio
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters


async def main():
    names = ("REPOLENS_MCP_ACCESS_TOKEN", "DATABASE_URL", "SECRET_KEY", "APP_ENV", "PYTHONPATH")
    env = {k: os.environ[k] for k in names if k in os.environ}
    async with Client(
        StdioServerParameters(
            command=sys.executable, args=["-B", "-m", "repolens_mcp.server"], env=env
        ),
        read_timeout_seconds=10,
    ) as client:
        tools = await client.list_tools(cache_mode="bypass")
        print("Tools:", ", ".join(t.name for t in tools.tools))
        result = await client.call_tool("list_repositories", {"limit": 1})
        print("Invocation:", "error" if result.is_error else "success")


if __name__ == "__main__":
    asyncio.run(main())
