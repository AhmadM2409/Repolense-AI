"""Official MCP SDK 2.3 stdio server. One trusted principal per process."""

import asyncio
import json
import logging
import os
import sys

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError


async def serve(read, write):
    # Imports and all application activity occur after the SDK claims stdout.
    from fastapi import HTTPException
    from pydantic import ValidationError

    from core.database import AsyncSessionLocal, engine
    from repolens_mcp.auth import principal
    from repolens_mcp.tools import SCHEMAS, execute

    engine.echo = False
    token = os.environ.pop("REPOLENS_MCP_ACCESS_TOKEN", "")
    async with AsyncSessionLocal() as db:
        uid = await principal(db, token)
        await db.rollback()
    slots = asyncio.Semaphore(2)

    async def discovery(ctx, params):
        try:
            async with asyncio.timeout(5), slots, AsyncSessionLocal() as db:
                await principal(db, token, uid)
                await db.rollback()
            return types.ListToolsResult(
                tools=[
                    types.Tool(
                        name=name,
                        description="Read-only owner-scoped " + name.replace("_", " "),
                        input_schema=model.model_json_schema(),
                    )
                    for name, model in SCHEMAS.items()
                ]
            )
        except Exception:
            raise MCPError(code=-32001, message="Authentication or service unavailable") from None

    async def invocation(ctx, params):
        try:
            async with asyncio.timeout(5), slots, AsyncSessionLocal() as db:
                await principal(db, token, uid)
                if params.name not in SCHEMAS:
                    return failure("unknown_tool")
                args = SCHEMAS[params.name].model_validate(params.arguments or {})
                result = await execute(db, uid, params.name, args)
                await principal(db, token, uid)
                payload = json.dumps(result, default=str, separators=(",", ":"))
                if len(payload.encode()) > 32768:
                    return failure("result_limit")
                await db.rollback()
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=payload)],
                    structured_content=json.loads(payload),
                    is_error=False,
                )
        except ValidationError:
            return failure("invalid_arguments")
        except HTTPException as exc:
            return failure(
                {
                    401: "unauthorized",
                    404: "not_found",
                    422: "invalid_arguments",
                    413: "result_limit",
                    409: "source_integrity",
                }.get(exc.status_code, "unavailable")
            )
        except TimeoutError:
            return failure("timeout")
        except Exception:
            return failure("unavailable")

    server = Server(
        "RepoLens AI", version="0.1.0", on_list_tools=discovery, on_call_tool=invocation
    )
    try:
        await server.run(read, write, server.create_initialization_options())
    finally:
        await engine.dispose()


def failure(code):
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps({"error": code}))],
        structured_content={"error": code},
        is_error=True,
    )


class CappedInput:
    def __aiter__(self):
        return self

    async def __anext__(self):
        import anyio

        line = await anyio.to_thread.run_sync(sys.stdin.buffer.readline, 16385)
        if not line:
            raise StopAsyncIteration
        if len(line) > 16384:
            raise ValueError("MCP frame exceeds limit")
        return line.decode("utf-8", errors="strict")


async def main():
    logging.disable(logging.CRITICAL)
    try:
        async with stdio_server(stdin=CappedInput()) as (read, write):
            await serve(read, write)
    except Exception:
        sys.stderr.write("RepoLens MCP unavailable\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
