"""Shared transport bootstrap for the AgensGraph MCP servers.

One ``run_server`` for stdio / Streamable HTTP / SSE, with CORS + DNS-rebinding
(TrustedHost) middleware applied to the HTTP transports.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

logger = logging.getLogger("mcp_agensgraph_common")


def build_middleware(
    allow_origins: Sequence[str], allowed_hosts: Sequence[str]
) -> list[Middleware]:
    """CORS + TrustedHost middleware for the HTTP/SSE transports."""
    return [
        Middleware(
            CORSMiddleware,
            allow_origins=list(allow_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        ),
        Middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts)),
    ]


async def run_server(
    mcp,
    *,
    transport: str = "stdio",
    host: Optional[str] = None,
    port: Optional[int] = None,
    path: Optional[str] = None,
    allow_origins: Sequence[str] = (),
    allowed_hosts: Sequence[str] = (),
    server_name: str = "AgensGraph MCP",
) -> None:
    """Run a FastMCP server over the selected transport."""
    if transport == "stdio":
        logger.info("Running %s over stdio", server_name)
        await mcp.run_stdio_async()
        return

    if transport not in ("http", "sse"):
        raise ValueError(
            f"Invalid transport: {transport!r}. Must be 'stdio', 'http', or 'sse'."
        )

    middleware = build_middleware(allow_origins, allowed_hosts)
    http_kwargs = dict(host=host, port=port, path=path, middleware=middleware)

    if transport == "http":
        logger.info("Running %s over Streamable HTTP on %s:%s%s", server_name, host, port, path)
        await mcp.run_http_async(stateless_http=True, **http_kwargs)
    else:  # sse
        logger.warning(
            "Running %s over SSE on %s:%s%s — SSE is deprecated; prefer 'http'.",
            server_name, host, port, path,
        )
        await mcp.run_http_async(transport="sse", **http_kwargs)
