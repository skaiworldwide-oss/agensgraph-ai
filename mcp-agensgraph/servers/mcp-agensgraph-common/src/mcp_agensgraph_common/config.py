"""Shared configuration parsing for the AgensGraph MCP servers.

Merges CLI arguments (highest priority) with environment variables, falling back
to documented defaults. Composable so each server pulls only the sections it needs:

- ``connection_config``  — db_url / username / password / database / graphname
- ``transport_config``   — namespace / transport / host / port / path / origins / hosts
- ``read_controls``      — read_timeout / token_limit / read_only  (cypher only)

Canonical environment variables (standardized across all servers):

    AGENSGRAPH_URL            AGENSGRAPH_USERNAME   AGENSGRAPH_PASSWORD
    AGENSGRAPH_DATABASE       AGENSGRAPH_GRAPHNAME  AGENSGRAPH_NAMESPACE
    AGENSGRAPH_TRANSPORT      AGENSGRAPH_MCP_SERVER_HOST / _PORT / _PATH
    AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS / _ALLOWED_HOSTS
    AGENSGRAPH_READ_TIMEOUT   AGENSGRAPH_RESPONSE_TOKEN_LIMIT   AGENSGRAPH_READ_ONLY
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Optional, Union

logger = logging.getLogger("mcp_agensgraph_common")


def parse_boolean_safely(value: Union[str, bool]) -> bool:
    """Parse a string/bool to bool with strict ``true``/``false`` validation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Invalid boolean value: {value!r}. Must be 'true' or 'false'")


def format_namespace(namespace: str) -> str:
    """Format a tool namespace with a trailing dash if non-empty."""
    if not namespace:
        return ""
    return namespace if namespace.endswith("-") else f"{namespace}-"


def _pick(arg: Any, *env_vars: str, default: Any = None) -> Any:
    """CLI arg (if not None) → first set env var → default."""
    if arg is not None:
        return arg
    for name in env_vars:
        val = os.getenv(name)
        if val is not None:
            return val
    return default


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def connection_config(
    args: argparse.Namespace, *, default_graphname: str = "agens"
) -> dict[str, Any]:
    """Connection settings for DB-backed servers (cypher, memory)."""
    cfg: dict[str, Any] = {}
    cfg["db_url"] = _pick(
        getattr(args, "db_url", None),
        "AGENSGRAPH_URL",
        "AGENSGRAPH_URI",
        default="postgresql://localhost:5432",
    )
    cfg["username"] = _pick(
        getattr(args, "username", None), "AGENSGRAPH_USERNAME", default="agens"
    )
    cfg["password"] = _pick(
        getattr(args, "password", None), "AGENSGRAPH_PASSWORD", default="agens"
    )
    cfg["database"] = _pick(
        getattr(args, "database", None), "AGENSGRAPH_DATABASE", "AGENSGRAPH_DB",
        default="agens",
    )
    cfg["graphname"] = _pick(
        getattr(args, "graphname", None),
        "AGENSGRAPH_GRAPHNAME",
        "AGENSGRAPH_GRAPH_NAME",
        default=default_graphname,
    )
    return cfg


def transport_config(args: argparse.Namespace) -> dict[str, Any]:
    """Transport + HTTP server settings, shared by all three servers."""
    cfg: dict[str, Any] = {}
    cfg["namespace"] = _pick(
        getattr(args, "namespace", None), "AGENSGRAPH_NAMESPACE", default=""
    )
    transport = _pick(
        getattr(args, "transport", None), "AGENSGRAPH_TRANSPORT", default="stdio"
    )
    cfg["transport"] = transport
    is_stdio = transport == "stdio"

    host = _pick(getattr(args, "server_host", None), "AGENSGRAPH_MCP_SERVER_HOST")
    cfg["host"] = host if host is not None else (None if is_stdio else "127.0.0.1")

    port = _pick(getattr(args, "server_port", None), "AGENSGRAPH_MCP_SERVER_PORT")
    if port is not None:
        cfg["port"] = int(port)
    else:
        cfg["port"] = None if is_stdio else 8000

    path = _pick(getattr(args, "server_path", None), "AGENSGRAPH_MCP_SERVER_PATH")
    cfg["path"] = path if path is not None else (None if is_stdio else "/mcp/")

    origins = getattr(args, "allow_origins", None)
    if origins is not None:
        cfg["allow_origins"] = _split_csv(origins)
    else:
        cfg["allow_origins"] = _split_csv(
            os.getenv("AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS")
        )

    hosts = getattr(args, "allowed_hosts", None)
    if hosts is not None:
        cfg["allowed_hosts"] = _split_csv(hosts)
    elif os.getenv("AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS") is not None:
        cfg["allowed_hosts"] = _split_csv(
            os.getenv("AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS")
        )
    else:
        # Secure default: only localhost when not explicitly configured.
        cfg["allowed_hosts"] = ["localhost", "127.0.0.1"]

    if not is_stdio and host is None and getattr(args, "server_host", None) is None:
        logger.info("No server host provided for %s transport; using 127.0.0.1", transport)
    return cfg


def read_controls(args: argparse.Namespace) -> dict[str, Any]:
    """Read-query controls (cypher server): timeout, token limit, read-only."""
    cfg: dict[str, Any] = {}

    read_timeout = _pick(getattr(args, "read_timeout", None), "AGENSGRAPH_READ_TIMEOUT")
    try:
        cfg["read_timeout"] = int(read_timeout) if read_timeout is not None else 30
    except (TypeError, ValueError):
        logger.warning("Invalid read timeout %r; using default 30s", read_timeout)
        cfg["read_timeout"] = 30

    token_limit = _pick(
        getattr(args, "token_limit", None), "AGENSGRAPH_RESPONSE_TOKEN_LIMIT"
    )
    cfg["token_limit"] = int(token_limit) if token_limit is not None else None

    if getattr(args, "read_only", False):
        cfg["read_only"] = True
    else:
        env_ro = os.getenv("AGENSGRAPH_READ_ONLY")
        cfg["read_only"] = parse_boolean_safely(env_ro) if env_ro is not None else False

    return cfg
