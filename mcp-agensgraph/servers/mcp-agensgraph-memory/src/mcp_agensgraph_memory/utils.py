"""Memory-server configuration — composed from the shared core.

The connection/transport parsing lives in ``mcp_agensgraph_common``; the memory
server needs the connection + transport sections (no read-only controls). Its graph
defaults to ``memory``.
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp_agensgraph_common.config import (
    connection_config,
    format_namespace,  # re-exported for back-compat
    transport_config,
)

__all__ = ["process_config", "format_namespace"]


def process_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the config dict for ``server.main`` from CLI args + env vars."""
    return {
        **connection_config(args, default_graphname="memory"),
        **transport_config(args),
    }
