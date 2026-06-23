"""Cypher-server configuration — composed from the shared core.

The connection/transport/read-control parsing lives in ``mcp_agensgraph_common``;
this server just stitches the three sections together.
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp_agensgraph_common.config import (
    connection_config,
    read_controls,
    transport_config,
)


def process_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the full config dict for ``server.main`` from CLI args + env vars."""
    return {
        **connection_config(args, default_graphname="agens"),
        **transport_config(args),
        **read_controls(args),
    }
