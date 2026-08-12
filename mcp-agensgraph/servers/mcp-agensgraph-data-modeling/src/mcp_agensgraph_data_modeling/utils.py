"""Data-modeling-server utilities.

Config parsing comes from the shared ``mcp_agensgraph_common`` core. This server holds no
connection, so it needs only the transport section.

Quoting belongs to the driver, and to the point where a name is placed into a statement rather
than to a pass over the finished text: see ``agensgraph.cypher.quote_identifier``.
"""

from __future__ import annotations

import argparse
from typing import Any

from mcp_agensgraph_common.config import format_namespace, transport_config

__all__ = ["format_namespace", "process_config"]


def process_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the config dict for ``server.main`` — transport only (no DB)."""
    return transport_config(args)
