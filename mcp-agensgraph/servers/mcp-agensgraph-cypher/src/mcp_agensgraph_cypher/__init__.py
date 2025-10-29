import argparse
import asyncio

from . import server
from .utils import process_config


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="Agensgraph Cypher MCP Server")
    parser.add_argument("--db-url", default=None, help="Agensgraph connection URL")
    parser.add_argument("--username", default=None, help="Agensgraph username")
    parser.add_argument("--password", default=None, help="Agensgraph password")
    parser.add_argument("--database", default=None, help="Agensgraph database name")
    parser.add_argument("--graphname", default=None, help="Agensgraph graph name")
    parser.add_argument(
        "--transport", default=None, help="Transport type (stdio, sse, http)"
    )
    parser.add_argument("--namespace", default=None, help="Tool namespace")
    parser.add_argument(
        "--server-path", default=None, help="HTTP path (default: /mcp/)"
    )
    parser.add_argument("--server-host", default=None, help="Server host")
    parser.add_argument("--server-port", type=int, default=None, help="Server port")
    parser.add_argument(
        "--allow-origins",
        default=None,
        help="Allow origins for remote servers (comma-separated list)",
    )
    parser.add_argument(
        "--allowed-hosts",
        default=None,
        help="Allowed hosts for DNS rebinding protection on remote servers(comma-separated list)",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for read queries (default: 30)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Allow only read-only queries (default: False)",
    )
    parser.add_argument("--token-limit", type=int, default=None, help="Response token limit")

    args = parser.parse_args()
    config = process_config(args)
    asyncio.run(server.main(**config))


__all__ = ["main", "server"]
