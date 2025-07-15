import argparse
import asyncio
import os

from . import server


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="Agensgraph Cypher MCP Server")
    parser.add_argument("--db-name", default=None, help="Agensgraph database name")
    parser.add_argument("--username", default=None, help="Agensgraph username")
    parser.add_argument("--password", default=None, help="Agensgraph password")
    parser.add_argument("--db-host", default=None, help="Agensgraph server host")
    parser.add_argument("--db-port", default=None, help="Agensgraph server port")
    parser.add_argument("--graph-name", default=None, help="Graph name")
    parser.add_argument("--transport", default=None, help="Transport type")
    parser.add_argument("--namespace", default=None, help="Tool namespace")
    parser.add_argument("--server-host", default=None, help="Server host")
    parser.add_argument("--server-port", default=None, help="Server port")

    args = parser.parse_args()
    asyncio.run(
        server.main(
            args.db_name or os.getenv("AGENSGRAPH_DB"),
            args.username or os.getenv("AGENSGRAPH_USERNAME"),
            args.password or os.getenv("AGENSGRAPH_PASSWORD"),
            args.graph_name or os.getenv("AGENSGRAPH_GRAPH_NAME"),
            args.db_host or os.getenv("AGENSGRAPH_HOST", "localhost"),
            args.db_port or os.getenv("AGENSGRAPH_PORT", 5432),  
            args.transport or os.getenv("AGENSGRAPH_TRANSPORT", "stdio"),
            args.namespace or os.getenv("AGENSGRAPH_NAMESPACE", ""),
            args.server_host or os.getenv("AGENSGRAPH_MCP_SERVER_HOST", "127.0.0.1"),
            args.server_port or os.getenv("AGENSGRAPH_MCP_SERVER_PORT", 8000),
        )
    )


__all__ = ["main", "server"]
