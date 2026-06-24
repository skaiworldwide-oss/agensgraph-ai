"""03 · cypher operate — writes, safety modes, and real transports.

The operating side of the cypher MCP server: the write tool (with stats), read-only
mode (the write tool disappears), tool namespacing, and running the *actual* server
process over **stdio** and **Streamable HTTP** (what Claude Desktop and remote
deployments use) — driving the same tools over each.

    cd mcp-agensgraph/examples/demos
    .venv/bin/python 03_cypher_operate/ask.py

Run 01_model_and_load/build.py first.
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import clients, config, console

DB, GRAPH = "mcp_flights", "flights"
SERVER = str(pathlib.Path(sys.executable).parent / "mcp-agensgraph-cypher")


def _wait_port(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with contextlib.suppress(OSError):
            with socket.create_connection((host, port), timeout=1):
                return True
        time.sleep(0.2)
    return False


async def main() -> None:
    # ---- write tool + stats (idempotent; cleaned up so the graph stays as loaded) ----
    console.section("write_agensgraph_cypher — mutate + get stats")
    async with clients.cypher_client(DB, GRAPH) as cy:
        stats = clients.data(await cy.call_tool("write_agensgraph_cypher", {
            "query": 'MATCH (a:"Airport" {iata: \'JFK\'}), (b:"Airport" {iata: \'LHR\'}) '
                     'MERGE (a)-[r:"ROUTE" {airline: \'DEMO\'}]->(b) SET r += {equipment: \'B999\', stops: 0}',
        }))
        console.kv("write stats", stats)
        check = clients.data(await cy.call_tool("read_agensgraph_cypher", {
            "query": 'MATCH (:"Airport" {iata:\'JFK\'})-[r:"ROUTE" {airline:\'DEMO\'}]->() RETURN count(r) AS n'}))
        console.kv("DEMO route present", check["rows"][0]["n"])
        await cy.call_tool("write_agensgraph_cypher", {
            "query": 'MATCH (:"Airport")-[r:"ROUTE" {airline: \'DEMO\'}]->() DELETE r'})
        console.kv("cleaned up", "DEMO route deleted")

    # ---- read-only mode: the write tool is not even exposed ----
    console.section("Read-only mode — the write tool disappears")
    async with clients.cypher_client(DB, GRAPH, read_only=True) as cy:
        tools = sorted(t.name for t in await cy.list_tools())
        console.kv("tools (read-only server)", tools)
        console.kv("write tool present", "write_agensgraph_cypher" in tools)

    # ---- namespacing: prefix every tool (run many servers side by side) ----
    console.section("Namespacing — prefix tool names")
    async with clients.cypher_client(DB, GRAPH, namespace="ops") as cy:
        console.kv("tools (namespace='ops')", sorted(t.name for t in await cy.list_tools()))

    # ---- real transports: same tools over stdio and Streamable HTTP ----
    console.section("Transport: stdio (spawns the real server process)")
    env = config.server_env(DB, GRAPH)
    async with clients.stdio_client(SERVER, ["--transport", "stdio"], env) as cy:
        schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
        console.kv("stdio get_schema → labels", list(schema))
        page = clients.data(await cy.call_tool(
            "read_agensgraph_cypher", {"query": 'MATCH (a:"Airport") RETURN a.iata AS iata', "limit": 3}))
        console.kv("stdio read (page)", [r["iata"] for r in page["rows"]])

    console.section("Transport: Streamable HTTP (real server on a port, CORS + host middleware)")
    host, port = "127.0.0.1", 8769
    proc = subprocess.Popen(
        [SERVER, "--transport", "http", "--server-host", host, "--server-port", str(port),
         "--allow-origins", "https://example.com", "--allowed-hosts", "127.0.0.1,localhost"],
        env={**__import__("os").environ, **env},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_port(host, port):
            console.kv("http server", "did not start in time")
        else:
            async with clients.http_client(f"http://{host}:{port}/mcp/") as cy:
                tools = sorted(t.name for t in await cy.list_tools())
                console.kv("http tools", tools)
                schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
                console.kv("http get_schema → labels", list(schema))
            console.kv("middleware", "CORS(allow_origins=example.com) + TrustedHost(127.0.0.1,localhost)")
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        console.kv("http server", "stopped")


if __name__ == "__main__":
    asyncio.run(main())
