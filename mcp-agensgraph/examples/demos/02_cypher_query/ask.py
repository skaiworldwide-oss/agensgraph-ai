"""02 · cypher query — read the flights graph through the cypher MCP server.

Exercises the cypher server's read surface against the OpenFlights graph that
01_model_and_load built: schema introspection, multi-hop reads, vertex/edge parsing,
**pagination** through tens of thousands of rows, read-only enforcement, and the
timeout / token-limit knobs.

    cd mcp-agensgraph/examples/demos
    .venv/bin/python 02_cypher_query/ask.py

Run 01_model_and_load/build.py first — this reads the `mcp_flights` graph it built.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import clients, console

DB, GRAPH = "mcp_flights", "flights"


async def read(cy, query, params=None):
    return clients.data(await cy.call_tool(
        "read_agensgraph_cypher", {"query": query, **({"params": params} if params else {})}
    ))


async def main() -> None:
    async with clients.cypher_client(DB, GRAPH) as cy:
        # ---- schema introspection (sampled) ----
        console.section("get_agensgraph_schema — what's in the graph")
        schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
        for label, info in schema.items():
            console.kv(label, f"{info.get('count')} nodes; props {list(info.get('properties', {}))}")
            for rel, meta in (info.get("relationships") or {}).items():
                console.kv(f"  -[:{rel}]->", meta.get("labels"))

        # ---- aggregations + multi-hop reads ----
        console.section("Reads — aggregate + multi-hop Cypher")
        console.sub("busiest airports (out-degree)")
        rows = await read(cy, 'MATCH (a:"Airport")-[:"ROUTE"]->() '
                              'RETURN a.iata AS iata, a.city AS city, count(*) AS routes '
                              'ORDER BY routes DESC LIMIT 8')
        console.table([(r["iata"], r["city"], r["routes"]) for r in rows["rows"]],
                      headers=["iata", "city", "out_routes"])

        console.sub("countries with the most airports")
        rows = await read(cy, 'MATCH (a:"Airport") WHERE a.country IS NOT NULL '
                              'RETURN a.country AS country, count(*) AS airports '
                              'ORDER BY airports DESC LIMIT 5')
        console.table([(r["country"], r["airports"]) for r in rows["rows"]],
                      headers=["country", "airports"])

        console.sub("2-hop: airports that reach JFK with one stopover")
        rows = await read(cy, 'MATCH (a:"Airport")-[:"ROUTE"]->(hub:"Airport")-[:"ROUTE"]->(:"Airport" {iata: \'JFK\'}) '
                              'WHERE a.iata <> \'JFK\' '
                              'RETURN DISTINCT a.iata AS origin, hub.iata AS via LIMIT 8')
        console.table([(r["origin"], r["via"]) for r in rows["rows"]], headers=["origin", "via_hub"])

        # ---- vertex / edge parsing ----
        console.section("Vertex & edge parsing (agtype-style strings → JSON)")
        v = await read(cy, 'MATCH (a:"Airport" {iata: \'JFK\'}) RETURN a')
        console.kv("vertex a", v["rows"][0]["a"])
        e = await read(cy, 'MATCH (a:"Airport" {iata: \'JFK\'})-[r:"ROUTE"]->(b:"Airport") RETURN a, r, b LIMIT 1')
        console.kv("edge triple (a, r, b)", e["rows"][0]["r"])

        # ---- pagination: fetch bounded pages, following next_offset ----
        console.section("Pagination — bounded pages over the ROUTE set")
        query = 'MATCH (a:"Airport")-[r:"ROUTE"]->(b:"Airport") RETURN a.iata AS src, b.iata AS dst, r.airline AS airline'
        offset = 0
        for n in range(3):
            page = clients.data(await cy.call_tool(
                "read_agensgraph_cypher", {"query": query, "limit": 1000, "offset": offset}))
            sample = page["rows"][0]
            console.kv(f"page {n} (offset {offset})",
                       f"{page['row_count']} rows, e.g. {sample['src']}-{sample['airline']}->{sample['dst']}, "
                       f"has_more={page['has_more']}, next_offset={page['next_offset']}")
            offset = page["next_offset"]
        total = clients.data(await cy.call_tool(
            "read_agensgraph_cypher", {"query": 'MATCH ()-[r:"ROUTE"]->() RETURN count(*) AS n'}))
        console.kv("total routes (count)", f"{total['rows'][0]['n']:,}")
        console.kv("note", "limit/offset is ideal for bounded browsing; deep OFFSET re-scans (see findings)")

        # ---- read-only enforcement ----
        console.section("Read-only enforcement")
        try:
            await cy.call_tool("read_agensgraph_cypher",
                               {"query": 'CREATE (:"Airport" {iata: \'XXX\'})'})
            console.kv("write via read tool", "ALLOWED (unexpected!)")
        except Exception as e:
            console.kv("write via read tool", f"rejected ({type(e).__name__})")

    # ---- knobs: read timeout + token limit (fresh clients) ----
    console.section("Knobs — read_timeout and token_limit")
    async with clients.cypher_client(DB, GRAPH, read_timeout=1) as cy:
        try:
            # a deliberately heavy cartesian product to trip the 1s timeout
            await cy.call_tool("read_agensgraph_cypher",
                               {"query": 'MATCH (a:"Airport"),(b:"Airport"),(c:"Airport") RETURN count(*) AS n'})
            console.kv("heavy query @1s timeout", "completed (small graph)")
        except Exception as e:
            console.kv("heavy query @1s timeout", f"timed out ({type(e).__name__})")
    async with clients.cypher_client(DB, GRAPH, token_limit=40) as cy:
        r = await cy.call_tool("read_agensgraph_cypher",
                               {"query": 'MATCH (a:"Airport") RETURN a.iata AS iata, a.name AS name', "limit": 200})
        console.kv("response truncated to ~40 tokens", f"{len(clients.text(r))} chars returned")


if __name__ == "__main__":
    asyncio.run(main())
