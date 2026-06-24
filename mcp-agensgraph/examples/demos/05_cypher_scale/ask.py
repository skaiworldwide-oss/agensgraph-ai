"""05 · cypher at scale — read-only, on a big pre-existing graph.

Points the cypher MCP server (read-only) at the `arxiv` graph already loaded in the
`agensgraph_demos` database (~150k nodes: Papers, Authors, Categories) — built by a
different integration's demo. Shows that schema introspection (sampled + bounded),
large aggregations/traversals, oversized-property sanitization, and pagination all
hold up on a graph far larger than the flights demo. **Never writes** to the graph.

    cd mcp-agensgraph/examples/demos
    .venv/bin/python 05_cypher_scale/ask.py

Gated: if the arxiv graph isn't present, the demo prints how to get one and exits.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import clients, config, console

DB, GRAPH = "agensgraph_demos", "arxiv"


async def read(cy, query):
    return clients.data(await cy.call_tool("read_agensgraph_cypher", {"query": query}))


async def main() -> None:
    if not config.graph_exists(DB, GRAPH):
        console.section("Skipped — no large graph available")
        print(f"  The '{GRAPH}' graph in '{DB}' isn't present on this instance.")
        print("  Point this demo at any large AgensGraph graph by editing DB/GRAPH,")
        print("  or load one (e.g. the langchain arxiv demo). The flights demos still run.")
        return

    # read_only=True → the write tool is not exposed; reads run in a READ ONLY tx.
    async with clients.cypher_client(DB, GRAPH, read_only=True) as cy:
        console.section("Read-only server on a ~150k-node graph")
        console.kv("tools", sorted(t.name for t in await cy.list_tools()))

        console.section("get_agensgraph_schema — bounded by node sampling")
        t = time.perf_counter()
        schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
        console.kv("schema introspection", f"{time.perf_counter() - t:.1f}s (sampled)")
        for label, info in schema.items():
            console.kv(label, f"{info.get('count'):,} nodes; props {list(info.get('properties', {}))}")
            for rel, meta in (info.get("relationships") or {}).items():
                console.kv(f"  -[:{rel}]->", meta.get("labels"))

        console.section("Aggregations + traversals at scale")
        console.sub("papers by year (most recent)")
        rows = await read(cy, 'MATCH (p:"Paper") WHERE p.year IS NOT NULL '
                              'RETURN p.year AS year, count(*) AS papers ORDER BY year DESC LIMIT 5')
        console.table([(r["year"], r["papers"]) for r in rows["rows"]], headers=["year", "papers"])

        console.sub("most prolific authors")
        rows = await read(cy, 'MATCH (:"Paper")-[:"AUTHORED_BY"]->(a:"Author") '
                              'RETURN a.name AS author, count(*) AS papers ORDER BY papers DESC LIMIT 5')
        console.table([(r["author"], r["papers"]) for r in rows["rows"]], headers=["author", "papers"])

        console.sub("biggest categories")
        rows = await read(cy, 'MATCH (:"Paper")-[:"IN_CATEGORY"]->(c:"Category") '
                              'RETURN c.name AS category, count(*) AS papers ORDER BY papers DESC LIMIT 5')
        console.table([(r["category"], r["papers"]) for r in rows["rows"]], headers=["category", "papers"])

        console.section("Oversized-property sanitization")
        p = await read(cy, 'MATCH (p:"Paper") RETURN p LIMIT 1')
        console.kv("Paper keys returned", list(p["rows"][0]["p"].keys()))
        console.kv("note", "the 'embedding' vector is dropped by value_sanitize (keeps context lean)")

        console.section("Pagination — bounded pages over 55k Papers")
        query = 'MATCH (p:"Paper") RETURN p.id AS id, p.title AS title'
        offset = 0
        for n in range(3):
            t = time.perf_counter()
            page = clients.data(await cy.call_tool(
                "read_agensgraph_cypher", {"query": query, "limit": 1000, "offset": offset}))
            console.kv(f"page {n} (offset {offset})",
                       f"{page['row_count']} rows in {time.perf_counter() - t:.2f}s, "
                       f"has_more={page['has_more']}, next_offset={page['next_offset']}")
            offset = page["next_offset"]
        console.kv("note", "deep OFFSET re-scans the inner query each page (O(N)/page); for a full "
                           "walk of a huge set use keyset pagination (WHERE id > last). See findings.")


if __name__ == "__main__":
    asyncio.run(main())
