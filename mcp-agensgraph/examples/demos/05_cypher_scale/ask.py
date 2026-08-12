"""05 · cypher at scale — read-only, on a big pre-existing graph.

Points the cypher MCP server (read-only) at the `arxiv` graph already loaded in the
`agensgraph_demos` database (50,000 Papers, 88,455 Authors, 147 Categories, 17 Years)
— built by a different integration's demo. Shows that schema introspection, large
aggregations, plan inspection, index advice, health, the top statements, oversized-property
sanitization and pagination all hold up on a graph far larger than the flights demo.

**It creates nothing and writes nothing**, and says so with a count rather than a promise:
the database it reads is somebody else's, so it does not make the database, does not make
the graph, and checks that `pg_class` and `pg_proc` are the size they were when it is done.

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


async def timed(cy, query):
    """One read, and how long the whole tool call took."""
    start = time.perf_counter()
    rows = await read(cy, query)
    return rows["rows"], (time.perf_counter() - start) * 1000


def catalog_size() -> tuple[int, int]:
    """How many relations and functions this database holds.

    The demo's claim is that it leaves the database alone, and this is what makes the claim
    checkable: a graph, a label, a table or an installed helper function all show up here.
    """
    import psycopg

    with psycopg.connect(config.dsn(DB), autocommit=True) as conn:
        relations = conn.execute("SELECT count(*) FROM pg_class").fetchone()[0]
        functions = conn.execute("SELECT count(*) FROM pg_proc").fetchone()[0]
    return relations, functions


async def main() -> None:
    if not config.graph_exists(DB, GRAPH):
        console.section("Skipped — no large graph available")
        print(f"  The '{GRAPH}' graph in '{DB}' isn't present on this instance.")
        print("  Point this demo at any large AgensGraph graph by editing DB/GRAPH,")
        print("  or load one (e.g. the langchain arxiv demo). The flights demos still run.")
        return

    before = catalog_size()

    # read_only=True → the write tool is not exposed, reads run in a READ ONLY transaction,
    # and the client makes neither the database nor the graph.
    async with clients.cypher_client(DB, GRAPH, read_only=True) as cy:
        console.section("Read-only server on a graph it did not build")
        console.kv("tools", sorted(t.name for t in await cy.list_tools()))
        console.kv("write tool present",
                   "write_agensgraph_cypher" in {t.name for t in await cy.list_tools()})

        console.section("get_agensgraph_schema — counts exact, property types sampled")
        start = time.perf_counter()
        schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
        console.kv("schema introspection", f"{(time.perf_counter() - start) * 1000:.0f} ms")
        for label, info in schema.items():
            console.kv(label, f"{info.get('count'):,} nodes; props {list(info.get('properties', {}))}")
            for rel, meta in (info.get("relationships") or {}).items():
                console.kv(f"  -[:{rel}]->", meta.get("labels"))

        # ---- the same answer, two ways, both timed ----
        console.section("Papers by year — reading the property, and reading the edge")
        console.sub("off every Paper's own year property")
        by_property, property_ms = await timed(
            cy,
            'MATCH (p:"Paper") WHERE p.year IS NOT NULL '
            'RETURN p.year AS year, count(*) AS papers ORDER BY year DESC LIMIT 5',
        )
        console.table([(r["year"], r["papers"]) for r in by_property], headers=["year", "papers"])
        console.kv("elapsed", f"{property_ms:,.0f} ms")

        console.sub("off the Year node this graph already models")
        by_edge, edge_ms = await timed(
            cy,
            'MATCH (:"Paper")-[:"UPDATED_IN"]->(y:"Year") '
            'RETURN y.year AS year, count(*) AS papers ORDER BY year DESC LIMIT 5',
        )
        console.table([(r["year"], r["papers"]) for r in by_edge], headers=["year", "papers"])
        console.kv("elapsed", f"{edge_ms:,.1f} ms")

        console.kv("same answer", by_property == by_edge)
        console.kv("faster by", f"{property_ms / edge_ms:,.0f}x")
        console.kv("why", "reading p.year visits every Paper's property bag, and each bag "
                          "carries a 1,536-item embedding that has to come back off TOAST to "
                          "be opened. The edge form reads 17 Year nodes and the edges into "
                          "them, and never opens a Paper.")

        console.sub("explain_agensgraph_cypher — the plan that explains the gap")
        plan = await cy.call_tool("explain_agensgraph_cypher", {
            "query": 'MATCH (p:"Paper") WHERE p.year IS NOT NULL '
                     'RETURN p.year AS year, count(*) AS papers ORDER BY year DESC LIMIT 5'})
        node = clients.data(plan)[0]["Plan"]
        console.kv("top node", f"{node['Node Type']} → {node['Total Cost']:,.0f} est. cost")
        console.kv("scanned", _deepest_scan(node))

        console.section("Aggregations + traversals at scale")
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
        properties = p["rows"][0]["p"]["properties"]
        console.kv("Paper properties", list(properties))
        console.kv("embedding", properties["embedding"])
        console.kv("note", "a list of 128 items or more is replaced rather than removed, so a "
                           "suppressed embedding cannot be read as an absent property")

        console.section("recommend_property_indexes — advice, and what it is not")
        advice = clients.data(await cy.call_tool("recommend_property_indexes", {
            "query": 'MATCH (p:"Paper") WHERE p.year = 2019 RETURN p.title AS title'}))
        for finding in advice["findings"]:
            console.kv(finding["kind"], finding["detail"])
            console.kv("  suggestion", finding["suggestion"])
        console.kv("verified", advice["verified"])
        console.kv("existing property indexes", len(advice["existing_indexes"]))

        console.section("agensgraph_health — each check stands on its own")
        health = clients.data(await cy.call_tool("agensgraph_health", {}))
        console.kv("checks", sorted(health))
        console.kv("optional extensions", health["extensions"])
        console.kv("auto_gather_graphmeta", health["graphmeta"][0]["auto_gather_graphmeta"])
        console.kv("note", "a check whose extension is absent reports that rather than "
                           "failing the others: " + str(health["pg_buffercache"]["available"]))

        console.section("top_cypher_queries — what this server has spent its time on")
        top = clients.data(await cy.call_tool("top_cypher_queries", {"limit": 3}))
        if isinstance(top, dict):
            console.kv("unavailable", top)
        else:
            console.table([(q["query"].splitlines()[0][:56], q["calls"], q["mean_ms"]) for q in top],
                          headers=["statement (first line)", "calls", "mean ms"])
            console.kv("note", "literals appear as parameters -- Cypher is normalised the way "
                               "SQL is, so one shape is one row however many literals it ran with")

        console.section("The read-only boundary, held by the database rather than by a reading")
        # A write sent to the read tool never reaches the server: the statement is read first
        # and refused here. That is a good error message, not a boundary.
        with console.expecting_refusal():
            try:
                await cy.call_tool("read_agensgraph_cypher", {"query": 'CREATE (:"Paper" {id: -1})'})
                console.kv("write via the read tool", "ALLOWED (unexpected!)")
            except Exception as e:
                console.kv("write via the read tool", f"refused by the server process: {e}")

            # `explain ... analyze` executes what it is given, and reads nothing about it.
            # What refuses this is the read-only transaction it runs in, with SQLSTATE 25006.
            try:
                await cy.call_tool("explain_agensgraph_cypher",
                                   {"query": 'CREATE (:"Paper" {id: -1})', "analyze": True})
                console.kv("the same write, run by explain", "ALLOWED (unexpected!)")
            except Exception as e:
                console.kv("the same write, run by explain", f"refused by the database: {e}")
        left = await read(cy, 'MATCH (p:"Paper") WHERE p.id = -1 RETURN count(*) AS n')
        console.kv("Papers the refused writes left behind", left["rows"][0]["n"])

        console.section("Pagination — bounded pages over 50k Papers")
        query = 'MATCH (p:"Paper") RETURN p.id AS id, p.title AS title'
        offset = 0
        for n in range(3):
            start = time.perf_counter()
            page = clients.data(await cy.call_tool(
                "read_agensgraph_cypher", {"query": query, "limit": 1000, "offset": offset}))
            console.kv(f"page {n} (offset {offset})",
                       f"{page['row_count']} rows in {time.perf_counter() - start:.2f}s, "
                       f"has_more={page['has_more']}, next_offset={page['next_offset']}")
            offset = page["next_offset"]
        console.kv("note", "deep OFFSET re-scans the inner query each page (O(N)/page); for a full "
                           "walk of a huge set ask for `walk`, or order by a key and filter on the "
                           "last one seen. See findings.")

    console.section("What this demo left in the database")
    after = catalog_size()
    console.kv("pg_class and pg_proc unchanged", before == after)
    console.kv("note", "the database and the graph were not created either -- a read-only "
                       "client makes neither, which is what lets this point at a graph it "
                       "does not own")


def _deepest_scan(node) -> str:
    """The scan at the bottom of a plan, named with what it read."""
    while node.get("Plans"):
        node = node["Plans"][0]
    where = node.get("Relation Name") or node.get("Alias") or "?"
    return f"{node['Node Type']} on {where}, {node.get('Plan Rows', 0):,} rows estimated"


if __name__ == "__main__":
    asyncio.run(main())
