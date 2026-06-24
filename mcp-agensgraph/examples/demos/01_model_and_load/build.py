"""01 · model & load — the cross-server pipeline.

Use the **data-modeling** MCP server to design a graph schema (Airports + ROUTEs),
validate it, visualize it (Mermaid), round-trip it through the Arrows format, and
**generate the constraint + ingest Cypher** — then run that generated Cypher through
the **cypher** MCP server's write tool to load real OpenFlights data into the
`mcp_flights` database. Nothing here is hand-written Cypher: the data-modeling server
produces it and the cypher server executes it.

    cd mcp-agensgraph/examples/demos
    .venv/bin/python 01_model_and_load/build.py
    FLIGHTS_ROUTES_LIMIT=3000 .venv/bin/python 01_model_and_load/build.py   # quick subset

Knobs: FLIGHTS_DB (default mcp_flights), FLIGHTS_GRAPH (default flights),
FLIGHTS_AIRPORTS_LIMIT, FLIGHTS_ROUTES_LIMIT (default: all), FLIGHTS_RESET=1 (rebuild),
FLIGHTS_BATCH (default 2000). No API key needed — the MCP servers are pure tools.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import clients, config, console
from _common.datautil import airports as load_airports
from _common.datautil import env_int
from _common.datautil import routes as load_routes

DB = __import__("os").getenv("FLIGHTS_DB", "mcp_flights")
GRAPH = __import__("os").getenv("FLIGHTS_GRAPH", "flights")
BATCH = env_int("FLIGHTS_BATCH", 2000)


def _prop(name, type_="STRING", desc=None):
    return {"name": name, "type": type_, "source": None, "description": desc}


# The schema we hand to the data-modeling server (plain dicts = its Node/Relationship/DataModel).
AIRPORT = {
    "label": "Airport",
    "key_property": _prop("iata", desc="3-letter IATA code"),
    "properties": [
        _prop("name"), _prop("city"), _prop("country"),
        _prop("lat", "FLOAT"), _prop("lon", "FLOAT"),
    ],
}
ROUTE = {
    "type": "ROUTE",
    "start_node_label": "Airport",
    "end_node_label": "Airport",
    "key_property": _prop("airline", desc="operating airline (IATA)"),
    "properties": [_prop("airline"), _prop("equipment"), _prop("stops", "INTEGER")],
    "metadata": {},
}
DATA_MODEL = {"nodes": [AIRPORT], "relationships": [ROUTE]}


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def main() -> None:
    config.ensure_db(DB)

    # ---------------------------------------------------------------- design (data-modeling server)
    console.section("Design the schema with the data-modeling MCP server")
    async with clients.data_modeling_client() as dm:
        ok = clients.data(await dm.call_tool("validate_data_model", {"data_model": DATA_MODEL}))
        console.kv("validate_data_model", ok)

        examples = clients.data(await dm.call_tool("list_example_data_models", {}))
        console.kv("example models available", len(examples) if isinstance(examples, (list, dict)) else examples)

        # Arrows round-trip (design tools interop with the arrows.app visual editor)
        arrows = clients.data(await dm.call_tool("export_to_arrows_json", {"data_model": DATA_MODEL}))
        reloaded = clients.data(await dm.call_tool("load_from_arrows_json", {"arrows_data_model_dict": arrows}))
        console.kv("arrows round-trip nodes", len(reloaded.get("nodes", [])))

        mermaid = clients.text(await dm.call_tool("get_mermaid_config_str", {"data_model": DATA_MODEL}))
        console.sub("Mermaid diagram (paste into any Mermaid renderer)")
        print(mermaid)

        constraints = clients.data(await dm.call_tool("get_constraints_cypher_queries", {"data_model": DATA_MODEL}))
        node_ingest = clients.text(await dm.call_tool("get_node_cypher_ingest_query", {"node": AIRPORT}))
        rel_ingest = clients.text(await dm.call_tool(
            "get_relationship_cypher_ingest_query",
            {"data_model": DATA_MODEL, "relationship_type": "ROUTE",
             "relationship_start_node_label": "Airport", "relationship_end_node_label": "Airport"},
        ))

    console.section("Generated Cypher (by the data-modeling server)")
    console.sub("constraints"); [print("  " + c) for c in constraints]
    console.sub("node ingest"); print(node_ingest)
    console.sub("relationship ingest"); print(rel_ingest)

    # ---------------------------------------------------------------- load (cypher server)
    console.section(f"Load OpenFlights into {DB}/{GRAPH} via the cypher MCP write tool")
    airports = load_airports(limit=env_int("FLIGHTS_AIRPORTS_LIMIT", 0) or None)
    valid = {a["iata"] for a in airports}
    routes = load_routes(limit=env_int("FLIGHTS_ROUTES_LIMIT", 0) or None, valid_iata=valid)
    console.kv("airports", f"{len(airports):,}")
    console.kv("routes", f"{len(routes):,}")

    if env_int("FLIGHTS_RESET", 0):
        import psycopg

        console.sub("FLIGHTS_RESET=1 — dropping the graph for a clean rebuild")
        with psycopg.connect(config.dsn(DB), autocommit=True) as conn:
            conn.execute(f'DROP GRAPH IF EXISTS "{GRAPH}" CASCADE')

    async with clients.cypher_client(DB, GRAPH) as cy:
        # Apply constraints — but SKIP the relationship UNIQUE constraint: a route key
        # (airline) is unique *within* an endpoint pair, not globally, so asserting
        # global uniqueness would reject the 2nd route any airline flies. The generated
        # `CREATE CONSTRAINT` also lacks IF NOT EXISTS, so re-runs are tolerated here.
        console.sub("applying constraints (skipping the global rel-key UNIQUE — see README)")
        for stmt in (s.strip() for c in constraints for s in c.split(";")):
            if not stmt:
                continue
            if stmt.upper().startswith("CREATE CONSTRAINT") and "ROUTE" in stmt:
                console.kv("skipped (global rel-key UNIQUE)", stmt[:50] + " …")
                continue
            try:
                await cy.call_tool("write_agensgraph_cypher", {"query": stmt})
            except Exception:
                console.kv("already applied", stmt[:50] + " …")

        with console.timer("load airports") as t:
            for chunk in _batches(airports, BATCH):
                await cy.call_tool("write_agensgraph_cypher", {"query": node_ingest, "params": {"records": chunk}})
        print("  " + t.rate(len(airports), "airports"))

        # routes: the rel-ingest query matches endpoints by record.sourceId / .targetId
        route_records = [
            {"sourceId": r["src"], "targetId": r["dst"], "airline": r["airline"],
             "equipment": r["equipment"], "stops": r["stops"]}
            for r in routes
        ]
        with console.timer("load routes") as t:
            for chunk in _batches(route_records, BATCH):
                await cy.call_tool("write_agensgraph_cypher", {"query": rel_ingest, "params": {"records": chunk}})
        print("  " + t.rate(len(route_records), "routes"))

        console.section("Result — the graph the other demos query (get_agensgraph_schema)")
        schema = clients.data(await cy.call_tool("get_agensgraph_schema", {}))
        for label, info in schema.items():
            console.kv(label, f"{info.get('count')} nodes; props {list(info.get('properties', {}))}")
            for rel, meta in (info.get("relationships") or {}).items():
                console.kv(f"  -[:{rel}]->", meta.get("labels"))
        edges = clients.data(await cy.call_tool(
            "read_agensgraph_cypher", {"query": 'MATCH ()-[r:"ROUTE"]->() RETURN count(r) AS routes'}))
        console.kv("ROUTE edges", edges["rows"][0]["routes"])
        print("\n  Loaded. Explore it with: .venv/bin/python 02_cypher_query/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
