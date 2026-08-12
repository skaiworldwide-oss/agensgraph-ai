# 01 · Model & load — the cross-server pipeline

The flagship demo, and the one that **builds the graph the other demos query**. It uses
two of the MCP servers together, with **no hand-written Cypher**:

1. The **data-modeling** server designs a graph schema (Airports + ROUTEs): validate the
   model, render it as **Mermaid**, round-trip it through the **Arrows** visual-editor
   format, and **generate the constraint + ingest Cypher**.
2. The **cypher** server's `write` tool runs that generated Cypher to load real
   [OpenFlights](https://openflights.org/data.html) data (~6k airports / ~67k routes,
   CC-BY-SA) into the `mcp_flights` database.

📓 **Guided tour:** [`model_and_load.ipynb`](./model_and_load.ipynb) — real outputs, no setup.

## Run

```bash
# from mcp-agensgraph/examples/demos  (no API key needed — the servers are pure tools)
FLIGHTS_ROUTES_LIMIT=3000 .venv/bin/python 01_model_and_load/build.py   # quick subset
.venv/bin/python 01_model_and_load/build.py                            # full ~67k routes
FLIGHTS_RESET=1 .venv/bin/python 01_model_and_load/build.py            # clean rebuild
```

Knobs: `FLIGHTS_DB` (default `mcp_flights`), `FLIGHTS_GRAPH` (default `flights`),
`FLIGHTS_AIRPORTS_LIMIT`, `FLIGHTS_ROUTES_LIMIT` (default: all), `FLIGHTS_BATCH`
(default 2000), `FLIGHTS_RESET=1` (drop + rebuild). The OpenFlights data downloads once
into `.data/` (a tiny vendored sample is used if you're offline).

## What it shows

```python
# data-modeling server — design + generate (no database)
await dm.call_tool("validate_data_model", {"data_model": DATA_MODEL})
await dm.call_tool("get_mermaid_config_str", {"data_model": DATA_MODEL})
await dm.call_tool("get_constraints_cypher_queries", {"data_model": DATA_MODEL})
node_q = await dm.call_tool("get_node_cypher_ingest_query", {"node": AIRPORT})

# cypher server — run the generated ingest Cypher with a JSONB records batch
await cy.call_tool("write_agensgraph_cypher", {"query": node_q, "params": {"records": batch}})
```

The generated ingest query is `UNWIND %(records)s ... MERGE ...`, and the records batch is a
JSON array — exactly what an MCP agent would pass. The placeholder is the database driver's
`%(name)s`, not `$name`: `$records` is not rewritten by anything and reaches the server as a
syntax error. A list bound this way arrives as JSONB, which is what `UNWIND` expects.

## Notes

- **Each generated statement is run whole.** `get_constraints_cypher_queries` answers with a
  list, one statement per item, and the demo runs each item as it stands. It does not join
  them and split the result: a label may hold whatever a label holds, including the separator
  it would be split on, and splitting there hands the tail of a name to the server as code.
- **No uniqueness is asserted on the relationship.** A route's `airline` is unique *within* an
  endpoint pair, not globally — an airline flies many routes — so the generator emits the
  label declaration alone for it, and the ingest merges on the endpoints *and* the key. The
  **Airport** key does get a unique property index.
- **Throughput** (local AgensGraph, 6,072 airports + 66,934 routes through the MCP write tool,
  batched `UNWIND`): 0.48 s for the airports and 3.95 s for the routes, 4.43 s in all, about
  16,500 rows/s. Wall-clock is the database, not the protocol.
- Re-running is safe: every generated statement carries `IF NOT EXISTS`, and
  `FLIGHTS_RESET=1` drops the graph first for a clean rebuild.
