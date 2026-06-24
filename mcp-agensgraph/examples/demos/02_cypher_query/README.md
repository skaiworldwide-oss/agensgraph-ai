# 02 · Cypher query — read the graph through the MCP server

Everything the **cypher** server's read surface does, against the OpenFlights graph
that [01_model_and_load](../01_model_and_load) built (~6k airports / ~67k routes):

- **`get_agensgraph_schema`** — node labels, properties (with types/indexing), and
  relationships, inferred from a sampled scan.
- **`read_agensgraph_cypher`** — aggregations, multi-hop traversals (e.g. 2-hop paths
  that reach JFK), and graph values: vertices parse to JSON dicts, edges to
  `(start, type, end)` triples.
- **Pagination** — walk all ~67k ROUTE edges in pages via `limit`/`offset`, following
  `has_more` / `next_offset` (the DB applies `LIMIT/OFFSET`, so it never materializes
  the whole result).
- **Read-only enforcement** — a write sent to the read tool is rejected (and would be
  blocked at the DB by the read-only transaction even if it slipped past).
- **Knobs** — `read_timeout` (a heavy query is cancelled) and `token_limit` (the
  response is truncated to a token budget).

📓 **Guided tour:** [`cypher_query.ipynb`](./cypher_query.ipynb).

## Run

```bash
# from mcp-agensgraph/examples/demos  (run 01_model_and_load/build.py first)
.venv/bin/python 02_cypher_query/ask.py
```

## What it shows

```python
async with clients.cypher_client("mcp_flights", "flights", read_timeout=1, token_limit=40) as cy:
    await cy.call_tool("get_agensgraph_schema", {})
    await cy.call_tool("read_agensgraph_cypher", {"query": "...", "limit": 5000, "offset": 10000})
```

The `limit`/`offset` page through the full edge set; `has_more`/`next_offset` in each
response tell the client when to stop. The cypher server is read-only here by
construction (the read tool runs in a `READ ONLY` transaction).
