# 05 · Cypher at scale — read-only, on a big pre-existing graph

The cypher server isn't just for graphs you load through it — point it (read-only) at
**any** AgensGraph graph. This demo reads the `arxiv` graph already loaded in the
`agensgraph_demos` database (~150k nodes: Papers, Authors, Categories, with embedding
vectors), built by a different integration's demo, and shows the server holds up at
scale. **It never writes** to the graph.

📓 **Guided tour:** [`cypher_scale.ipynb`](./cypher_scale.ipynb).

## Run

```bash
# from mcp-agensgraph/examples/demos
.venv/bin/python 05_cypher_scale/ask.py
```

If the `arxiv` graph isn't present, the demo prints how to get one and exits cleanly —
the flights demos (01–04) don't depend on it.

## What it shows

- **Schema introspection, bounded by sampling** — `get_agensgraph_schema` samples nodes
  (default 1000) so it stays bounded even on a 150k-node graph.
- **Aggregations + traversals at scale** — papers by year, most prolific authors,
  biggest categories (`MATCH (:Paper)-[:IN_CATEGORY]->(:Category)`).
- **Oversized-property sanitization** — a `Paper` vertex comes back as
  `{id, year, title, abstract}` — the 1536-dim `embedding` is dropped by `value_sanitize`,
  so vectors never flood the agent's context.
- **Pagination** — `limit`/`offset` over the Paper set. Watch the per-page time grow with
  offset: deep `OFFSET` re-scans the inner query (O(N) per page), so it's ideal for
  bounded browsing but not for walking an entire huge set — use keyset pagination
  (`WHERE id > last_seen`) for that (see the findings log).

## Notes

- The server runs `read_only=True` here, so only `get_agensgraph_schema` and
  `read_agensgraph_cypher` are exposed and reads run in a `READ ONLY` transaction —
  the existing graph is never modified (only two helper SQL functions are ensured in the
  database for schema introspection).
- Schema introspection on this graph improved from ~16 s to ~7 s after a `count(*)` fix
  (it was de-TOASTing every node's embedding via `count(n)`); see the findings log.
