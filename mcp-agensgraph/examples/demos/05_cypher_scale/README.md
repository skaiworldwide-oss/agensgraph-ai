# 05 · Cypher at scale — read-only, on a graph this demo did not build

The cypher server isn't just for graphs you load through it — point it (read-only) at
**any** AgensGraph graph. This demo reads the `arxiv` graph already loaded in the
`agensgraph_demos` database (50,000 Papers, 88,455 Authors, 147 Categories, 17 Years, with
1,536-dimension embedding vectors), built by a different integration's demo, and shows the
server holds up at scale.

📓 **Guided tour:** [`cypher_scale.ipynb`](./cypher_scale.ipynb).

## Run

```bash
# from mcp-agensgraph/examples/demos
.venv/bin/python 05_cypher_scale/ask.py
```

If the `arxiv` graph isn't present, the demo prints how to get one and exits cleanly —
the flights demos (01–04) don't depend on it.

## What it shows

- **It creates nothing.** A read-only client makes neither the database nor the graph, which
  is what lets this point at a database somebody else owns. The demo counts `pg_class` and
  `pg_proc` before and after and reports that both are the size they were — nothing is
  installed for schema introspection either.
- **Six tools, not two.** `--read-only` withholds the write tool; `get_agensgraph_schema`,
  `read_agensgraph_cypher`, `explain_agensgraph_cypher`, `recommend_property_indexes`,
  `agensgraph_health` and `top_cypher_queries` all remain.
- **The same answer, two ways, both timed.** "Papers by year" read off `p.year` takes about
  **12.6 s**; read off the `Year` node this graph already models it takes about **1.8 ms** for
  an identical answer. Reading the property visits every Paper's property bag, and each bag
  carries an embedding that has to come back off TOAST before the bag can be opened. The
  edge-driven form reads 17 `Year` nodes and the edges into them and never opens a Paper.
  `explain_agensgraph_cypher` prints the `Seq Scan on Paper` that explains it.
- **Index advice** — `recommend_property_indexes` on a `WHERE p.year = 2019` filter names the
  index to build, and says out loud that the advice is reasoned from the plan rather than
  costed against a built index, because an AgensGraph property index cannot be simulated.
- **Health, and the top statements** — `agensgraph_health` reports each check separately, so
  one whose extension is absent says so rather than failing the others;
  `top_cypher_queries` reads `pg_stat_statements`, where Cypher is normalised the way SQL is.
- **Oversized-property sanitization** — a `Paper` comes back with all five of its properties
  and its `embedding` replaced by `<omitted: list of 1536 items>`, so a suppressed vector
  cannot be read as an absent property.
- **The read-only boundary, held by the database.** A write sent to `read_agensgraph_cypher`
  never reaches the server: the statement is read first and refused locally, which is a good
  error message rather than a boundary. The same write given to `explain_agensgraph_cypher`
  with `analyze` *is* executed — and refused by the read-only transaction with `25006`,
  leaving nothing behind, which the demo checks by counting.
- **Pagination** — `limit`/`offset` over the Paper set. Watch the per-page time grow with
  offset: deep `OFFSET` re-scans the inner query (O(N) per page), so it's ideal for
  bounded browsing but not for walking an entire huge set — ask for `walk`, or order by a key
  and filter on the last one seen.

## Notes

- This demo shares the `agensgraph_demos` database with other suites, which is the point: it
  is the one demo here that reads a graph it did not create. The flights demos (01–04) use
  their own databases, `mcp_flights` and `mcp_memory`.
- `get_agensgraph_schema` on this graph takes about **0.25 s**. It used to take ~7 s, and
  before that ~16 s: the counts stopped being taken with `count(n)`, which de-TOASTed every
  node's embedding, and the triples now come from the catalog the planner keeps rather than
  from a scan of the edges. See the findings log.
