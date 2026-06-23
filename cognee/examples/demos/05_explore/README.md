# 05 · Explore — the cognee memory is a real, queryable AgensGraph graph

cognee's memory isn't a black box. Because it's stored in AgensGraph, you can
measure it, break it down, query it with raw **Cypher**, and visualize it. This
reuses demo 1's `cognee_wiki` graph.

📓 **Guided tour:** [`explore.ipynb`](./explore.ipynb).

## Run

```bash
# from cognee/  (run 01_search_modes/build.py first — this reads its graph)
.venv/bin/python examples/demos/05_explore/explore.py
```

## What it shows

```python
g = await get_graph_engine()                 # the AgensGraph adapter
await g.get_graph_metrics()                  # num_nodes/edges, mean degree, density
await g.get_graph_data()                     # full (nodes, edges) — break down by type
await g.get_disconnected_nodes()             # connectivity check
await g.query('MATCH (n:"__Node__") ... RETURN n.name, count(r) ...')   # raw Cypher
await cognee.visualize_graph("wiki_graph.html")                          # interactive HTML
```

The script prints graph metrics, the node-type breakdown (Entity / EntityType /
DocumentChunk / TextSummary / …), the disconnected-node count, the top entities
by degree via raw AgensGraph Cypher, and writes an HTML visualization.

## What you get

Full visibility into the knowledge graph cognee built — metrics, structure,
ad-hoc Cypher, and a visual — directly against AgensGraph.
