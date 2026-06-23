# 05 · Explore — the cognee memory is a real, queryable AgensGraph graph

cognee's memory isn't a black box. Because it lives in AgensGraph, you can measure
it, break it down by type, slice it, traverse it, query it with raw **Cypher**, and
visualize it — all through the graph adapter cognee already gave you. This reuses
demo 1's `cognee_wiki` graph (so run that demo's `build.py` first).

📓 **Just want to read it?** Open [`explore.ipynb`](./explore.ipynb) — already executed,
with real outputs.

## Run it

```bash
# from the cognee/ directory of this repo (run 01_search_modes/build.py first)
.venv/bin/python examples/demos/05_explore/explore.py
```

No LLM calls here — it's pure graph inspection, so it's fast and free.

## What it shows

```python
g = await get_graph_engine()                         # the AgensGraph adapter

await g.get_graph_metrics()                           # num_nodes/edges, mean degree, density
await g.get_graph_data()                              # full (nodes, edges) — break down by type
await g.get_disconnected_nodes()                      # isolated nodes
await g.get_degree_one_nodes("Entity")                # leaf entities
await g.get_filtered_graph_data([{"type": ["Entity"]}])   # just the Entity subgraph

await g.query('MATCH (n:"__Node__") ... RETURN n.name, count(r) ...')   # raw Cypher

await g.get_neighbors(node_id)                        # adjacent nodes
await g.get_connections(node_id)                      # (node, edge, node) connections

await cognee.visualize_graph("wiki_graph.html")       # interactive HTML
```

The script prints the graph metrics, the node-type breakdown (Entity / EntityType /
DocumentChunk / TextSummary / TextDocument), connectivity counts, an Entity-only
subgraph, the top entities by degree via raw AgensGraph Cypher, a traversal from the
busiest node, and writes an interactive HTML visualization.

## What you get

Full visibility into the knowledge graph cognee built — metrics, structure, slices,
traversal, ad-hoc Cypher, and a visual — directly against AgensGraph, the same
database that served every search in the other demos.
