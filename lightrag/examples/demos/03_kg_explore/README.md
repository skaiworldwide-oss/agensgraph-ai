# 03 · Explore — the extracted KG is a first-class, queryable graph

LightRAG's knowledge graph isn't hidden behind the retriever — it's a real
property graph in AgensGraph you can inspect and traverse. This reuses demo 1's
`lightrag_wiki` graph to show the graph-store API, then answers a **multi-hop**
question that needs the graph to connect entities across documents — something
naive (chunks-only) RAG can't do.

📓 **Guided tour:** [`kg_explore.ipynb`](./kg_explore.ipynb).

## Run

```bash
# from lightrag/  (run 01_kg_modes/build.py first — this reads its graph)
.venv/bin/python examples/demos/03_kg_explore/explore.py
.venv/bin/python examples/demos/03_kg_explore/explore.py "Entity A" "Entity B"
```

## The graph-store API

```python
g = rag.chunk_entity_relation_graph
await g.get_all_labels()                       # every entity
await g.get_popular_labels(limit=12)           # most-connected entities (by degree)
await g.search_labels("Einstein", limit=10)    # substring search over entity names
await g.node_degree("Albert Einstein")         # how connected an entity is
kg = await g.get_knowledge_graph(hub, max_depth=2, max_nodes=40)   # export an ego-network
kg.nodes, kg.edges, kg.is_truncated
```

The demo prints the top entities with their degrees, searches labels, exports the
ego-network around the most-connected entity, and then asks *"How are A and B
connected?"* in `naive` vs `mix` mode — the graph mode can stitch a path across
documents.

## What you get

The auto-extracted knowledge graph as a queryable artifact — topology, search,
subgraph export, and multi-hop reasoning — backed by AgensGraph's indexed Cypher.
