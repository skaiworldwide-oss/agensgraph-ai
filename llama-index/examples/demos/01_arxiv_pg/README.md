# 01 · arXiv — graph + vector in one store

Build a property graph of papers, authors and categories, then run **four kinds
of query over the same store**: Cypher analytics, vector search, graph
expansion, and GraphRAG. The graph is built directly (no LLM extraction), so it
scales to the full dataset.

```
(Paper {title, abstract, year}) -[AUTHORED_BY]-> (Author)
(Paper)                         -[IN_CATEGORY]-> (Category)
```

📓 **Guided tour:** [`arxiv_pg.ipynb`](./arxiv_pg.ipynb) is a pre-executed notebook
that walks through every capability below with real outputs — open it after `prepare.py`.

## Run

```bash
# from llama-index/
.venv/bin/python examples/demos/01_arxiv_pg/prepare.py        # ingest + embed (ARXIV_LIMIT, default 50000)
.venv/bin/python examples/demos/01_arxiv_pg/query.py
.venv/bin/python examples/demos/01_arxiv_pg/query.py "your question"

# quick dry-run (a couple of cents):
ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_pg/prepare.py
```

Knobs: `ARXIV_LIMIT` (papers), `ARXIV_BATCH`, `EMBED_CONCURRENCY`, `EMBED_BATCH`,
`ARXIV_RESET=1` (drop & rebuild).

## The patterns (`prepare.py`)

Ingest entities and relationships in batches, then add embeddings:

```python
from llama_index.core.graph_stores.types import EntityNode, Relation

store.upsert_nodes([
    EntityNode(name=paper_id, label="Paper", properties={"title": t, "abstract": a, "year": y}),
    EntityNode(name=author,   label="Author"),
])
store.upsert_relations([Relation(label="AUTHORED_BY", source_id=paper_id, target_id=author)])

# add the embedding for vector search (re-upsert just the embedding):
await store.aupsert_nodes([EntityNode(name=paper_id, label="Paper", embedding=vec)])
```

## The patterns (`query.py`)

```python
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import VectorContextRetriever

# (a) analytics — plain Cypher
store.structured_query(
    'MATCH (p:"__Node__")-[:"AUTHORED_BY"]->(a:"__Node__") '
    'RETURN a.name AS author, count(*) AS papers ORDER BY papers DESC LIMIT 10')

# (b) semantic search — HNSW
nodes, scores = store.vector_query(VectorStoreQuery(query_embedding=vec, similarity_top_k=5))

# (c) expand the hits through the graph
triplets = store.get_rel_map(nodes, depth=1)

# (d) GraphRAG — attach a PropertyGraphIndex to the populated store
index = PropertyGraphIndex.from_existing(store, embed_model=embed, llm=llm, kg_extractors=[])
answer = index.as_query_engine(
    sub_retrievers=[VectorContextRetriever(graph_store=store, embed_model=embed)]
).query("recent graph neural network methods?")
```

## What you get

One AgensGraph graph that answers analytical Cypher, vector search, graph
expansion and GraphRAG over the same entities — no separate graph and vector DBs
to keep in sync.

## Tips

- Pass `vector_dimension=` when constructing the store, or the HNSW index isn't
  built and vector search falls back to a sequential scan.
- For analytics, prefer `count(*)` over `count(p)` and walk the **edges** rather
  than filtering on node type (`'Author' IN n.labels`) — both are much faster on
  a graph whose nodes carry embeddings.
