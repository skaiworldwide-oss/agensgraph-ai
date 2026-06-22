# 05 · Workspace — many tenants, one database, fully isolated

LightRAG's `workspace` lets several tenants share one database without seeing
each other's data. With `lightrag-agensgraph`: the relational stores (vector /
KV / doc-status) partition by a `workspace` column, and the graph gets a
per-workspace name — so two tenants' knowledge graphs never mix. This demo
ingests distinct, made-up corpora for two tenants into one `lightrag_tenants`
database and proves the isolation both ways.

📓 **Guided tour:** [`workspace.ipynb`](./workspace.ipynb).

## Run

```bash
# from lightrag/  (small + fast)
.venv/bin/python examples/demos/05_workspace/tenants.py
```

## The pattern

```python
from _common.rag import build_rag

acme   = build_rag("lightrag_tenants", workspace="acme")     # graph: acme_chunk_entity_relation
globex = build_rag("lightrag_tenants", workspace="globex")   # graph: globex_chunk_entity_relation
# ... initialize, ainsert each tenant's docs ...

await acme.chunk_entity_relation_graph.get_all_labels()        # only Acme's entities
await acme.chunk_entity_relation_graph.has_node("Hank Scorpio") # False — Globex's entity
await acme.aquery("Who is Hank Scorpio?")                       # no answer for Acme
```

The demo shows the two tenants' entity sets are disjoint, that neither graph
contains the other's nodes, and that a query in one workspace can't retrieve the
other's content.

## What you get

Multi-tenant RAG on a single AgensGraph database — each tenant's graph, vectors,
documents, and ingestion status isolated by `workspace`.

> How it works: per-workspace **graph** isolation needs the graph name to include
> the workspace. `lightrag-agensgraph` folds `workspace` into the graph name
> (e.g. `acme_chunk_entity_relation`) so each tenant's graph is isolated alongside
> its relational stores. An empty workspace keeps the default graph, so existing
> single-tenant setups are unchanged.
