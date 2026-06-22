# 02 · Incremental — grow the graph over time, watch entities merge

A corpus arrives in waves. LightRAG ingests each wave **incrementally**: new
documents extend the same knowledge graph, entities that recur across documents
are **merged into one node** (its degree and description grow), the **doc-status
pipeline** tracks every document (with a per-wave `track_id`), and re-submitting a
document is a **no-op**. This runs on CC-News (people/companies/places recur
across articles) in the `lightrag_news` database.

📓 **Guided tour:** [`incremental.ipynb`](./incremental.ipynb).

## Run

```bash
# from lightrag/
NEWS_LIMIT=40 .venv/bin/python examples/demos/02_incremental/ingest.py   # tiny dry-run
.venv/bin/python examples/demos/02_incremental/ingest.py                 # ~600 articles, 2 waves
```

Knobs: `NEWS_LIMIT` (articles total), `NEWS_WAVES` (default 2), `NEWS_CHARS`,
`NEWS_RESET=1`. Like demo 1, insert is extraction-bound — a cost estimate prints first.

## The pattern

```python
# wave 1, then wave 2 — each tagged with a track_id for provenance
await rag.ainsert(texts_w1, ids=ids_w1, file_paths=urls_w1, track_id="wave-1")
await rag.ainsert(texts_w2, ids=ids_w2, file_paths=urls_w2, track_id="wave-2")

await rag.doc_status.get_all_status_counts()         # processed vs failed (failed includes duplicates)
await rag.doc_status.get_docs_by_track_id("wave-1")  # which docs arrived in wave 1
await rag.doc_status.get_docs_paginated(page=1, page_size=5, sort_field="updated_at")

# entities that recur across many documents are merged into one node
node = await rag.chunk_entity_relation_graph.get_node(entity)   # description + accumulated source_id
await rag.chunk_entity_relation_graph.node_degree(entity)
```

After both waves the demo lists the **most cross-document entities** — those found
in the most source documents, which LightRAG has merged into a single node
(accumulating their relationships and source chunks) — then re-submits an
already-processed document and shows it's **detected as a duplicate, not
re-processed**.

> Public news corpora contain many duplicate articles; LightRAG detects them (by
> filename / content hash) and records them as `failed` in doc-status rather than
> re-extracting — so a non-zero `failed` count here is the dedup pipeline at work,
> not an error.

## What you get

A living knowledge graph you can keep feeding — incremental indexing with no full
rebuild, full ingestion bookkeeping (doc-status + track_id), and automatic
cross-document entity consolidation — all in one AgensGraph database.
