# 04 · Curation — a LightRAG knowledge graph is maintainable

Extraction isn't perfect and corpora change. LightRAG supports the full curation
lifecycle on a live graph: **merge** duplicate entities, **edit** entities and
relations, **delete** an entity (purging its vectors), and **delete a whole
document** (regenerating what it contributed). This runs on a small,
deterministic `lightrag_curation` graph built from a handful of crafted
paragraphs, so every before/after is easy to read.

📓 **Guided tour:** [`curation.ipynb`](./curation.ipynb).

## Run

```bash
# from lightrag/  (tiny + fast — a few crafted docs)
.venv/bin/python examples/demos/04_curation/curate.py
```

Knob: `CURATION_RESET=0` keeps the graph between runs (default rebuilds it).

## The lifecycle

```python
# merge two entities the extractor kept separate (e.g. "OpenAI" / "Open AI")
await rag.amerge_entities(source_entities=["Open AI"], target_entity="OpenAI")

# edit an entity's attributes (rename allowed)
await rag.aedit_entity("OpenAI", {"description": "...", "entity_type": "organization"})

# edit a relation between two entities
await rag.aedit_relation("OpenAI", "Microsoft", {"description": "...", "weight": 9.0})

# delete an entity (and its vectors), or a whole source document
await rag.adelete_by_entity("Google DeepMind")
await rag.adelete_by_doc_id("doc-0")        # -> DeletionResult; doc-status updated
```

The demo prints entity counts and node/edge state before and after each operation
so you can see the merge fold one node into another, the edits land, and the
deletes remove entities/documents (and their vectors) cleanly.

## What you get

A knowledge graph you can keep clean over its lifetime — dedup, edit, and
retraction — with the graph and its `pgvector` indexes staying consistent.
