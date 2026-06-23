# 02 · Typed — ontology-guided extraction

By default cognee lets the LLM pick whatever entity types it likes
("organization", "thing", …). Hand it a domain **ontology** and extraction is
grounded in *your* vocabulary instead: the ontology's classes become `EntityType`
nodes, and entities matched to them are flagged `ontology_valid`. Same robust
graph build as demo 1, now schema-aligned.

📓 **Guided tour:** [`typed.ipynb`](./typed.ipynb).

## Run

```bash
# from cognee/
TYPED_LIMIT=15 .venv/bin/python examples/demos/02_typed/build.py   # tiny dry-run
.venv/bin/python examples/demos/02_typed/build.py                 # ~120 articles
.venv/bin/python examples/demos/02_typed/ask.py
```

Knobs: `TYPED_LIMIT`, `TYPED_CHARS`, `TYPED_RESET=0`. The ontology is
[`ontology.ttl`](./ontology.ttl) (Turtle/OWL) — edit it to fit your domain.

## The pattern

```python
# ontology.ttl declares the classes you care about (Person, Organization, …)
await cognee.cognify(["typed"], ontology_file_path="ontology.ttl")

# extracted entities now align to the ontology
nodes, _ = await (await get_graph_engine()).get_graph_data()
[p["name"] for _, p in nodes if p["type"] == "EntityType"]         # your classes
[p["name"] for _, p in nodes if p.get("ontology_valid")]           # aligned entities
```

`build.py` reports how many entities aligned to the ontology and which classes
the graph ended up using; `ask.py` shows the typed triplets (`INSIGHTS`) and a
grounded answer.

## What you get

A knowledge graph that follows a schema you control — entities resolved against a
domain ontology instead of ad-hoc LLM labels — stored in AgensGraph.

> Tip: cognee also accepts a custom pydantic `graph_model` for cognify, but that
> path is best for **flat document-level typing** (e.g. a `ScientificPaper` record
> per chunk); for a connected, typed *entity graph*, the ontology route above
> keeps cognee's full entity/relationship extraction.
