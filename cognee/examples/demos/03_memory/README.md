# 03 · Memory — many datasets, one growing memory

cognee is a **memory layer**: you grow one memory over time from named
**datasets**, never rebuilding from scratch. This demo builds two datasets in one
`cognee_memory` database — an `encyclopedia` (Wikipedia) and then a `news` feed
(CC-News) — cognifying them one after the other so you can **watch the graph grow**,
then queries the **unified** memory, where a single search draws on both.

📓 **Just want to read it?** Open [`memory.ipynb`](./memory.ipynb) — already executed,
with real outputs.

## Run it

```bash
# from the cognee/ directory of this repo
MEM_LIMIT=15 .venv/bin/python examples/demos/03_memory/build.py   # start tiny: a dry-run
.venv/bin/python examples/demos/03_memory/build.py                # ~60 + ~60 docs
.venv/bin/python examples/demos/03_memory/ask.py
```

Knobs: `MEM_LIMIT` (docs per dataset), `MEM_CHARS`, `MEM_RESET=0` (add to the existing
memory). `build.py` prints a cost estimate first.

## The pattern

```python
# grow the memory one dataset at a time (each tagged with a node_set)
await cognee.add(wiki_docs, dataset_name="encyclopedia", node_set=["reference"])
await cognee.cognify(["encyclopedia"])
await cognee.add(news_docs, dataset_name="news", node_set=["current_events"])
await cognee.cognify(["news"])                       # extends the memory — no rebuild

# query the unified memory (draws on every dataset)
await cognee.search(query_text="...", query_type=SearchType.GRAPH_COMPLETION)

# or pull back just one dataset's slice, by its node_set tag
from cognee.modules.engine.models.node_set import NodeSet
nodes, edges = await (await get_graph_engine()).get_nodeset_subgraph(NodeSet, ["current_events"])
```

`build.py` shows the graph growing as the second dataset is cognified; `ask.py` runs
one query that returns hits from **both** datasets, then uses `get_nodeset_subgraph`
to show the size of each dataset's tagged slice.

> **Good to know — `datasets=[...]` is a permission scope, not a retrieval filter.**
> cognee's `search(datasets=[...])` argument checks which datasets you may read; it
> does **not** restrict retrieval to them — a scoped query still draws on the whole
> unified memory (the graph and vector collections are shared). For hard per-tenant
> isolation, give each tenant its own database (as the other demos do, one DB each).
> To scope *within* one memory, the `node_set` tags above are the tool.

## What you get

A single AgensGraph-backed memory you grow incrementally from multiple named
datasets — accumulate knowledge over time, query it as one, and slice it by tag.
