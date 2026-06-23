# 03 · Memory — many datasets, one database, isolated or together

cognee is a **memory layer**: you grow one memory over time from named
**datasets**, with no full rebuild. This builds two datasets in one
`cognee_memory` database — an `encyclopedia` (Wikipedia) and then a `news` feed
(CC-News) — cognifying them one after the other so you can watch the graph grow,
then queries the **unified** memory (a single search draws on both datasets).

📓 **Guided tour:** [`memory.ipynb`](./memory.ipynb).

## Run

```bash
# from cognee/
MEM_LIMIT=15 .venv/bin/python examples/demos/03_memory/build.py   # tiny dry-run
.venv/bin/python examples/demos/03_memory/build.py               # ~60 + ~60 docs
.venv/bin/python examples/demos/03_memory/ask.py
```

Knobs: `MEM_LIMIT` (docs per dataset), `MEM_CHARS`, `MEM_RESET=0`.

## The pattern

```python
# build two datasets incrementally (each also tagged with a node_set)
await cognee.add(wiki_docs, dataset_name="encyclopedia", node_set=["reference"])
await cognee.cognify(["encyclopedia"])
await cognee.add(news_docs, dataset_name="news", node_set=["current_events"])
await cognee.cognify(["news"])                       # grows the memory, no rebuild

# query the unified memory (draws on every dataset)
await cognee.search(query_text="...", query_type=SearchType.GRAPH_COMPLETION)
```

`build.py` shows the graph growing as the second dataset is cognified (no
rebuild); `ask.py` runs one query and shows results coming from **both** datasets.

> Heads-up: cognee's `datasets=[...]` search argument is a read-**permission**
> scope, not a retrieval filter — a scoped query still draws on the whole unified
> memory (the vector collections and graph are shared). For hard per-tenant
> isolation, use a separate database per tenant (one cognee DB each).

## What you get

A single AgensGraph-backed memory you grow incrementally from multiple named
datasets — accumulate knowledge over time and query it as one.
