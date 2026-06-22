# 04 · Router — one AgensEngine, both stores

Ties the suite together: a LlamaIndex `RouterQueryEngine` routes a natural-language
question to either the arXiv **property-graph** engine (demo 1) or the news
**vector** engine (demo 3). Both stores run on the **same shared `AgensEngine`
pool** — different graphs in one database, served concurrently.

## Run

Run after demos 1 and 3 have populated their graphs, then:

```bash
cd llama-index
.venv/bin/python examples/demos/04_router/router.py
.venv/bin/python examples/demos/04_router/router.py "your question"
```

## What it demonstrates

- One `agens.get_engine()` pool backs **both** an `AgensPropertyGraphStore`
  (graph `arxiv`) and an `AgensgraphVectorStore` (graph `news`).
- `RouterQueryEngine` with an `LLMSingleSelector` picks the right engine per
  question (and prints its reasoning):
  - a scientific question → the arXiv graph engine (`VectorContextRetriever`);
  - a current-events question → the news vector engine.
- A `pg_stat_activity` readout confirms the connections all come from the single
  `application_name = 'llama-index-agensgraph'` pool.

## The end result

A graph engine and a vector engine, on different AgensGraph graphs in one
database, behind one connection pool and one natural-language entry point —
demonstrating that the property-graph store and the vector store interoperate in
a single LlamaIndex pipeline.

## Notes

- The engine re-applies `SET graph_path` on every pooled checkout, so the two
  stores never collide despite sharing the pool.
- An agentic alternative (a `FunctionAgent` over the same two `QueryEngineTool`s)
  is a natural extension; the router is the more deterministic, demonstrative default.
