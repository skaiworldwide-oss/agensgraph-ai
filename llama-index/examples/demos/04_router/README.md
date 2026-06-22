# 04 · Router — one engine, graph + vector

Route a natural-language question to the right store: a scientific question goes
to the arXiv **graph** engine (demo 1), a current-events question to the news
**vector** engine (demo 3). Both run on the **same `AgensEngine` pool** —
different graphs in one database, served together. The demo then hands the same
two tools to a `FunctionAgent` as an autonomous alternative to the router.

📓 **Guided tour:** [`router.ipynb`](./router.ipynb) is a pre-executed notebook
walking through routing + the shared pool with real outputs.

## Run

Run after demos 1 and 3 have populated their graphs:

```bash
# from llama-index/
.venv/bin/python examples/demos/04_router/router.py
.venv/bin/python examples/demos/04_router/router.py "your question"
```

## The pattern

Build a query engine over each store, wrap each as a tool, and let a selector
route:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

engine = agens.get_engine()                       # ONE shared pool
graph_qe = PropertyGraphIndex.from_existing(arxiv_store, ...).as_query_engine(...)
news_qe  = VectorStoreIndex.from_vector_store(news_store, ...).as_query_engine(...)

router = RouterQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool.from_defaults(graph_qe, name="arxiv_papers",
            description="academic / scientific questions about papers, authors, methods"),
        QueryEngineTool.from_defaults(news_qe, name="news_articles",
            description="current events: business, technology, world news"),
    ],
    selector=LLMSingleSelector.from_defaults(llm=llm),
)
router.query("What are companies doing with AI?")   # -> news engine
router.query("Who works on graph neural networks?") # -> arXiv graph engine
```

Or hand the **same two tools** to a `FunctionAgent` and let it choose (and chain)
calls itself — `run` is async, so it exercises the engine's async pool:

```python
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(tools=[graph_tool, news_tool], llm=llm)
await agent.run("What are companies doing with artificial intelligence?")
```

## What you get

A single natural-language entry point — router *or* agent — over a graph store
and a vector store on different AgensGraph graphs, one connection pool, no
cross-talk.

## Tips

- Pass `engine=` to every store to share one pool. The engine re-binds
  `graph_path` on each checkout, so the two graphs never collide.
- The router (one LLM selector pick) is the simpler, more predictable default;
  the `FunctionAgent` is the autonomous alternative — it can call more than one
  tool. After an async agent run, close with `await agens.aclose()` (not the sync
  `close()`) so the async pool's workers are released from inside the event loop.
