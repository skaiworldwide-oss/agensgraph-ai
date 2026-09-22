# 🦜️🔗 LangChain AgensGraph

LangChain integration for [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), Skai's PostgreSQL-based multi-model graph database. Ships a `GraphStore`, a pgvector-backed `VectorStore`, chat-message history, a LangGraph checkpointer, a LangGraph long-term memory store, an LLM graph transformer, and a connection-pooling engine — with async variants throughout.

## What's new in 0.3.0

Every statement goes through the [`agensgraph-python`](https://github.com/skaiworldwide-oss/agensgraph-python) 2.0 driver, and the package gains retrievers, a text2cypher chain and a LangGraph store.

**On the driver**
- `AgensEngine` is a driver connection pool, sync and async, shared by every component built on it.
- `query()` returns the driver's `Vertex`, `Edge` and `Path` values, with the label, the id and the properties kept apart. `query_many()` runs a burst of reads without waiting for each in turn.
- A `str` parameter is sent as text. `json.dumps(...)` around a string parameter is no longer needed, and has to go: a JSON-encoded string now matches nothing.
- `AgensgraphVector` is on the driver's vector types and indexes. `filter_properties` names the metadata properties a search filters on, and they get indexes, so a filtered search reads from an index rather than the label.
- `log_queries()` reports every statement a block of code ran, and how long each took.

**Retrievers** — three `BaseRetriever`s over the same store and pool, each one server round trip: `AgensVectorRetriever`; `AgensGraphContextRetriever`, which fetches each seed's neighbourhood within `expand_by_hops` in the seed search's own statement, with `render_graph_context` to format it for a prompt; and `AgensText2CypherRetriever`, where a model writes the Cypher and the server contains it, with `max_retries` and `retry_on_empty` feeding a failure back for a corrected attempt. See [Retrievers](#retrievers).

**Text2Cypher** — `AgensCypherQAChain` answers a question with generated Cypher run in a read-only transaction, and `create_cypher_tool` gives an agent the same. The schema the model reads names the graph's property indexes (`include_indexes`), `examples=` adds few-shot pairs and `DIALECT_EXAMPLES` is a curated set of them; `evals/text2cypher/` scores generated Cypher by executing it. See [Asking questions in plain language](#asking-questions-in-plain-language-text2cypher).

**LangGraph** — `AgensStore`, a `BaseStore` whose filters and namespace search run in the database. `AgensSaver` / `AsyncAgensSaver` complete the thread lifecycle with `prune`, `copy_thread` and `delete_for_runs`, and page the history reads. See [AgensStore](#agensstore-langgraph-long-term-memory).

**Fixed**
- A store search filter ignored its comparison operators.
- `prune` deleted the checkpoints a delta channel rebuilds from, so the channel came back short of its history with no error.
- `LLMGraphTransformer` wrote duplicate nodes when an entity type differed only in case.
- `AgensChatMessageHistory` appends and windowed reads grew with the length of the history.
- `AgensQueryException` read `details` while every raise wrote `detail`, so the server's account of a failure was lost.

**Requirements** — AgensGraph 2.17 or later, refused at connect below it; Python 3.11–3.14.

## Installation

```bash
pip install -U langchain-agensgraph
```

### AgensGraph requirements

AgensGraph 2.17 or later is required; `agensgraph-python` 2.0 refuses an older server at connect. AgensGraph does **not** bundle the pgvector or `meta` extensions; build and install them against your AgensGraph install's `pg_config`:

```bash
# pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector && PG_CONFIG=/path/to/agens/bin/pg_config make && make install

# meta extension (ships in AgensGraph's contrib/)
cd /path/to/agensgraph/contrib/meta
PG_CONFIG=/path/to/agens/bin/pg_config make USE_PGXS=1 install

# in your AgensGraph database:
CREATE EXTENSION vector;
CREATE EXTENSION meta;
```

The integration works without `meta` (falls back to `ag_label` catalog scans) but `refresh_schema` is much faster with it.

## Usage

### AgensGraph (graph store)

```python
from langchain_agensgraph import AgensGraph

conf = {
    "dbname": "...",
    "user": "...",
    "password": "...",
    "host": "...",
    "port": 5432,
}

graph = AgensGraph(graph_name="my_graph", conf=conf, create=True)
graph.query("MATCH (n) RETURN n LIMIT 1")

# Optional: cache the schema between refreshes (seconds)
graph = AgensGraph(graph_name="my_graph", conf=conf, schema_cache_ttl=60)

# Async
results = await graph.aquery("MATCH (n) RETURN count(n) AS c")
await graph.aclose()
```

### AgensgraphVector (vector store)

```python
from langchain_agensgraph import AgensgraphVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = AgensgraphVector.from_documents(
    docs,
    embeddings,
    url="postgresql://user:pwd@host:5432/db",
)

# Search
docs_with_score = db.similarity_search_with_score("What is LangChain?", k=4)

# Higher recall — fetch 3× candidates from the ANN index, then trim to k
hits = db.similarity_search("...", k=10, effective_search_ratio=3.0)

# Mutation
db.add_texts(["...", "..."], ids=["a", "b"], batch_size=500)
db.delete(["a"])
got = db.get_by_ids(["b"])

# Async
hits = await db.asimilarity_search("...", k=10)
await db.aadd_texts(["..."], batch_size=500)
await db.aclose()
```

### Retrievers

Three `BaseRetriever`s over the same store and pool, each one server round trip
per retrieval, sync and async alike:

```python
from langchain_agensgraph import (
    AgensGraphContextRetriever,
    AgensText2CypherRetriever,
    AgensVectorRetriever,
)
from langchain_agensgraph.retrievers import render_graph_context

# Seed-only: the store's vector (or server-side hybrid RRF) search, scored.
retriever = AgensVectorRetriever(store=db, k=6)
docs = retriever.invoke("what failed over the weekend?")
docs[0].metadata["score"]

# Seeds plus their neighbourhood within N hops, fetched in the SAME statement
# as the seed search — measured 3–4.5x over querying each seed's context
# separately. metadata carries _context_nodes_ / _context_rels_; the ready-made
# formatter renders them under the text for a prompt.
retriever = AgensGraphContextRetriever(
    store=db, k=4, expand_by_hops=2, document_formatter=render_graph_context
)

# A model writes the Cypher; the server contains it (write refusal, EXPLAIN
# check, read-only transaction, one deadline across the attempt). max_retries
# feeds an execution error back for a corrected attempt, on the same budget;
# retry_on_empty extends that to a query that runs clean and matches nothing
# (what a relationship written against its schema direction looks like).
retriever = AgensText2CypherRetriever(
    graph=graph, llm=llm, max_retries=1, retry_on_empty=True
)
rows = retriever.invoke("How many people joined after 2024?")
rows[0].metadata["cypher"]

# Every knob is a per-call override too:
retriever.invoke("...", k=10)
await retriever.ainvoke("...")
```

`AgensVectorRetriever` also takes a `retrieval_query` that shapes what each hit
returns — several retrievers can share one store while each reads a different
context (the query sees `node` and `score`; braces are doubled, as in the
store's own contract).

### AgensStore (LangGraph long-term memory)

`AgensSaver` persists a single thread's state; `AgensStore` is the other half — cross-thread
long-term memory, implementing LangGraph's `BaseStore` (`get` / `put` / `search` / `delete` /
`list_namespaces`, sync and async). Because items are ordinary vertices, a memory can be linked
to other memories and to your domain data with ordinary edges — which is the thing LangGraph's
stock `PostgresStore` cannot do.

```python
from langchain_agensgraph import AgensGraph, AgensStore

graph = AgensGraph("memories", conf=conf, create=True)
store = AgensStore(graph=graph)

store.put(("users", "alice", "memories"), "m1", {"text": "prefers tea", "topic": "prefs"})
item = store.get(("users", "alice", "memories"), "m1")

# a namespace search also returns its descendants
hits = store.search(("users", "alice"), filter={"topic": "prefs"}, limit=10)
namespaces = store.list_namespaces(prefix=("users", "*", "memories"))

# async throughout
item = await store.aget(("users", "alice", "memories"), "m1")
```

Semantic search is opt-in and needs `pgvector`:

```python
from langchain_openai import OpenAIEmbeddings

store = AgensStore(
    graph=graph,
    index={"dims": 1536, "embed": OpenAIEmbeddings(), "fields": ["text"]},
)
hits = store.search(("users", "alice"), query="what do they drink?", limit=5)
```

Embeddings are deliberately **not** stored as a property. A vector serialised into the jsonb
property bag pushes the bag out of line, after which every property read on that row pays a
detoast. They live instead in a narrow table in a companion schema (`<graph>_store`), keyed by
graphid with an HNSW index, and a foreign key that cascades — so deleting a memory removes its
embedding with it and searching never touches jsonb until the surviving rows are read back.

Two notes on the storage layout:

- **Namespaces** are flattened to a `.`-joined path. LangGraph already forbids `.` inside a
  namespace label, so nothing is lost and no escaping is needed.
- **`promoted=[...]`** is an opt-in tier that mirrors chosen properties into typed columns, so
  filters and sorts compare in the column's native type rather than as jsonb. It is faster, but
  it stores those values twice for now, and native comparison is not jsonb comparison — so
  filter and sort results can legitimately differ from the default layout. Leave it unset unless
  you have measured a reason to set it.

### Asking questions in plain language (text2cypher)

`AgensCypherQAChain` turns a question into Cypher, runs it read-only, and answers from the
rows. No Neo4j package required.

```python
from langchain_agensgraph import AgensCypherQAChain, AgensGraph

graph = AgensGraph("my_graph", conf=conf)
chain = AgensCypherQAChain.from_llm(llm, graph=graph)

chain.invoke({"query": "Which people work at Acme?"})
# {"query": "...", "result": "Alice and Bob."}

# see what it ran
chain = AgensCypherQAChain.from_llm(llm, graph=graph, return_intermediate_steps=True)
out = chain.invoke({"query": "..."})
out["intermediate_steps"]   # [{"query": "MATCH ..."}, {"context": [...]}]
```

The same pipeline as a tool for an agent:

```python
from langchain.agents import create_agent
from langchain_agensgraph import create_cypher_tool

agent = create_agent(llm, [create_cypher_tool(graph, llm)])

# or hand the agent rows instead of prose, to combine with other sources
tool = create_cypher_tool(graph, llm, answer=False)
```

Generated Cypher is not trusted. Writes are refused unless you pass
`allow_dangerous_requests=True`, the query is checked with `EXPLAIN` before it runs so a
malformed one fails without executing, and execution carries a `timeout`.

**Why a dedicated prompt.** AgensGraph is not Neo4j, and a model writing Cypher from habit
gets two things wrong. It uses constructs that do not exist here — pattern expressions like
`size((n)--())`, `COUNT { }`, `EXISTS { }`, `CALL { }` subqueries, `apoc.*` — which fail
loudly. And it leaves identifiers unquoted, which fails silently: AgensGraph folds an
unquoted identifier to lower case, so `(n:Person)` matches nothing at all. The prompt rules
the first out by name and requires quoting for the second.

Because folding runs both ways, the repair pass is driven by your schema rather than by a
rule of thumb. A property written as `{"firstName": ...}` is stored case-preserved and must
be read as `n."firstName"`; the same property written as `{firstName: ...}` is stored as
`firstname` and must be read bare. The chain quotes an identifier only when your graph
actually holds that spelling, so it cannot turn a working query into a silent miss.

### Shared connection pool

```python
from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector

engine = AgensEngine.from_url("postgresql://user:pwd@host:5432/db", min_size=2, max_size=20)
graph = AgensGraph("my_graph", conf={...}, engine=engine, create=True)
store = AgensgraphVector(embeddings, graph_name="my_graph", engine=engine)
# ... concurrent requests each borrow their own pooled connection ...
engine.close()
```

## Production tips

- **Connection pooling**: use `AgensEngine` (backed by `psycopg-pool`) and share it across your graph and vector stores so concurrent requests don't serialize on a single connection.
- **PgBouncer transaction mode**: AgensGraph speaks the standard PG wire protocol, so PgBouncer works unchanged. In transaction-pool mode, disable psycopg's server-side prepared-statement cache (`prepare_threshold=None`).
- **HNSW + AgensGraph 2.17**: two June-2026 commits (`e7e1be9`, `47b38ed`) finally make `CREATE PROPERTY INDEX ... USING HNSW (((embedding)::vector(N)) vector_cosine_ops)` use an `Index Scan` plan instead of falling back to seq-scan. If you see seq-scan on v2.17 with a small table, that's expected — the planner picks seq-scan when it's cheaper.
- **`auto_gather_graphmeta`**: enable on the database (`ALTER DATABASE x SET auto_gather_graphmeta = on`) for ~30× faster `DETACH DELETE` on large graphs.

## Compatibility

| | `0.2.0` | `0.3.0` |
|---|---|---|
| Database access | psycopg 3 directly | [`agensgraph-python`](https://github.com/skaiworldwide-oss/agensgraph-python) 2.0 |
| `query()` rows | property maps, decoded by regex | `Vertex`, `Edge` and `Path` values |
| A `str` parameter | JSON text, so `json.dumps(...)` was needed | text, as given |
| AgensGraph | 2.17 recommended | 2.17 or later, refused at connect below it |
| Python | 3.10–3.14 | 3.11–3.14 |
| `langchain-core` / `langgraph` | `>=1.0.0,<2.0.0` | unchanged |
| Retrievers | — | `AgensVectorRetriever`, `AgensGraphContextRetriever`, `AgensText2CypherRetriever` |
| Text2Cypher | — | `AgensCypherQAChain`, `create_cypher_tool`, `evals/text2cypher/` |
| LangGraph | checkpointer | checkpointer with the full thread lifecycle, and `AgensStore` |

## License

Apache-2.0.
