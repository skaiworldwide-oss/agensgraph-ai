# 🦜️🔗 LangChain AgensGraph

LangChain integration for [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), Skai's
PostgreSQL-based graph database: a graph store, a vector store and its retrievers, a Cypher QA
chain, an LLM graph transformer, chat message history, and a LangGraph checkpointer and store.
Every class works blocking and awaiting.

## Requirements

- Python 3.11 to 3.14.
- AgensGraph 2.14.1 or later (2.14, 2.15, 2.16, 2.17 and 2.18 are tested).
- The `vector` extension (pgvector 0.8.0 or later) for the vector store and the store's semantic
  search. AgensGraph does not bundle it; build it against your server's `pg_config`:

  ```bash
  git clone https://github.com/pgvector/pgvector.git
  cd pgvector && PG_CONFIG=/path/to/agens/bin/pg_config make && make install
  # then, in the database: CREATE EXTENSION vector;
  ```

## Installation

```bash
pip install -U langchain-agensgraph
pip install -U "langchain-agensgraph[langgraph]"   # the checkpointer and the store
```

## Connecting

One `AgensEngine` per process holds the connection pools; every component takes it, or a conninfo
string, and a graph name. Components handed the same conninfo string share one engine. Nothing
connects until the first call, and each component makes what it needs on that call.

```python
from langchain_agensgraph import AgensEngine, AgensGraph

engine = AgensEngine("postgresql://app_rw:secret@localhost:5432/db")
graph = AgensGraph(engine, graph_name="kg", create_graph=True)
graph.query("MATCH (n) RETURN count(*) AS n")
```

A component's tables live in the graph's companion schema, `"<graph>_store"`.

## Roles

`langchain_agensgraph.roles` prints the SQL for the roles the package expects:

```python
from langchain_agensgraph.roles import render_grants, render_roles

print(render_roles(password="..."))   # app_rw, llm_ro and llm_rw, once per server
print(render_grants("kg"))            # what each role may do on one graph, as its owner
```

- `app_rw` runs the application.
- `llm_ro` runs Cypher a model wrote. It reads the graph and nothing else.
- `llm_rw` runs model-written writes (`allow_writes=True`), on 2.16 and later only.

Give the engine those roles' conninfo strings:

```python
engine = AgensEngine(
    "postgresql://app_rw:secret@localhost/db",
    untrusted_conninfo="postgresql://llm_ro:secret@localhost/db",
)
```

## The graph store

```python
graph.query("MATCH (p:\"Person\") WHERE p.age > %s RETURN p.name AS name", (30,))
graph.query_many(["CREATE (:\"Tag\" {name: 'a'})", "CREATE (:\"Tag\" {name: 'b'})"])  # all or none
graph.add_graph_documents(documents, include_source=True, baseEntityLabel=True)
print(graph.get_schema)
```

- Rows hold the driver's `Vertex`, `Edge` and `Path` values; `sanitize=True` gives plain maps and
  leaves out lists longer than 128 items.
- A server refusal raises `AgensQueryError` carrying its `sqlstate`.
- The schema is read on first use and kept for `schema_cache_ttl` seconds. Its counts are the
  planner's estimates.
- `add_graph_documents` merges each node by `id`, and each edge once per pair of ends
  (`dedupe_edges=True`).
- `query_untrusted(text)` runs Cypher a model wrote, as `llm_ro`. The text is checked before it is
  sent and refused unless it is plainly one Cypher read. It then runs in a read-only transaction
  that is always rolled back, reading at most `max_rows` rows.

## The vector store

Each chunk is a vertex. Its embedding is a row of a table in `"<graph>_store"`, deleted with the
vertex.

```python
from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.vectorstores import KeywordConfig

store = AgensgraphVector(
    embeddings,
    engine=engine,
    graph_name="kg",
    label="Chunk",
    filter_keys=("source", "year"),         # indexed for filtered search
    keyword=KeywordConfig(regconfig="english"),
    search_type="hybrid",                    # vector and keyword, fused
    create_graph=True,
)
store.add_documents(docs, ids=ids)
store.similarity_search("what failed?", k=5, filter={"year": {"$gte": 2024}})
store.similarity_search_with_relevance_scores("what failed?", k=5)
store.max_marginal_relevance_search("what failed?", k=5, fetch_k=20)
store.delete(ids=["a"])
```

- The approximate index (HNSW by default, or IVFFlat) is built after the first write. Before
  then, and for a role that does not own the table, search is exact.
- Filters take `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`, `$between`, `$like`,
  `$ilike`, `$exists`, `$and`, `$or` and `$not`.
- `*_with_score` returns the distance; `*_with_relevance_scores` turns it into a relevance score.
  In hybrid mode both return the fused score; `hybrid_search_with_details` returns each half.
- Embeddings of up to 2,000 dimensions are stored as `vector`, wider ones as `halfvec`.

## Retrievers

```python
from langchain_agensgraph import (
    AgensGraphContextRetriever,
    AgensText2CypherRetriever,
    AgensVectorRetriever,
)
from langchain_agensgraph.retrievers import render_graph_context

AgensVectorRetriever(vectorstore=store, k=6)
AgensGraphContextRetriever(
    vectorstore=store, k=4, hops=2, document_formatter=render_graph_context
)
AgensText2CypherRetriever(graph=graph, llm=llm, allow_dangerous_requests=True)
```

- `AgensVectorRetriever` takes a `retrieval_query`, Cypher that shapes each hit; it sees `node`
  and `score`.
- `AgensGraphContextRetriever` adds each hit's neighbourhood, within `hops` edges, to its metadata.
- A call may pass `k=` and other search arguments; they win over the retriever's own.

## Questions in plain language

```python
from langchain_agensgraph import AgensCypherQAChain, create_cypher_tool

chain = AgensCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
chain.invoke({"query": "Who works at Acme?"})   # {"query": ..., "result": "Ada and Bob."}

tool = create_cypher_tool(graph, llm, allow_dangerous_requests=True)
```

The model's Cypher runs only through `query_untrusted`, as `llm_ro`. Before it is sent:

- the names the graph stores with capitals are quoted;
- an edge written against the only direction the schema has is turned round;
- the forms a release reads wrongly are rewritten.

A refusal, or a server error about the statement, goes back to the model, up to `max_retries`
times. The prompt carries the rules of the server it runs on. `examples=DIALECT_EXAMPLES` adds
few-shot pairs. `allow_writes=True` runs model writes as `llm_rw` on 2.16 and later.

## Extracting a graph from text

```python
from langchain_agensgraph import LLMGraphTransformer

transformer = LLMGraphTransformer(
    llm,
    allowed_nodes=["Person", "Company"],
    allowed_relationships=[("Person", "WORKS_AT", "Company")],
    node_properties=["role"],
)
graph.add_graph_documents(transformer.convert_to_graph_documents(docs), include_source=True)
```

A model without structured output answers in JSON instead (`ignore_tool_usage=True` forces it).

## Chat history, checkpoints and memory

```python
from langchain_agensgraph import AgensChatMessageHistory, AgensSaver, AgensStore

history = AgensChatMessageHistory("session-1", engine, graph_name="kg")
saver = AgensSaver(engine, graph_name="kg")           # sync and async graphs alike
store = AgensStore(engine, graph_name="kg", index={"dims": 1536, "embed": embeddings})
app = builder.compile(checkpointer=saver, store=store)
```

`AsyncAgensStore` is the store for an event loop. All three keep their state in plain tables in
`"<graph>_store"`.

## Agent state next to the graph

Each of them gives a SQL subquery a Cypher statement can hold, so one statement reads the agent's
state and the knowledge graph together:

```python
graph.query(
    f"UNWIND [1] AS a WITH {history.cypher_subquery()} AS messages "
    "UNWIND messages AS m MATCH (p:\"Person\") WHERE m.text CONTAINS p.name "
    "RETURN p.name AS who, m.seq AS seq",
    (history.session_id,),
)
```

`store.cypher_subquery()` gives the items in a namespace. For the checkpointer, declare the
channels to read: `AgensSaver(..., queryable_channels={"messages": "messages"})`, then use
`saver.cypher_subquery("messages")`.

## Transactions

Calls through an engine join a transaction the caller opens:

```python
with engine.transaction() as conn:
    store.add_documents(docs)
    graph.add_graph_documents(graph_docs)
    conn.execute("...")   # statements of your own, in the same transaction
```

## What differs between releases

| | 2.14 | 2.15 | 2.16 | 2.17 | 2.18 |
|---|---|---|---|---|---|
| model-written writes (`allow_writes`) | no | no | yes | yes | yes |
| deleting a vertex scans every edge label | yes | yes | yes | no | no |
| GQL clauses in the QA prompt | no | no | no | no | yes |

## Writing Cypher for AgensGraph

- Quote labels and types with capitals: `(n:"Person")`. An unquoted name folds to lower case.
- Write edges with brackets, `(a)-[]->(b)`. AgensGraph reads `--` as the start of a comment.
- Name a variable-length edge, `[r:"KNOWS"*1..3]`; an unnamed one is wrong on 2.14 and 2.15.
- Before 2.17, `ORDER BY` names a returned alias, and `x IN %s` with a list parameter fails; use
  `UNWIND`.
- On 2.14, `collect()` keeps nulls and gives null for no rows.
- `id(n) = '3.1'` is the fastest lookup. Inside a SQL subquery, read a vertex's properties with
  `properties(n)`.
- Cast to convert, `n.x::float8`; `toFloat()` and `toInteger()` are unreliable.

## Upgrading from 0.3

See [CHANGELOG.md](CHANGELOG.md).

## Demos and evals

Runnable demos are under [`examples/demos/`](examples/demos/). `evals/text2cypher/` scores
generated Cypher by running it.

## License

Apache-2.0.
