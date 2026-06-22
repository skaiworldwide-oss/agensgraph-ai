# Build with AgensGraph + LlamaIndex

`llama-index-agensgraph` gives you two LlamaIndex stores backed by a single
AgensGraph (PostgreSQL) database:

- **`AgensPropertyGraphStore`** — a property graph for `PropertyGraphIndex`:
  build a knowledge graph (by hand or with an LLM), run Cypher, do vector +
  graph retrieval, and natural-language-to-Cypher.
- **`AgensgraphVectorStore`** — a vector store for `VectorStoreIndex`: HNSW
  semantic search, metadata filters, and hybrid (vector + keyword) search.

One database, one connection pool (`AgensEngine`) — graph RAG, vector RAG and
NL-to-Cypher without running a separate graph DB and vector DB.

These demos run on real public datasets (arXiv, Wikipedia, CC-News) and use
**OpenAI** for all embeddings/LLM (nothing runs on a local model).

## Quickstart

From the `llama-index/` directory of this repo:

```bash
# 1. install the integration + demo extras (uv)
uv venv .venv --python 3.13
uv pip install -e . -r examples/demos/requirements-demos.txt

# 2. a database (AgensGraph 2.17+ with pgvector, on localhost:55432)
createdb -h 127.0.0.1 -p 55432 llamaindex_demos
psql -h 127.0.0.1 -p 55432 -d llamaindex_demos -c 'CREATE EXTENSION IF NOT EXISTS vector'

# 3. your OpenAI key
cp examples/demos/.env.example examples/demos/.env     # then edit: OPENAI_API_KEY=...

# 4. run a demo — start small (a couple of cents, ~1 min)
ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_pg/prepare.py
.venv/bin/python examples/demos/01_arxiv_pg/query.py
```

## The building blocks

Share one pool across stores:

```python
from llama_index_agensgraph.engine import AgensEngine
engine = AgensEngine.from_url("postgresql://you@localhost:55432/llamaindex_demos")
```

A **property graph store** (knowledge graph + Cypher + graph/vector retrieval):

```python
from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
graph = AgensPropertyGraphStore(
    graph_name="kg",
    conf={"dbname": "llamaindex_demos", "user": "you", "host": "localhost", "port": 55432},
    vector_dimension=1536,   # build the HNSW index on entity embeddings
    engine=engine,
)
```

A **vector store** (semantic / filtered / hybrid search):

```python
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore
vectors = AgensgraphVectorStore(
    url="postgresql://you@localhost:55432/llamaindex_demos",
    embedding_dimension=1536,
    graph_name="docs", node_label="Chunk",
    hybrid_search=False,     # True for vector + keyword RRF
    engine=engine,
)
```

## The demos

Each folder is a focused, runnable example with its own README and copy-paste
code patterns.

| # | What you build | Run |
|---|----------------|-----|
| [**01_arxiv_pg**](01_arxiv_pg) | a structured graph + vector search + GraphRAG over arXiv | `prepare.py` → `query.py` |
| [**02_wikipedia_pgindex**](02_wikipedia_pgindex) | an LLM-extracted knowledge graph + natural-language Q&A | `build.py` → `ask.py` |
| [**03_news_vector_rag**](03_news_vector_rag) | vector RAG over news: semantic, filtered, hybrid, cited | `ingest.py` → `rag.py` |
| [**04_router**](04_router) | one engine routing questions to the graph or the vector store | `router.py` |

## Scale, cost, knobs

Big by default (~50k arXiv papers, ~100k news chunks, ~500 Wikipedia articles;
≈$2–4 OpenAI for a full run). Start small with the `*_LIMIT` env vars (all listed
in [`.env.example`](.env.example)); every prepare/ingest script prints a
token-and-cost estimate before embedding.

At scale the wall-clock is **OpenAI embedding/LLM latency**, not AgensGraph —
graph ingest is seconds; the scripts embed in parallel.

## Configuration

- **Database** `llamaindex_demos` by default. Connection comes from
  `AGENS_DB / AGENS_USER / AGENS_PASSWORD / AGENS_HOST / AGENS_PORT`, or a single
  `AGENS_URL`. Under the dev instance's trust auth no password is needed.
- **Models** `text-embedding-3-small` + `gpt-4o-mini`; override with
  `DEMO_EMBED_MODEL` / `DEMO_LLM_MODEL`.
- The demo scripts wire all of this up once in `_common/` (one shared
  `AgensEngine`, OpenAI models, config) so each demo stays focused on its own code.
