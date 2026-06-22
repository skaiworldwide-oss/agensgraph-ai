# 02 · Wikipedia — LLM-built knowledge graph + natural-language Q&A

Let an LLM extract a typed knowledge graph from text, store it in AgensGraph,
then ask questions in natural language — the LLM writes the Cypher for you.

## Run

```bash
# from llama-index/
.venv/bin/python examples/demos/02_wikipedia_pgindex/build.py   # LLM extraction (WIKI_LIMIT, default 500)
.venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py
.venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py "your question"

# quick dry-run:
WIKI_LIMIT=20 WIKI_RESET=1 .venv/bin/python examples/demos/02_wikipedia_pgindex/build.py
```

Knobs: `WIKI_LIMIT` (articles), `WIKI_CHARS` (lead chars/article), `WIKI_WORKERS`
(extraction concurrency), `WIKI_RESET=1`.

## The patterns (`build.py`)

Extract and store a knowledge graph in one call. Give it an ontology (entity and
relation types) for a clean, queryable graph:

```python
from typing import Literal
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=Literal["Person", "Organization", "Place", "Event", "Work"],
    possible_relations=Literal["FOUNDED", "LOCATED_IN", "BORN_IN", "CREATED", "MEMBER_OF", ...],
    strict=False,            # see Tips
)
index = PropertyGraphIndex.from_documents(
    docs,
    property_graph_store=AgensPropertyGraphStore("kg", conf, vector_dimension=1536, enhanced_schema=True),
    kg_extractors=[extractor],
    embed_model=embed, llm=llm,
)
```

## The patterns (`ask.py`)

The store ships an AgensGraph-aware Text2Cypher prompt as its default, so a plain
retriever stack works out of the box — combine keyword, vector and Cypher retrieval:

```python
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever, VectorContextRetriever, TextToCypherRetriever)

qe = index.as_query_engine(sub_retrievers=[
    LLMSynonymRetriever(graph_store=store, llm=llm),
    VectorContextRetriever(graph_store=store, embed_model=embed),
    TextToCypherRetriever(graph_store=store, llm=llm),   # NL → AgensGraph Cypher
])
print(qe.query("Which 5 entities are connected to the most others?"))
```

## What you get

A queryable knowledge graph built from prose, asked in plain English — with the
LLM translating questions into valid AgensGraph Cypher against your schema.

## Tips

- **`strict=False`** on `SchemaLLMPathExtractor`: with `strict=True`, gpt-4o-mini
  often returns *zero* triplets; `strict=False` still follows your ontology but
  reliably produces a graph.
- **Text2Cypher works out of the box** — the store sets an AgensGraph-dialect
  prompt (it knows the storage model and avoids Neo4j-only syntax). For
  robustness, wrap the retriever so one bad generation can't abort the query; the
  demo's `SafeTextToCypherRetriever` (in `_common/cypher.py`) does exactly that.
- `enhanced_schema=True` gives the LLM example property values for better Cypher.
- Extraction is LLM-latency-bound; `WIKI_WORKERS` parallelizes it.
