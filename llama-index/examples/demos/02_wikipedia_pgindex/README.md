# 02 · Wikipedia LLM-built PropertyGraphIndex (+ Text2Cypher)

The signature LlamaIndex graph feature: an **LLM extracts a typed knowledge
graph** from Wikipedia articles and writes it into AgensGraph via
`AgensPropertyGraphStore`, then the graph is queried with the full PropertyGraph
retriever stack — including **AgensGraph-dialect Text2Cypher**.

```
PropertyGraphIndex.from_documents(
    docs, property_graph_store=AgensPropertyGraphStore(enhanced_schema=True, ...),
    kg_extractors=[SchemaLLMPathExtractor(possible_entities=Person/Organization/Place/Event/Work,
                                          possible_relations=FOUNDED/LOCATED_IN/...)],
    embed_model=OpenAIEmbedding, llm=OpenAI)
```

## Run

```bash
cd llama-index
.venv/bin/python examples/demos/02_wikipedia_pgindex/build.py   # LLM extraction (WIKI_LIMIT, default 500)
.venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py     # retriever stack + Text2Cypher
.venv/bin/python examples/demos/02_wikipedia_pgindex/ask.py "your question"

# quick dry run:
WIKI_LIMIT=20 WIKI_RESET=1 .venv/bin/python examples/demos/02_wikipedia_pgindex/build.py
```

Knobs: `WIKI_LIMIT` (articles), `WIKI_CHARS` (lead chars/article), `WIKI_WORKERS`
(extraction concurrency), `WIKI_RESET=1`.

## What it demonstrates

- **`build.py`** — `PropertyGraphIndex.from_documents` with `SchemaLLMPathExtractor`
  over a curated ontology, `enhanced_schema=True` (so `get_schema_str` carries
  example values for better Cypher), entities embedded for vector retrieval.
- **`ask.py`** — `index.as_retriever(sub_retrievers=[…])` combining:
  `LLMSynonymRetriever`, `VectorContextRetriever` (entity `vector_query` +
  `get_rel_map`), and **`SafeTextToCypherRetriever`** with the AgensGraph dialect
  prompt. It prints the generated Cypher so you can see it's AgensGraph-valid.

## Gotchas this demo encodes

- **`strict=False`** on `SchemaLLMPathExtractor`: with `strict=True`, gpt-4o-mini's
  enforced structured output returns **zero** triplets; `strict=False` still emits
  only the ontology's types but actually produces a graph.
- **Text2Cypher dialect:** the store's default prompt yields Neo4j-style
  Cypher AgensGraph can't run. `_common/cypher.py` supplies `AGENS_CYPHER_PROMPT`
  (teaches the `"__Node__"` + `labels`-list model, forbids Neo4j-isms) and
  `SafeTextToCypherRetriever` (a bad generation degrades gracefully instead of
  crashing the query engine).

## Notes

- Wall-clock is **LLM extraction latency** (one structured-output call per chunk);
  `WIKI_WORKERS` parallelizes it. Keep `WIKI_CHARS` modest so an entity-dense
  article doesn't overrun the output-token budget.
