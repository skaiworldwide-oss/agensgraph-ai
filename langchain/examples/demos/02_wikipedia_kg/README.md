# 02 · Wikipedia knowledge graph + Text2Cypher

The **LangChain-forward** demo. An LLM builds a knowledge graph from unstructured
Wikipedia text, and natural-language questions are answered by generating Cypher
— both through LangChain idioms.

```
(:Person|Organization|Location|Event|Concept|Work|…)-[:<LLM-named>]->(…)
(:Document {title,url})-[:MENTIONS]->(entity)        # provenance
```

## Run

```bash
cd langchain
.venv/bin/python examples/demos/02_wikipedia_kg/build_kg.py     # extract + load (WIKI_LIMIT, default 500)
.venv/bin/python examples/demos/02_wikipedia_kg/ask.py          # ask in natural language
.venv/bin/python examples/demos/02_wikipedia_kg/ask.py "Which organizations are mentioned, and where are they located?"

# quick, near-free dry run:
WIKI_LIMIT=30 WIKI_RESET=1 .venv/bin/python examples/demos/02_wikipedia_kg/build_kg.py
```

Knobs: `WIKI_LIMIT` (articles), `WIKI_CHARS` (lead chars per article fed to the
LLM, default 1800), `WIKI_CONCURRENCY` (parallel extractions, default 8),
`WIKI_RESET=1` (rebuild the graph).

## Where LangChain does the work

- **`build_kg.py`** — `LLMGraphTransformer` (LangChain): the LLM extracts typed
  entities and relationships from each article as **structured output**, returning
  `GraphDocument`s that drop straight into `AgensGraph.add_graph_documents(...,
  include_source=True)` (which also records the source article + `MENTIONS`
  edges). Extraction is run with bounded concurrency via `aconvert_to_graph_documents`.
- **`ask.py`** — an idiomatic **LCEL** Text2Cypher pipeline:

  ```python
  RunnablePassthrough.assign(cypher = cypher_prompt | llm | StrOutputParser() | clean)
  | RunnablePassthrough.assign(results = run_cypher)          # read-only, timed
  | RunnablePassthrough.assign(answer  = answer_prompt | llm | StrOutputParser())
  ```

  The graph's `get_schema` (with `enhanced_schema=True`, so it carries example
  property values) is fed to the model so it writes valid AgensGraph Cypher
  (double-quoted labels, read-only, bounded). Generated queries are checked to be
  read-only and run with a statement timeout before grounding the answer.

## Notes

- Uses OpenAI for both extraction and Text2Cypher; cost scales with `WIKI_LIMIT`
  (the default 500-article build is well under a dollar with `gpt-4o-mini`).
- `add_graph_documents` issues a statement per node/edge; fine for the small
  per-article graphs here. For bulk *structured* loads, batched `UNWIND` is far
  faster — see demo 01's `prepare.py`.
- Generated Cypher is model output: it's validated read-only and time-bounded,
  but a wrong query yields an empty/odd answer rather than an error.
