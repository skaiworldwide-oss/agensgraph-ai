# 02 · Wikipedia knowledge graph + Text2Cypher

The **LangChain-forward** demo. An LLM builds a knowledge graph from unstructured
Wikipedia text, and natural-language questions are answered by generating Cypher
— both through LangChain idioms.

```
(:Person|Organization|Location|Event|Concept|Work|…)-[:LOCATED_IN|PART_OF|…]->(…)
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

- **A fixed relationship vocabulary.** `build_kg.py` constrains both ends of the
  extraction: `allowed_nodes` and `allowed_relationships`. Constraining only the nodes
  is a trap — the model then names each relationship after the sentence it came from,
  and a 500-article build produced **1,626 relationship types for 11,888 edges, 951 of
  them holding a single edge** (`HAS_NOT_DEVELOPED`, `QUESTIONS_AUTHORESHIP`, …). That
  costs twice: every type is a table with its own indexes, so the graph took 203 MB to
  hold a few MB of facts, and it cannot be queried, because nobody can write Cypher
  against a vocabulary they cannot enumerate. `strict_mode` (on by default) drops
  anything outside either list.
- **`build_kg.py`** — `LLMGraphTransformer` (LangChain): the LLM extracts typed
  entities and relationships from each article as **structured output**, returning
  `GraphDocument`s that drop straight into `AgensGraph.add_graph_documents(...,
  include_source=True)` (which also records the source article + `MENTIONS`
  edges). Extraction is run with bounded concurrency via `aconvert_to_graph_documents`.
- **`ask.py`** — **`AgensCypherQAChain`**, the library's Text2Cypher chain:

  ```python
  chain = AgensCypherQAChain.from_llm(llm, graph=graph, return_intermediate_steps=True)
  out = chain.invoke({"query": "Which people are mentioned most often?"})
  out["intermediate_steps"]   # [{"query": "MATCH ..."}, {"context": [...]}]
  ```

  The chain carries the parts that are easy to get wrong: the AgensGraph dialect
  rules in the prompt, a repair pass for identifiers the model leaves unquoted,
  refusal of generated writes, an `EXPLAIN` check so a malformed query never runs,
  and a statement timeout.

  The graph's `get_schema` (with `enhanced_schema=True`, so it carries example
  property values) goes into the prompt, and the same schema decides what the repair
  pass may quote — AgensGraph folds unquoted identifiers, and folding runs both ways,
  so only the schema knows whether `n.firstName` or `n."firstName"` is the right
  spelling for your data.

## Notes

- Uses OpenAI for both extraction and Text2Cypher; cost scales with `WIKI_LIMIT`
  (the default 500-article build is well under a dollar with `gpt-4o-mini`).
- `add_graph_documents` issues a statement per node/edge; fine for the small
  per-article graphs here. For bulk *structured* loads, batched `UNWIND` is far
  faster — see demo 01's `prepare.py`.
- Generated Cypher is model output: it's validated read-only and time-bounded,
  but a wrong query yields an empty/odd answer rather than an error.
