# 05 · Graph retrievers (vector vs. graph context vs. text2cypher)

The **retriever family** showcase, over a catalog where every film, person and
plot is invented — the model answering has nothing to lean on but what each
retriever hands it, so the differences between them are the retrieval, not the
model's memory.

## Run

```bash
cd langchain
.venv/bin/python examples/demos/05_graph_retrievers/ingest.py    # 8 films, people, genres
.venv/bin/python examples/demos/05_graph_retrievers/retrieve.py  # the three retrievers, side by side
```

## What it demonstrates

- **`ingest.py`** — the movies are the vector store's own nodes (label `Movie`,
  plot text embedded, the title as `__id__`), with directors, actors and genres
  as ordinary graph structure around them.
- **`retrieve.py`** — the same questions through three retrievers:
  1. **`AgensVectorRetriever`** finds the right film by its plot — and cannot
     name its director, because the plot never says.
  2. **`AgensGraphContextRetriever`** runs the identical search with the film's
     2-hop neighbourhood fetched **in the same statement**; the rendered
     context names the director, and a director question routed through a
     shared director ("which other film was made by...") resolves across two
     hops.
  3. **`AgensText2CypherRetriever`** answers the aggregate ("how many films did
     X direct?") by writing the Cypher itself — a similarity search has no
     notion of *count*.

Each answer prints with its retrieval wall-clock, and the text2cypher answer
prints the Cypher it generated.
