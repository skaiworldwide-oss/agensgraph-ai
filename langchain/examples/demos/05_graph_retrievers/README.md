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

# the same retrievers on 138k real vertices (needs demo 01's arxiv graph):
.venv/bin/python examples/demos/05_graph_retrievers/arxiv.py
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

## At scale

- **`arxiv.py`** — the identical retrievers against demo 01's `arxiv` graph
  (50k papers, 88k authors, 275k edges): an abstract never names its authors,
  so the vector answer cannot and the 1-hop context answer can. The aggregate
  question also shows `retry_on_empty` recovering from a relationship written
  against its schema direction — a query that runs clean and matches nothing.
- **`bench/retriever_bench_arxiv.py`** — the module's numbers on that graph,
  with no embedding API in the loop (query vectors are sampled from the store):
  one-statement context vs one-query-per-seed, plan-shape proof, concurrency,
  and the cautionary unfiltered two-hop walk through the Category/Year hubs
  that a `relationship_type` filter exists to avoid.
