"""Wikipedia knowledge graph — build.

Uses LLMGraphTransformer to extract a knowledge graph from Wikipedia article leads, then writes it
to AgensGraph with add_graph_documents:

    (:Person|Organization|Location|...)-[:LOCATED_IN|PART_OF|...]->(...)
    (:Document {title,url,text})-[:MENTIONS]->(entity)     # provenance (include_source)

The model does the entity and relationship extraction (structured output), and the same
GraphDocument objects load straight into AgensGraph. With ``DEMOS_FAKE=1`` the articles are the
50 short leads of ``_common/fixtures/wikipedia.jsonl``, written for the fixture, and the fake model
takes their capitalized phrases as entities.

    cd langchain
    .venv/bin/python examples/demos/02_wikipedia_kg/build_kg.py
    WIKI_LIMIT=30 WIKI_RESET=1 .venv/bin/python examples/demos/02_wikipedia_kg/build_kg.py   # quick
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Iterable, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from _common import agens, config, console
from _common.datautil import batched, env_int, fixture, stream_hf
from _common.models import get_llm
from langchain_agensgraph import AgensGraph, GraphDocument, LLMGraphTransformer

GRAPH = "wikipedia_kg"
DATASET = ("wikimedia/wikipedia", "20231101.en")

# A curated-but-generous label set: clean enough for Text2Cypher, broad enough
# that important entities aren't dropped by strict_mode.
ALLOWED_NODES = [
    "Person", "Organization", "Location", "Event",
    "Concept", "Work", "Field", "Group", "Technology", "Award",
]

# Left unconstrained, the model names a relationship after the sentence it came from,
# so an encyclopaedia yields a type per fact: HAS_NOT_DEVELOPED, QUESTIONS_AUTHORESHIP,
# INITIATED_INDUSTRIAL_PRODUCTION_OF. A 500-article build produced 1,626 types for 11,888
# edges, 951 of them holding a single edge. That is expensive twice over — every type is
# a table with its own indexes, so the graph cost 203MB to hold a few MB of facts — and
# it is useless to query, because no one can write Cypher against a vocabulary they
# cannot enumerate. A fixed vocabulary keeps the graph answerable.
ALLOWED_RELATIONSHIPS = [
    "LOCATED_IN", "PART_OF", "MEMBER_OF", "FOUNDED", "WORKED_FOR",
    "CREATED", "PARTICIPATED_IN", "INFLUENCED", "RELATED_TO", "IS_A",
    "USES", "DEVELOPED", "RECEIVED", "SUCCEEDED", "BORN_IN",
    "DIED_IN", "OCCURRED_IN", "STUDIED",
]


def documents(limit: int, chars: int) -> Iterable[Document]:
    """Each article's lead as a Document: the fixture with DEMOS_FAKE=1, else the stream."""
    source = (
        fixture("wikipedia", limit)
        if config.fake()
        else stream_hf(DATASET[0], config=DATASET[1], limit=limit)
    )
    for rec in source:
        text = (rec.get("text") or "")[:chars].strip()
        if len(text) < 200:
            continue
        yield Document(
            page_content=text,
            metadata={"title": rec["title"], "url": rec.get("url", ""), "source": "wikipedia"},
        )


def transformer(llm, concurrency: int = 8) -> LLMGraphTransformer:
    """The extractor: a fixed vocabulary, and a failed article left out rather than raised."""
    return LLMGraphTransformer(
        llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=False,  # strict_mode drops anything outside either list
        max_concurrency=concurrency,
        on_error="skip",
    )


def extract(
    extractor: LLMGraphTransformer, docs: Sequence[Document], chunk: int = 50
) -> Tuple[List[GraphDocument], int]:
    """Graph documents for the articles, and how many were skipped.

    ``on_error="skip"`` logs and leaves out an article that fails (an entity-dense one that
    exceeds the model's output limit, say) instead of aborting the batch.
    """
    out: List[GraphDocument] = []
    done = 0
    for part in batched(docs, chunk):
        out.extend(extractor.convert_to_graph_documents(part))
        done += len(part)
        print(f"    ... processed {done} articles ({done - len(out)} skipped)")
    return out, done - len(out)


def load(graph: AgensGraph, graph_docs: Sequence[GraphDocument]) -> None:
    """Write the graph documents, each source article as a Document with MENTIONS edges."""
    graph.add_graph_documents(graph_docs, include_source=True)


def main() -> None:
    limit = env_int("WIKI_LIMIT", 500)
    chars = env_int("WIKI_CHARS", 1800)
    concurrency = env_int("WIKI_CONCURRENCY", 8)
    config.require_openai_key()

    console.section(f"Wikipedia KG — build  (WIKI_LIMIT={limit:,}, lead={chars} chars)")
    graph = agens.make_graph(GRAPH, create=True)
    try:
        if os.getenv("WIKI_RESET"):
            graph.drop_graph()
            print(f"[reset] dropped graph {GRAPH!r}")

        console.sub("LLM extraction (LLMGraphTransformer, structured output)")
        docs = list(documents(limit, chars))
        with console.timer("extraction") as t:
            graph_docs, failed = extract(transformer(get_llm(), concurrency), docs)
        n_nodes = sum(len(g.nodes) for g in graph_docs)
        n_rels = sum(len(g.relationships) for g in graph_docs)
        print(f"  {len(graph_docs)} articles -> {n_nodes:,} nodes, {n_rels:,} relationships "
              f"({len(docs) / max(t.seconds, 1e-9):.1f} articles/s)"
              + (f"; {failed} skipped (extraction error)" if failed else ""))

        console.sub("load into AgensGraph (add_graph_documents, include_source=True)")
        with console.timer("add_graph_documents"):
            load(graph, graph_docs)

        total_nodes = graph.query("MATCH (n) RETURN count(*) AS c")[0]["c"]
        total_edges = graph.query("MATCH ()-[r]->() RETURN count(*) AS c")[0]["c"]
        labels = graph.query(
            "MATCH (n) RETURN label(n) AS label, count(*) AS n ORDER BY n DESC LIMIT 12"
        )
        console.sub("graph")
        console.table([(r["label"], r["n"]) for r in labels], headers=["label", "count"])
        print(f"\n  total: {total_nodes:,} nodes, {total_edges:,} edges")
        print("\nNext:  .venv/bin/python examples/demos/02_wikipedia_kg/ask.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
