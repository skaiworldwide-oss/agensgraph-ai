"""Wikipedia knowledge graph — build.

Uses LangChain's LLMGraphTransformer to extract a knowledge graph from Wikipedia
article leads, then writes it to AgensGraph with add_graph_documents:

    (:Person|Organization|Location|...)-[:<LLM-named relationship>]->(...)
    (:Document {title,url})-[:MENTIONS]->(entity)     # provenance (include_source)

This is the LangChain-native graph-construction path: the LLM does the entity /
relationship extraction (structured output), and the same GraphDocument objects
load straight into AgensGraph.

    cd langchain
    .venv/bin/python examples/demos/02_wikipedia_kg/build_kg.py
    WIKI_LIMIT=30 WIKI_RESET=1 .venv/bin/python examples/demos/02_wikipedia_kg/build_kg.py   # quick
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from langchain_core.documents import Document

from langchain_agensgraph import LLMGraphTransformer

from _common import agens, config, console
from _common.datautil import batched, env_int, stream_hf
from _common.models import get_llm

GRAPH = "wikipedia_kg"
DATASET = ("wikimedia/wikipedia", "20231101.en")

# A curated-but-generous label set: clean enough for Text2Cypher, broad enough
# that important entities aren't dropped by strict_mode.
ALLOWED_NODES = [
    "Person", "Organization", "Location", "Event",
    "Concept", "Work", "Field", "Group", "Technology", "Award",
]


def _docs(limit: int, chars: int):
    """Stream Wikipedia and yield LangChain Documents from each article's lead."""
    for rec in stream_hf(DATASET[0], config=DATASET[1], limit=limit):
        text = (rec.get("text") or "")[:chars].strip()
        if len(text) < 200:
            continue
        yield Document(
            page_content=text,
            metadata={"title": rec["title"], "url": rec.get("url", ""), "source": "wikipedia"},
        )


async def _extract(transformer, docs, concurrency):
    """Bounded-concurrency extraction with per-document error isolation.

    Uses aprocess_response per doc under gather(return_exceptions=True) instead
    of aconvert_to_graph_documents, so a single article that errors (e.g. an
    entity-dense one that exceeds the LLM output-token limit -> OpenAI
    LengthFinishReasonError) is skipped rather than aborting the whole batch.
    See FINDINGS F-007.
    """
    out, failed, done = [], 0, 0
    for chunk in batched(docs, concurrency):
        results = await asyncio.gather(
            *(transformer.aprocess_response(d) for d in chunk),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                failed += 1
            else:
                out.append(r)
        done += len(chunk)
        print(f"    ... processed {done} articles ({failed} skipped)")
    return out, failed


def _reset(conf) -> None:
    with psycopg.connect(**conf, autocommit=True) as conn:
        conn.execute('DROP GRAPH IF EXISTS "%s" CASCADE' % GRAPH)
    print(f"[reset] dropped graph {GRAPH!r}")


def main() -> None:
    limit = env_int("WIKI_LIMIT", 500)
    chars = env_int("WIKI_CHARS", 1800)
    concurrency = env_int("WIKI_CONCURRENCY", 8)
    config.require_openai_key()

    console.section(f"Wikipedia KG — build  (WIKI_LIMIT={limit:,}, lead={chars} chars)")

    if __import__("os").getenv("WIKI_RESET"):
        _reset(config.conf())

    transformer = LLMGraphTransformer(
        get_llm(),
        allowed_nodes=ALLOWED_NODES,
        node_properties=False,
    )

    console.sub("LLM extraction (LLMGraphTransformer, structured output)")
    docs = list(_docs(limit, chars))
    with console.timer("extraction") as t:
        graph_docs, failed = asyncio.run(_extract(transformer, docs, concurrency))
    n_nodes = sum(len(g.nodes) for g in graph_docs)
    n_rels = sum(len(g.relationships) for g in graph_docs)
    print(f"  {len(graph_docs)} articles -> {n_nodes:,} nodes, {n_rels:,} relationships "
          f"({len(docs) / t.seconds:.1f} articles/s)"
          + (f"; {failed} skipped (extraction error)" if failed else ""))

    console.sub("load into AgensGraph (add_graph_documents, include_source=True)")
    graph = agens.make_graph(GRAPH, create=True, refresh_schema=False)
    try:
        with console.timer("add_graph_documents") as t:
            graph.add_graph_documents(graph_docs, include_source=True)

        total_nodes = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        total_edges = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
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
