"""arXiv GraphRAG — query.

Three ways to interrogate the graph and vector store built by ``prepare.py``:

  (a) graph analytics in Cypher        (prolific authors, categories, co-authors)
  (b) vector semantic search           (HNSW over the Paper chunks)
  (c) hybrid GraphRAG                  (vector retrieve → graph expand → LLM answer)

    cd langchain
    .venv/bin/python examples/demos/01_arxiv_graphrag/query.py
    .venv/bin/python examples/demos/01_arxiv_graphrag/query.py "your question here"
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agensgraph import Jsonb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from _common import agens, config, console
from _common.models import get_embeddings, get_llm
from langchain_agensgraph import AgensGraph, AgensgraphVector

GRAPH = "arxiv"
LABEL = "Paper"
DEFAULT_QUESTION = "What methods are used for studying black hole thermodynamics?"


# ── (a) graph analytics ──────────────────────────────────────────────────────

def graph_analytics(graph: AgensGraph) -> Dict[str, List[Dict[str, Any]]]:
    console.section("(a) graph analytics — Cypher over the knowledge graph")
    out: Dict[str, List[Dict[str, Any]]] = {}

    console.sub("most prolific authors")
    out["authors"] = rows = graph.query(
        'MATCH (a:"Author")<-[:"AUTHORED_BY"]-(p:"Paper") '
        "RETURN a.name AS author, count(*) AS papers "
        "ORDER BY papers DESC LIMIT 10"
    )
    console.table([(r["author"], r["papers"]) for r in rows], headers=["author", "papers"])

    console.sub("largest categories")
    out["categories"] = rows = graph.query(
        'MATCH (c:"Category")<-[:"IN_CATEGORY"]-(p:"Paper") '
        "RETURN c.name AS category, count(*) AS papers "
        "ORDER BY papers DESC LIMIT 10"
    )
    console.table([(r["category"], r["papers"]) for r in rows], headers=["category", "papers"])

    console.sub("top co-authorship pairs")
    out["pairs"] = rows = graph.query(
        'MATCH (a1:"Author")<-[:"AUTHORED_BY"]-(p:"Paper")-[:"AUTHORED_BY"]->(a2:"Author") '
        "WHERE a1.name < a2.name "
        "RETURN a1.name AS author_1, a2.name AS author_2, count(*) AS together "
        "ORDER BY together DESC LIMIT 10"
    )
    console.table(
        [(r["author_1"], r["author_2"], r["together"]) for r in rows],
        headers=["author 1", "author 2", "papers"],
    )

    console.sub("papers per year")
    out["years"] = rows = graph.query(
        'MATCH (p:"Paper")-[:"UPDATED_IN"]->(y:"Year") '
        "RETURN y.year AS year, count(*) AS papers ORDER BY year"
    )
    console.table([(r["year"], r["papers"]) for r in rows], headers=["year", "papers"])
    return out


# ── the store over the existing Paper chunks (nothing is embedded again) ─────

def get_store(embedding: Embeddings | None = None, graph_name: str = GRAPH) -> AgensgraphVector:
    """The vector store ``prepare.py`` made; refused if it is not there."""
    return AgensgraphVector.from_existing_index(
        embedding or get_embeddings(),
        engine=agens.get_engine(),
        graph_name=graph_name,
        label=LABEL,
    )


# ── (b) vector semantic search ───────────────────────────────────────────────

def vector_search(store: AgensgraphVector, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    console.section(f"(b) vector semantic search — {query!r}")
    with console.timer("similarity_search_with_relevance_scores") as t:
        hits = store.similarity_search_with_relevance_scores(query, k=k)
    print("  " + t.rate(k, "results"))
    for doc, relevance in hits:
        print(f"\n  [{relevance:.3f}] {doc.metadata.get('title', '')[:100]}")
        print(f"        arXiv:{doc.metadata.get('id')}  year:{doc.metadata.get('year')}")
    return hits


# ── (c) hybrid GraphRAG ──────────────────────────────────────────────────────

def graphrag(
    store: AgensgraphVector, graph: AgensGraph, llm, question: str, k: int = 5
) -> Dict[str, Any]:
    console.section(f"(c) hybrid GraphRAG — {question!r}")

    # 1) vector-retrieve seed papers
    seeds = store.similarity_search(question, k=k)
    seed_ids = [d.metadata["id"] for d in seeds]
    print(f"  seed papers (vector): {', '.join(seed_ids)}")

    # 2) graph-expand: related papers that share an author
    related = graph.query(
        "UNWIND %(ids)s AS pid "
        'MATCH (p:"Paper" {id: pid})-[:"AUTHORED_BY"]->(:"Author")<-[:"AUTHORED_BY"]-(rel:"Paper") '
        "WHERE rel.id <> pid "
        "RETURN DISTINCT rel.title AS title, rel.id AS id LIMIT 8",
        {"ids": Jsonb(seed_ids)},
    )
    print(f"  graph-expanded related papers: {len(related)}")

    # 3) assemble grounded context and ask the LLM
    context = "\n\n".join(f"- {d.page_content}" for d in seeds)
    if related:
        context += "\n\nRelated work (same authors):\n" + "\n".join(
            f"- {r['title']}" for r in related
        )
    prompt = (
        "You are a research assistant. Using ONLY the arXiv abstracts below, "
        "answer the question concisely and cite paper titles you rely on. "
        "If the abstracts don't cover it, say so.\n\n"
        f"Question: {question}\n\nAbstracts:\n{context}\n\nAnswer:"
    )
    with console.timer("LLM answer"):
        answer = llm.invoke(prompt).text
    print(f"\n{answer}\n")
    return {"seeds": seed_ids, "related": related, "answer": answer}


def main() -> None:
    config.require_openai_key()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    graph = agens.make_graph(GRAPH)
    try:
        graph_analytics(graph)
        store = get_store()
        vector_search(store, "quantum entanglement and information theory")
        graphrag(store, graph, get_llm(), question)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
