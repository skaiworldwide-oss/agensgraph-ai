"""News vector RAG — query (semantic, filtered, hybrid, cited).

Run after ingest.py. Demonstrates, over the AgensgraphVectorStore behind a
LlamaIndex VectorStoreIndex:

  (a) plain semantic search        — VectorIndexRetriever
  (b) metadata-filtered retrieval  — MetadataFilters (IN domain, GTE date, AND)
  (c) hybrid RRF                   — a separate hybrid_search store instance
  (d) cited RAG                    — CitationQueryEngine with [N] source markers

    cd llama-index
    .venv/bin/python examples/demos/03_news_vector_rag/rag.py
    .venv/bin/python examples/demos/03_news_vector_rag/rag.py "your question"
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from _common import agens, config, console
from _common.models import configure_settings, get_embed_model

GRAPH = "news"
NODE_LABEL = "Article"
DEFAULT_QUESTION = "What is happening with artificial intelligence in business?"


def _meta(node) -> str:
    m = node.metadata or {}
    return f"{m.get('domain','?')} · {m.get('date','?')} · {(m.get('title') or '')[:60]}"


def top_domains(store, k: int = 4) -> list[str]:
    rows = store.database_query(
        f'MATCH (n:"{NODE_LABEL}") WHERE n.domain IS NOT NULL '
        "RETURN n.domain AS domain, count(*) AS c ORDER BY c DESC LIMIT %(k)s",
        {"k": k},
    )
    return [r["domain"] for r in rows]


def semantic(index, question: str) -> None:
    console.section("(a) plain semantic search")
    retriever = index.as_retriever(similarity_top_k=5)
    with console.timer("retrieve k=5"):
        hits = retriever.retrieve(question)
    for h in hits:
        print(f"  {h.score:.3f}  {_meta(h.node)}")


def filtered(index, store, question: str) -> None:
    console.section("(b) metadata-filtered retrieval (IN domain AND GTE date)")
    domains = top_domains(store)
    print(f"  filtering to domains={domains} and date >= 2017-01-01")
    filters = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
            MetadataFilter(key="domain", operator=FilterOperator.IN, value=domains),
            MetadataFilter(key="date", operator=FilterOperator.GTE, value="2017-01-01"),
        ],
    )
    retriever = index.as_retriever(similarity_top_k=5, filters=filters)
    with console.timer("filtered retrieve"):
        hits = retriever.retrieve(question)
    for h in hits:
        print(f"  {h.score:.3f}  {_meta(h.node)}")
    assert all((h.node.metadata or {}).get("domain") in domains for h in hits), \
        "filter leaked a non-matching domain"
    print("  ✓ all hits respect the filter")


def hybrid(question: str) -> None:
    console.section("(c) hybrid search (RRF: vector + keyword)")
    # hybrid is incompatible with metadata filters, so it gets its OWN store
    # instance (same graph/label/data; it additionally builds the FTS index).
    hstore = agens.make_vector_store(graph_name=GRAPH, node_label=NODE_LABEL, hybrid_search=True)
    hindex = VectorStoreIndex.from_vector_store(hstore, embed_model=get_embed_model())
    retriever = hindex.as_retriever(similarity_top_k=5, vector_store_query_mode="hybrid")
    with console.timer("hybrid retrieve"):
        hits = retriever.retrieve(question)
    for h in hits:
        print(f"  {h.score:.3f}  {_meta(h.node)}")


def cited_rag(index, question: str) -> None:
    console.section("(d) cited RAG — CitationQueryEngine")
    qe = CitationQueryEngine.from_args(index, similarity_top_k=5)
    print(f"  Q: {question}\n")
    with console.timer("cited answer"):
        resp = qe.query(question)
    print("  Answer:\n" + str(resp).strip())
    print("\n  Sources:")
    for i, s in enumerate(resp.source_nodes, 1):
        print(f"   [{i}] {_meta(s.node)}")


def main() -> None:
    config.require_openai_key()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    configure_settings()  # Settings.llm / Settings.embed_model for the query engines
    store = agens.make_vector_store(graph_name=GRAPH, node_label=NODE_LABEL)
    index = VectorStoreIndex.from_vector_store(store, embed_model=get_embed_model())
    try:
        semantic(index, question)
        filtered(index, store, question)
        hybrid(question)
        cited_rag(index, question)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
