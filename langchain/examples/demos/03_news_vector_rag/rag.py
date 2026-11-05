"""News vector RAG — query.

Five ways to use the AgensgraphVector store built by ingest.py:

  (a) vector semantic search          (HNSW)
  (b) metadata-filtered search        ($gte date, $in domain, $and)
  (c) hybrid search                   (vector + keyword, RRF fusion)
  (d) ef_search                       (hnsw.ef_search — pgvector's recall control)
  (e) RAG                             (as_retriever -> LCEL chain -> cited answer)

    cd langchain
    .venv/bin/python examples/demos/03_news_vector_rag/rag.py
    .venv/bin/python examples/demos/03_news_vector_rag/rag.py "your question"
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from _common import agens, config, console
from _common.models import get_embeddings, get_llm
from langchain_agensgraph import AgensGraph, AgensgraphVector
from langchain_agensgraph.vectorstores import SearchType

GRAPH = "news"
NODE_LABEL = "Article"
FILTER_KEYS = ("domain", "date")
DEFAULT_QUESTION = "How is artificial intelligence being used in business?"


def open_store(search_type: SearchType, graph_name: str = GRAPH, **kwargs: Any) -> AgensgraphVector:
    """The store ingest.py made, searched the given way; refused if it is not there.

    Each store object is one configuration over the same chunks: a vector one and a hybrid one
    read the same vertices and the same index.
    """
    return AgensgraphVector.from_existing_index(
        get_embeddings(),
        engine=agens.get_engine(),
        graph_name=graph_name,
        label=NODE_LABEL,
        search_type=search_type,
        filter_keys=FILTER_KEYS,
        **kwargs,
    )


def corpus(graph: AgensGraph) -> Tuple[Dict[str, Any], List[str]]:
    """The date range and the three largest domains, so the filters below use real values.

    A date the record did not carry is absent rather than "", so IS NOT NULL is the whole test.
    It matters for more than tidiness: "" sorts below every real date, so a corpus holding one
    would give `lo = ""` and a filter of `date >= lo` built from it would select everything.
    """
    dates = graph.query(
        'MATCH (n:"Article") WHERE n.date IS NOT NULL '
        "RETURN min(n.date::text) AS lo, max(n.date::text) AS hi"
    )[0]
    domains = [
        r["domain"]
        for r in graph.query(
            'MATCH (n:"Article") RETURN n.domain AS domain, count(*) AS c '
            "ORDER BY c DESC LIMIT 3"
        )
    ]
    return dates, domains


def show(hits: List[Tuple[Document, float]]) -> None:
    for doc, score in hits:
        m = doc.metadata
        print(f"  [{score:.3f}] {m.get('title', '')[:70]}")
        print(f"        {m.get('domain', '')}  {m.get('date', '')}")


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('domain', '?')} {d.metadata.get('date', '')}] "
        f"{d.metadata.get('title', '')}\n{d.page_content}"
        for d in docs
    )


def rag_chain(store: AgensgraphVector, llm) -> Runnable:
    """The store as a retriever, in an LCEL chain that answers with citations."""
    retriever = store.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the news snippets below. "
                   "Cite the source domains you rely on. If they don't cover it, say so."),
        ("human", "Question: {question}\n\nNews snippets:\n{context}"),
    ])
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    config.require_openai_key()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    vector = open_store(SearchType.VECTOR)
    graph = agens.make_graph(GRAPH)
    try:
        dates, domains = corpus(graph)
        console.section("corpus")
        print(f"  date range: {dates['lo']} .. {dates['hi']}   top domains: {', '.join(domains)}")

        # Relevance between 0 and 1; *_with_score would give the raw cosine distance.
        console.section("(a) vector semantic search")
        show(vector.similarity_search_with_relevance_scores(
            "technology and artificial intelligence", k=5))

        console.section("(b) metadata-filtered search (domain $in + date $gte)")
        flt = {"$and": [{"domain": {"$in": domains}}, {"date": {"$gte": dates["lo"]}}]}
        print(f"  filter = {flt}")
        show(vector.similarity_search_with_relevance_scores("business and markets", k=5,
                                                            filter=flt))

        console.section("(c) hybrid search (vector + keyword, RRF)")
        print("  keywords weighted twice the vector ranking, fusion_weights=(1.0, 2.0):")
        hybrid = open_store(SearchType.HYBRID, fusion_weights=(1.0, 2.0))
        for hit in hybrid.hybrid_search_with_details("stock market", k=5):
            m = hit.document.metadata
            print(f"  [{hit.fused_score:.4f}] vector rank {hit.vector_rank}, "
                  f"keyword rank {hit.keyword_rank}  {m.get('title', '')[:60]}")
        print("  and a filter narrows both halves:")
        show(hybrid.similarity_search_with_relevance_scores(
            "stock market", k=3, filter={"date": {"$gte": dates["lo"]}}))

        console.section("(d) ef_search (hnsw.ef_search — pgvector's recall control)")
        print("  ef_search is how many candidates the index considers before ranking;")
        print("  raising it trades latency for recall. Over-fetching k does not:")
        print("  a larger LIMIT returns the same rows, because the index already")
        print("  stopped looking.")
        for ef in (20, 200):
            tuned = open_store(SearchType.VECTOR, ef_search=ef)
            with console.timer(f"hnsw.ef_search={ef}"):
                hits = tuned.similarity_search_with_relevance_scores(
                    "sports", k=5, filter={"date": {"$gte": dates["lo"]}})
            show(hits)

        console.section(f"(e) RAG (as_retriever -> LCEL chain): {question!r}")
        with console.timer("RAG answer"):
            print("\n" + rag_chain(vector, get_llm()).invoke(question))
    finally:
        agens.close()


if __name__ == "__main__":
    main()
