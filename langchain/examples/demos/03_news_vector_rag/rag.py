"""News vector RAG — query.

Five ways to use the AgensgraphVector store built by ingest.py:

  (a) vector semantic search          (HNSW)
  (b) metadata-filtered search        ($gte date, $in domain, $and)
  (c) hybrid search                   (vector + keyword, RRF fusion)
  (d) effective_search_ratio          (over-fetch for recall under a filter)
  (e) RAG                             (as_retriever -> LCEL chain -> cited answer)

    cd langchain
    .venv/bin/python examples/demos/03_news_vector_rag/rag.py
    .venv/bin/python examples/demos/03_news_vector_rag/rag.py "your question"
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.vectorstores.agensgraph_vector import HybridSearchConfig, SearchType

from _common import agens, config, console
from _common.models import get_embeddings, get_llm

GRAPH = "news"
NODE_LABEL = "Article"
DEFAULT_QUESTION = "How is artificial intelligence being used in business?"


def _vec(search_type, keyword=None):
    return AgensgraphVector.from_existing_index(
        embedding=get_embeddings(),
        index_name="vector",
        search_type=search_type,
        keyword_index_name=keyword,
        node_label=NODE_LABEL,
        graph_name=GRAPH,
        engine=agens.get_engine(),
    )


def _show(hits):
    for doc, score in hits:
        m = doc.metadata
        print(f"  [{score:.3f}] {m.get('title','')[:70]}")
        print(f"        {m.get('domain','')}  {m.get('date','')}")


def main() -> None:
    config.require_openai_key()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    vector = _vec(SearchType.VECTOR)
    hybrid = _vec(SearchType.HYBRID, keyword="keyword")
    graph = agens.make_graph(GRAPH, create=False, refresh_schema=False)
    try:
        # corpus stats so the filters below use real values
        dates = graph.query(
            'MATCH (n:"Article") WHERE n.date IS NOT NULL '
            "RETURN min(n.date) AS lo, max(n.date) AS hi"
        )[0]
        domains = [r["domain"] for r in graph.query(
            'MATCH (n:"Article") RETURN n.domain AS domain, count(*) AS c ORDER BY c DESC LIMIT 3'
        )]
        console.section("corpus")
        print(f"  date range: {dates['lo']} .. {dates['hi']}   top domains: {', '.join(domains)}")

        console.section("(a) vector semantic search")
        _show(vector.similarity_search_with_score("technology and artificial intelligence", k=5))

        console.section("(b) metadata-filtered search (domain $in + date $gte)")
        flt = {"$and": [{"domain": {"$in": domains}}, {"date": {"$gte": dates["lo"]}}]}
        print(f"  filter = {flt}")
        _show(vector.similarity_search_with_score("business and markets", k=5, filter=flt))

        console.section("(c) hybrid search (vector + keyword, RRF)")
        print("  weighting keywords higher with HybridSearchConfig(keyword_weight=2.0):")
        hits = hybrid.similarity_search_with_score(
            "stock market", k=5, hybrid_config=HybridSearchConfig(keyword_weight=2.0)
        )
        _show(hits)

        console.section("(d) effective_search_ratio (over-fetch for recall under a filter)")
        _show(vector.similarity_search_with_score(
            "sports", k=5, filter={"date": {"$gte": dates["lo"]}}, effective_search_ratio=4.0
        ))

        # (e) RAG: AgensgraphVector.as_retriever plugged into an LCEL chain
        console.section(f"(e) RAG (as_retriever -> LCEL chain): {question!r}")
        retriever = vector.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(
                f"[{d.metadata.get('domain','?')} {d.metadata.get('date','')}] "
                f"{d.metadata.get('title','')}\n{d.page_content}"
                for d in docs
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the news snippets below. "
                       "Cite the source domains you rely on. If they don't cover it, say so."),
            ("human", "Question: {question}\n\nNews snippets:\n{context}"),
        ])
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | get_llm()
            | StrOutputParser()
        )
        with console.timer("RAG answer"):
            print("\n" + chain.invoke(question))
    finally:
        agens.close()


if __name__ == "__main__":
    main()
