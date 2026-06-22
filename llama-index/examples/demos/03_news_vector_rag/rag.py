"""News vector RAG — query (semantic, filtered, hybrid, cited).

Run after ingest.py. Demonstrates, over the AgensgraphVectorStore behind a
LlamaIndex VectorStoreIndex:

  (a) plain semantic search        — VectorIndexRetriever
  (b) metadata-filtered retrieval  — MetadataFilters (IN domain, GTE date, AND)
  (b2) richer operators            — NIN / CONTAINS / TEXT_MATCH / OR / nested groups
  (c) hybrid RRF                   — a separate hybrid_search store instance
  (d) cited RAG                    — CitationQueryEngine with [N] source markers
  (e) store lifecycle              — add / get_nodes / delete_nodes / delete / clear

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
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from _common import agens, config, console
from _common.models import EMBED_DIM, configure_settings, get_embed_model

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


def filtered_advanced(index, store, question: str) -> None:
    console.section("(b2) richer operators — NIN / CONTAINS / TEXT_MATCH / OR / nested groups")
    domains = top_domains(store, 5)
    # The domain of the current top semantic hit — excluding it below makes the
    # second result set visibly different from the first.
    top = index.as_retriever(similarity_top_k=1).retrieve(question)
    top_domain = (top[0].node.metadata or {}).get("domain") if top else domains[0]
    # "Anything NOT from the rarest of the top domains" — a NIN with a nested OR
    # over date whose two halves span every date, so the date clause is always
    # satisfied (it's here to show boolean composition + nested groups). The top
    # hit survives this one.
    f_nin = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
            MetadataFilter(key="domain", operator=FilterOperator.NIN, value=domains[-1:]),
            MetadataFilters(
                condition=FilterCondition.OR,
                filters=[
                    MetadataFilter(key="date", operator=FilterOperator.GTE, value="2017-01-01"),
                    MetadataFilter(key="date", operator=FilterOperator.LT, value="2017-01-01"),
                ],
            ),
        ],
    )
    # Exclude the top-hit's domain and require a url substring — a deliberately
    # different result set (those hits disappear). CONTAINS / TEXT_MATCH are
    # case-sensitive substring matches on a string property.
    f_text = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
            MetadataFilter(key="domain", operator=FilterOperator.NIN, value=[top_domain]),
            MetadataFilter(key="url", operator=FilterOperator.CONTAINS, value="http"),
        ],
    )
    for label, filters in [
        (f"NIN domain != {domains[-1]} AND nested-OR(date) — keeps the top hit", f_nin),
        (f"NIN domain != {top_domain} AND url CONTAINS 'http' — drops it", f_text),
    ]:
        console.sub(label)
        hits = index.as_retriever(similarity_top_k=5, filters=filters).retrieve(question)
        print(f"  {len(hits)} hit(s)")
        for h in hits[:3]:
            print(f"  {h.score:.3f}  {_meta(h.node)}")
    print("  (supported: EQ/NE/GT/GTE/LT/LTE/IN/NIN/CONTAINS/TEXT_MATCH/ANY/ALL/IS_EMPTY)")


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


def vector_crud() -> None:
    console.section("(e) store lifecycle — add / get_nodes / delete_nodes / delete / clear")
    # A scratch graph + label so the populated `news` graph is untouched. The
    # embeddings are constant placeholders: get_nodes / delete / clear don't run
    # vector search, so there are no OpenAI calls in this section.
    store = agens.make_vector_store(graph_name="news_crud_demo", node_label="Doc")
    count = lambda: store.database_query('MATCH (n:"Doc") RETURN count(*) AS c')[0]["c"]

    def node(nid: str, doc: str, domain: str) -> TextNode:
        n = TextNode(id_=nid, text=f"scratch chunk {nid}", embedding=[0.1] * EMBED_DIM,
                     metadata={"domain": domain})
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc)
        return n

    store.clear()  # idempotent reset
    try:
        store.add([node("c1", "docA", "a.com"), node("c2", "docA", "a.com"),
                   node("c3", "docB", "b.com"), node("c4", "docB", "b.com")])
        console.sub("added 4 nodes across 2 source docs")
        print(f"  total nodes: {count()}")

        a_com = MetadataFilters(filters=[MetadataFilter(key="domain", operator=FilterOperator.EQ, value="a.com")])
        console.sub("get_nodes(filters: domain == a.com)")
        print("  ids: " + ", ".join(n.node_id for n in store.get_nodes(filters=a_com)))
        console.sub("get_nodes(node_ids=['c3'])")
        print("  ids: " + ", ".join(n.node_id for n in store.get_nodes(node_ids=["c3"])))

        b_com = MetadataFilters(filters=[MetadataFilter(key="domain", operator=FilterOperator.EQ, value="b.com")])
        console.sub("delete_nodes(filters: domain == b.com)")
        store.delete_nodes(filters=b_com)
        print(f"  total nodes: {count()}")
        console.sub("delete(ref_doc_id='docA') — delete by source document")
        store.delete(ref_doc_id="docA")
        print(f"  total nodes: {count()}")
        console.sub("clear() — drop everything for this label")
        store.clear()
        print(f"  total nodes: {count()}")
        print("\n  ✓ add → get_nodes → delete_nodes → delete → clear verified (news untouched)")
    finally:
        store.clear()


def main() -> None:
    config.require_openai_key()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    configure_settings()  # Settings.llm / Settings.embed_model for the query engines
    store = agens.make_vector_store(graph_name=GRAPH, node_label=NODE_LABEL)
    index = VectorStoreIndex.from_vector_store(store, embed_model=get_embed_model())
    try:
        semantic(index, question)
        filtered(index, store, question)
        filtered_advanced(index, store, question)
        hybrid(question)
        cited_rag(index, question)
        vector_crud()
    finally:
        agens.close()


if __name__ == "__main__":
    main()
