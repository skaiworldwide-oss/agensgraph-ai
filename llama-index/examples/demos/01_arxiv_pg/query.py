"""arXiv property graph — query (four capabilities over ONE store).

Run after prepare.py. Demonstrates, against the same AgensPropertyGraphStore:

  (a) structured_query  — Cypher analytics (top authors, categories, years)
  (b) vector_query      — HNSW semantic search over Paper entities, with scores
  (c) get_rel_map       — expand the vector hits through the graph (shared authors)
  (d) GraphRAG          — PropertyGraphIndex.from_existing → grounded LLM answer

    cd llama-index
    .venv/bin/python examples/demos/01_arxiv_pg/query.py
    .venv/bin/python examples/demos/01_arxiv_pg/query.py "your question"

(Text2Cypher over a PropertyGraphStore is showcased in 02_wikipedia_pgindex.)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import VectorStoreQuery

from _common import agens, config, console
from _common.models import EMBED_DIM, get_embed_model, get_llm

GRAPH = "arxiv"
DEFAULT_QUESTION = "What are recent approaches to graph neural networks for molecular property prediction?"

# Cypher analytics. Drive these off the RELATIONSHIPS, not a `'Author' IN n.labels`
# predicate: every node lives on one "__Node__" label with its type in a jsonb
# `labels` list, so a label filter is a full scan + a jsonb membership test per
# node (slow at scale). The edge already implies the endpoint's type (AUTHORED_BY
# always points at an Author), so walking edges is much cheaper.
ANALYTICS = [
    ("most prolific authors", """
        MATCH (p:"__Node__")-[:"AUTHORED_BY"]->(a:"__Node__")
        RETURN a.name AS author, count(p) AS papers ORDER BY papers DESC LIMIT 10"""),
    ("largest categories", """
        MATCH (p:"__Node__")-[:"IN_CATEGORY"]->(c:"__Node__")
        RETURN c.name AS category, count(p) AS papers ORDER BY papers DESC LIMIT 10"""),
    ("papers per year", """
        MATCH (p:"__Node__") WHERE p.year IS NOT NULL
        RETURN p.year AS year, count(p) AS papers ORDER BY year DESC LIMIT 10"""),
]


def analytics(store) -> None:
    console.section("(a) graph analytics — structured_query (Cypher)")
    for label, q in ANALYTICS:
        console.sub(label)
        with console.timer(label):
            rows = store.structured_query(q)
        if rows:
            cols = list(rows[0].keys())
            console.table([[r[c] for c in cols] for r in rows], headers=cols)
        else:
            print("  (no rows)")


def vector_search(store, embed_model, question: str):
    console.section("(b) vector search — vector_query (HNSW over Paper entities)")
    qvec = embed_model.get_query_embedding(question)
    with console.timer("vector_query k=5"):
        nodes, scores = store.vector_query(
            VectorStoreQuery(query_embedding=qvec, similarity_top_k=5)
        )
    for n, s in zip(nodes, scores):
        title = (n.properties or {}).get("title", n.name)
        print(f"  {s:.3f}  [{n.name}] {title[:90]}")
    return nodes


def expand(store, nodes) -> None:
    console.section("(c) graph expansion — get_rel_map (depth=1)")
    if not nodes:
        print("  (no seed nodes)")
        return
    seeds = nodes[:3]
    print("  seeds: " + ", ".join(n.name for n in seeds))
    # depth=1 (each paper's authors + categories) is the fast path; depth>=2 uses
    # AgensGraph variable-length edges, which are far slower at scale.
    with console.timer("get_rel_map"):
        triplets = store.get_rel_map(seeds, depth=1, limit=30)
    for src, rel, tgt in triplets[:20]:
        sn = (src.properties or {}).get("title", src.name) if src.properties else src.name
        tn = (tgt.properties or {}).get("title", tgt.name) if tgt.properties else tgt.name
        print(f"  ({str(sn)[:40]}) -[{rel.label}]-> ({str(tn)[:40]})")


def graphrag(store, embed_model, llm, question: str) -> None:
    console.section("(d) GraphRAG — PropertyGraphIndex.from_existing → grounded answer")
    # kg_extractors=[] so attaching the index never triggers LLM re-extraction;
    # the store is already populated.
    index = PropertyGraphIndex.from_existing(
        property_graph_store=store,
        embed_model=embed_model,
        llm=llm,
        embed_kg_nodes=True,
        kg_extractors=[],
        use_async=False,  # keep retrieval on the sync pool (clean shutdown via close())
    )
    # VectorContextRetriever uses the store's vector_query (HNSW) then expands via
    # get_rel_map(path_depth) — i.e. exactly the (b)+(c) flow, packaged.
    retriever = index.as_retriever(
        sub_retrievers=[
            VectorContextRetriever(
                graph_store=store, embed_model=embed_model,
                # path_depth=1 (paper + its authors/categories) keeps the graph
                # context rich but cheap; depth=2 fans out expensively at 50k.
                similarity_top_k=5, path_depth=1, include_text=True,
            )
        ]
    )
    qe = RetrieverQueryEngine.from_args(retriever, llm=llm)
    print(f"  Q: {question}\n")
    with console.timer("graphrag answer"):
        resp = qe.query(question)
    print("  Answer:\n" + str(resp).strip())
    srcs = {s.node.ref_doc_id or s.node.node_id for s in (resp.source_nodes or [])}
    if srcs:
        print(f"\n  grounded on {len(resp.source_nodes)} retrieved context nodes")


def main() -> None:
    config.require_openai_key()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    embed_model = get_embed_model()
    llm = get_llm()
    # create=False: open the existing graph (don't rebuild). vector_dimension so
    # the store knows the HNSW index dimension for vector_query.
    store = agens.make_pg_store(GRAPH, vector_dimension=EMBED_DIM, create=False)
    try:
        analytics(store)
        hits = vector_search(store, embed_model, question)
        expand(store, hits)
        graphrag(store, embed_model, llm, question)
    finally:
        agens.close()


if __name__ == "__main__":
    main()
