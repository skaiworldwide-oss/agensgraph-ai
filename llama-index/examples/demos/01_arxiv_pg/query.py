"""arXiv property graph — query (four capabilities over ONE store).

Run after prepare.py. Demonstrates, against the same AgensPropertyGraphStore:

  (a) structured_query  — Cypher analytics (top authors, categories, years)
  (b) vector_query      — HNSW semantic search over Paper entities, with scores
  (c) get_rel_map       — expand the vector hits through the graph (shared authors)
  (d) GraphRAG          — PropertyGraphIndex.from_existing → grounded LLM answer
  (e) get / get_triplets — fetch nodes by id/property and the triplets around them
  (f) mutation lifecycle — upsert → get → delete on a throwaway scratch graph

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
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore

from _common import agens, config, console
from _common.models import EMBED_DIM, get_embed_model, get_llm

GRAPH = "arxiv"
DEFAULT_QUESTION = "What are recent approaches to graph neural networks for molecular property prediction?"

# Cypher analytics — two rules that keep these fast at scale on this store:
#  1. Use count(*), not count(p)/count(n): counting a node variable materializes
#     each node's full properties (including its embedding), which is much slower.
#  2. Drive aggregations off the EDGES (top authors / categories): the edge implies
#     the endpoint type, so no node-type filter is needed and the endpoints carry
#     no embedding to read.
# A type-scoped node filter `WHERE n.__type__ = 'X'` (indexed) beats
# `'X' IN n.labels` (a jsonb scan); papers-per-year reads `year` from each Paper,
# so it filters on `year` directly.
ANALYTICS = [
    ("most prolific authors", """
        MATCH (p:"__Node__")-[:"AUTHORED_BY"]->(a:"__Node__")
        RETURN a.name AS author, count(*) AS papers ORDER BY papers DESC LIMIT 10"""),
    ("largest categories", """
        MATCH (p:"__Node__")-[:"IN_CATEGORY"]->(c:"__Node__")
        RETURN c.name AS category, count(*) AS papers ORDER BY papers DESC LIMIT 10"""),
    ("papers per year", """
        MATCH (p:"__Node__") WHERE p.year IS NOT NULL
        RETURN p.year AS year, count(*) AS papers ORDER BY year DESC LIMIT 10"""),
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


def read_back(store, hits) -> None:
    console.section("(e) read by id / triplets — get + get_triplets (read-only)")
    if not hits:
        print("  (no seed nodes)")
        return
    ids = [n.name for n in hits[:3]]
    console.sub("get(ids=...) — fetch specific Paper nodes by id")
    for n in store.get(ids=ids):
        title = (n.properties or {}).get("title", n.name)
        print(f"  [{n.name}] {str(title)[:80]}")
    console.sub("get_triplets(entity_names=...) — relations leaving the top paper")
    triplets = store.get_triplets(entity_names=ids[:1])
    for src, rel, tgt in triplets[:12]:
        print(f"  ({src.name}) -[{rel.label}]-> ({tgt.name})")
    if not triplets:
        print("  (no outgoing triplets for this paper)")


def crud_lifecycle() -> None:
    console.section("(f) mutation lifecycle — upsert / get / get_triplets / delete")
    # Everything here runs on a SEPARATE throwaway graph so the populated `arxiv`
    # graph is never mutated. Its engine is built with from_conf (the dict-based
    # constructor) — the shared demo pool uses from_url; both reach the same DB.
    engine = AgensEngine.from_conf(config.conf())
    names = lambda nodes: ", ".join(n.name for n in nodes) or "(none)"
    try:
        store = AgensPropertyGraphStore(
            "crud_demo",  # a plain scratch graph (AgensGraph reserves the pg_ prefix)
            conf=config.conf(),
            vector_dimension=EMBED_DIM,
            create=True,
            create_indexes=True,
            refresh_schema=False,
            engine=engine,
        )
        store.structured_query('MATCH (n:"__Node__") DETACH DELETE n')  # idempotent reset
        store.upsert_nodes([
            EntityNode(name="demo:p1", label="Paper", properties={"title": "Graphs for X", "year": 2024}),
            EntityNode(name="demo:p2", label="Paper", properties={"title": "Graphs for Y", "year": 2025}),
            EntityNode(name="demo:alice", label="Author"),
        ])
        store.upsert_relations([
            Relation(label="AUTHORED_BY", source_id="demo:p1", target_id="demo:alice"),
            Relation(label="AUTHORED_BY", source_id="demo:p2", target_id="demo:alice"),
        ])
        console.sub("after upsert — get(ids=...)")
        print("  " + names(store.get(ids=["demo:p1", "demo:p2", "demo:alice"])))
        console.sub("get(properties={'year': 2025})")
        print("  " + names(store.get(properties={"year": 2025})))
        console.sub("get_triplets(entity_names=['demo:p1', 'demo:p2'])")
        for src, rel, tgt in store.get_triplets(entity_names=["demo:p1", "demo:p2"]):
            print(f"  ({src.name}) -[{rel.label}]-> ({tgt.name})")
        console.sub("delete(ids=['demo:p1']) then delete(entity_names=['demo:alice'])")
        store.delete(ids=["demo:p1"])
        store.delete(entity_names=["demo:alice"])
        print("  remaining: " + names(store.get(ids=["demo:p1", "demo:p2", "demo:alice"])))
        store.structured_query('MATCH (n:"__Node__") DETACH DELETE n')  # leave the scratch graph empty
        print("\n  ✓ upsert → get → get_triplets → delete verified (arxiv untouched)")
    finally:
        engine.close()


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
        read_back(store, hits)
        crud_lifecycle()
    finally:
        agens.close()


if __name__ == "__main__":
    main()
