"""arXiv GraphRAG — ingest.

Builds, in ONE AgensGraph graph, a knowledge graph of papers, authors, categories and years, and
a vector store over the same Paper vertices:

    (:Paper {id,title,year,text})-[:AUTHORED_BY]->(:Author {name})
    (:Paper)-[:IN_CATEGORY]->(:Category {name})
    (:Paper)-[:UPDATED_IN]->(:Year {year})

The papers are the vector store's own chunks (label ``Paper``), so the graph links to them with
ordinary edges; their embeddings live in the graph's companion schema ``"arxiv_store"``, indexed
with HNSW. The authors, categories and years go in with ``AgensGraph.add_graph_documents``,
merged by id. Data: real arXiv metadata streamed from Hugging Face, or with ``DEMOS_FAKE=1`` the
50 invented papers of ``_common/fixtures/arxiv.jsonl``.

Embedding requests run concurrently (many in flight). A re-run embeds only papers not already
stored, so a larger ARXIV_LIMIT appends to an existing load.

    cd langchain
    .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py
    ARXIV_LIMIT=2000 .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py   # quick
    ARXIV_RESET=1    .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py   # rebuild

Knobs: ARXIV_LIMIT (default 50000), ARXIV_BATCH (papers per write, default 1000), ARXIV_RESET,
EMBED_CONCURRENCY (parallel embedding requests, default 10), EMBED_BATCH (texts per request,
default 100).
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

# Make the demos root importable (dir names like "01_..." aren't valid packages).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import Embeddings

from _common import agens, config, console
from _common.datautil import batched, env_int, fixture, stream_hf
from _common.models import get_embeddings
from langchain_agensgraph import AgensGraph, AgensgraphVector, GraphDocument, Node, Relationship

GRAPH = "arxiv"
LABEL = "Paper"
DATASET = "UniverseTBD/arxiv-abstracts-large"
MAX_AUTHORS = 20  # cap pathological author lists (some physics papers have 100s)


def normalize(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a raw arXiv record into {id,title,abstract,year,authors,categories}."""
    pid = (rec.get("id") or "").strip()
    title = " ".join((rec.get("title") or "").split())
    abstract = " ".join((rec.get("abstract") or "").split())
    if not pid or not title or not abstract:
        return None

    upd = rec.get("update_date")
    year = getattr(upd, "year", None)
    if year is None:  # some dumps carry a string
        try:
            year = int(str(upd)[:4])
        except (ValueError, TypeError):
            year = 0

    authors: List[str] = []
    for parts in (rec.get("authors_parsed") or [])[:MAX_AUTHORS]:
        name = " ".join(p for p in (parts or []) if p).strip()
        if name:
            authors.append(name)

    categories = [c for c in (rec.get("categories") or "").split() if c]

    return {
        "id": pid,
        "title": title,
        "abstract": abstract,
        "year": int(year),
        "authors": authors,
        "categories": categories,
    }


def records(limit: int) -> Iterable[Dict[str, Any]]:
    """Normalized papers: the fixture with DEMOS_FAKE=1, else the Hugging Face stream."""
    raw = fixture("arxiv", limit) if config.fake() else stream_hf(DATASET, limit=limit)
    return (paper for paper in map(normalize, raw) if paper)


def paper_store(embedding: Embeddings, graph_name: str = GRAPH, **kwargs: Any) -> AgensgraphVector:
    """The vector store whose chunks are the Paper vertices.

    ``filter_keys`` gives the arXiv id and the year an index each, for filtered searches.
    """
    return agens.make_vector(
        embedding,
        graph_name=graph_name,
        label=LABEL,
        filter_keys=("id", "year"),
        **kwargs,
    )


def paper_text(paper: Dict[str, Any]) -> str:
    return f"title: {paper['title']}\nabstract: {paper['abstract']}"


def paper_document(paper: Dict[str, Any]) -> GraphDocument:
    """The paper's authors, categories and year, and its edges to them.

    The Paper vertex itself is the vector store's; ``add_graph_documents`` finds it again by
    ``id`` and only adds the edges.
    """
    node = Node(id=paper["id"], type=LABEL)
    authors = [Node(id=name, type="Author", properties={"name": name}) for name in paper["authors"]]
    categories = [
        Node(id=name, type="Category", properties={"name": name}) for name in paper["categories"]
    ]
    year = Node(id=paper["year"], type="Year", properties={"year": paper["year"]})
    relationships = (
        [Relationship(source=node, target=a, type="AUTHORED_BY") for a in authors]
        + [Relationship(source=node, target=c, type="IN_CATEGORY") for c in categories]
        + [Relationship(source=node, target=year, type="UPDATED_IN")]
    )
    return GraphDocument(nodes=[*authors, *categories, year], relationships=relationships)


def embed_all(
    embed: Embeddings, texts: List[str], concurrency: int, sub_batch: int
) -> List[List[float]]:
    """Embed texts with ``concurrency`` requests in flight, ``sub_batch`` texts each, in order."""
    parts = [texts[i : i + sub_batch] for i in range(0, len(texts), sub_batch)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return [vector for part in pool.map(embed.embed_documents, parts) for vector in part]


def ingest(
    graph: AgensGraph,
    store: AgensgraphVector,
    papers: Iterable[Dict[str, Any]],
    *,
    batch_size: int = 1000,
    concurrency: int = 10,
    sub_batch: int = 100,
) -> Dict[str, int]:
    """Store each batch's new papers as chunks, then write their graph around them.

    A paper already stored is not embedded again. Database time is reported apart from the wall
    clock, which embedding and the Hugging Face stream dominate.
    """
    n_papers = n_new = n_edges = 0
    db_seconds = 0.0
    wall_start = time.perf_counter()
    for chunk in batched(papers, batch_size):
        t0 = time.perf_counter()
        stored = {doc.id for doc in store.get_by_ids([p["id"] for p in chunk])}
        db_seconds += time.perf_counter() - t0
        new = [p for p in chunk if p["id"] not in stored]
        if new:
            texts = [paper_text(p) for p in new]
            vectors = embed_all(store.embeddings, texts, concurrency, sub_batch)
            t0 = time.perf_counter()
            store.add_embeddings(
                texts,
                vectors,
                metadatas=[{"id": p["id"], "title": p["title"], "year": p["year"]} for p in new],
                ids=[p["id"] for p in new],
            )
            db_seconds += time.perf_counter() - t0
        documents = [paper_document(p) for p in chunk]
        t0 = time.perf_counter()
        graph.add_graph_documents(documents, batch_size=batch_size)
        db_seconds += time.perf_counter() - t0

        n_papers += len(chunk)
        n_new += len(new)
        n_edges += sum(len(d.relationships) for d in documents)
        if n_papers % (batch_size * 10) == 0:
            print(f"    ... {n_papers:,} papers")

    wall = time.perf_counter() - wall_start
    if db_seconds:
        print(
            f"  DB ingest:   {n_papers:,} papers + {n_edges:,} edges in {db_seconds:.2f}s "
            f"({n_papers / db_seconds:,.0f} papers/s, {n_edges / db_seconds:,.0f} edges/s)"
        )
    print(f"  wall (incl. embedding and streaming): {wall:.1f}s")
    return {"papers": n_papers, "embedded": n_new, "edges": n_edges}


def main() -> None:
    limit = env_int("ARXIV_LIMIT", 50000)
    batch_size = env_int("ARXIV_BATCH", 1000)
    concurrency = env_int("EMBED_CONCURRENCY", 10)
    sub_batch = env_int("EMBED_BATCH", 100)
    config.require_openai_key()

    console.section(f"arXiv GraphRAG — ingest  (ARXIV_LIMIT={limit:,})")
    graph = agens.make_graph(GRAPH, create=True)
    try:
        if os.getenv("ARXIV_RESET"):
            graph.drop_graph()
            print(f"[reset] dropped graph {GRAPH!r} and its companion schema")

        # index_build="manual": the HNSW index is built once, after the load, instead of being
        # kept up to date while every batch arrives.
        store = paper_store(
            get_embeddings(max_retries=10, timeout=60), create_graph=True, index_build="manual"
        )

        console.sub("embedding papers, writing chunks and graph documents")
        counts = ingest(
            graph,
            store,
            records(limit),
            batch_size=batch_size,
            concurrency=concurrency,
            sub_batch=sub_batch,
        )
        console.table(list(counts.items()), headers=["what", "count"])

        # build_index sizes maintenance_work_mem for the build itself.
        with console.timer("HNSW index build"):
            store.build_index()

        console.sub("done")
        total_nodes = graph.query("MATCH (n) RETURN count(*) AS c")[0]["c"]
        total_edges = graph.query("MATCH ()-[r]->() RETURN count(*) AS c")[0]["c"]
        console.table(
            [["nodes", f"{total_nodes:,}"], ["edges", f"{total_edges:,}"]],
            headers=["total", "count"],
        )
        print("\nNext:  .venv/bin/python examples/demos/01_arxiv_graphrag/query.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
